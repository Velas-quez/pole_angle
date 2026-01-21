import argparse
import os
from itertools import chain
from typing import Iterable, List, Optional, Tuple

import numpy as np
import rosbag

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pa = None
    pq = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


OPTITRACK_DEFAULT_TOPICS = [
    "/optitrack/pole/pose",
    "/optitrack/pole/euler",
    "/optitrack/pole/marker_error",
    "/optitrack/pole/tracking_valid",
    "/optitrack/base/pose",
    "/optitrack/base/euler",
    "/optitrack/base/marker_error",
    "/optitrack/base/tracking_valid",
]


def _find_bag_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.endswith(".bag")
        )
    return [input_path]


def _extract_image_message(msg):
    if hasattr(msg, "image"):
        return msg.image
    return msg


def _get_image_shape(image_msg, fallback_height: Optional[int], fallback_width: Optional[int]) -> Tuple[int, int]:
    height = getattr(image_msg, "height", None)
    width = getattr(image_msg, "width", None)

    if height is None:
        height = fallback_height
    if width is None:
        width = fallback_width

    if height is None or width is None:
        raise ValueError("Image height/width not found; provide --height and --width.")

    return int(height), int(width)


def _msg_timestamp(msg, t):
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "stamp"):
        return header.stamp.to_sec()
    return t.to_sec()


def _iter_sparse_rows(
    bag_path: str,
    topic: str,
    background_value: int,
    fallback_height: Optional[int],
    fallback_width: Optional[int],
    max_messages: Optional[int],
) -> Iterable[dict]:
    bag = rosbag.Bag(bag_path)
    count = 0
    try:
        for msg_topic, msg, t in bag.read_messages(topics=[topic]):
            image_msg = _extract_image_message(msg)
            height, width = _get_image_shape(image_msg, fallback_height, fallback_width)

            data = np.frombuffer(image_msg.data, dtype=np.uint8)
            if data.size != height * width:
                raise ValueError(
                    f"Unexpected image size ({data.size}); expected {height * width} from {height}x{width}."
                )

            flat = data.ravel()
            mask = flat != background_value
            indices = np.flatnonzero(mask)
            values = flat[indices]

            yield {
                "timestamp": _msg_timestamp(msg, t),
                "height": height,
                "width": width,
                "indices": indices.astype(np.int32).tolist(),
                "values": values.astype(np.uint8).tolist(),
                "nnz": int(indices.size),
            }

            count += 1
            if max_messages is not None and count >= max_messages:
                break
    finally:
        bag.close()


def _iter_optitrack_rows(
    bag_path: str,
    topics: List[str],
    max_messages: Optional[int],
) -> Iterable[dict]:
    bag = rosbag.Bag(bag_path)
    count = 0
    try:
        for msg_topic, msg, t in bag.read_messages(topics=topics):
            row = _optitrack_row(msg_topic, msg, t)
            if row is None:
                continue

            yield row

            count += 1
            if max_messages is not None and count >= max_messages:
                break
    finally:
        bag.close()


def _optitrack_row(msg_topic, msg, t) -> Optional[dict]:
    row = {
        "timestamp": _msg_timestamp(msg, t),
        "topic": msg_topic,
        "frame_id": None,
        "pos_x": None,
        "pos_y": None,
        "pos_z": None,
        "ori_x": None,
        "ori_y": None,
        "ori_z": None,
        "ori_w": None,
        "vec_x": None,
        "vec_y": None,
        "vec_z": None,
        "value_float": None,
        "value_bool": None,
    }

    header = getattr(msg, "header", None)
    if header is not None:
        row["frame_id"] = getattr(header, "frame_id", None)

    if hasattr(msg, "pose"):
        pose = msg.pose
        row["pos_x"] = pose.position.x
        row["pos_y"] = pose.position.y
        row["pos_z"] = pose.position.z
        row["ori_x"] = pose.orientation.x
        row["ori_y"] = pose.orientation.y
        row["ori_z"] = pose.orientation.z
        row["ori_w"] = pose.orientation.w
        return row

    if hasattr(msg, "vector"):
        vector = msg.vector
        row["vec_x"] = vector.x
        row["vec_y"] = vector.y
        row["vec_z"] = vector.z
        return row

    if hasattr(msg, "data"):
        data = msg.data
        if isinstance(data, bool):
            row["value_bool"] = bool(data)
        elif isinstance(data, (int, float)):
            row["value_float"] = float(data)
        return row

    return None


def _write_parquet_stream(rows: Iterable[dict], output_path: str, chunk_size: int) -> None:
    if pq is None or pa is None:
        raise RuntimeError("pyarrow is required for streaming parquet output.")

    schema = pa.schema(
        [
            ("timestamp", pa.float64()),
            ("height", pa.int32()),
            ("width", pa.int32()),
            ("indices", pa.list_(pa.int32())),
            ("values", pa.list_(pa.uint8())),
            ("nnz", pa.int32()),
        ]
    )

    writer = None
    batch = []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= chunk_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema)
                writer.write_table(table)
                batch = []

        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _write_optitrack_parquet_stream(rows: Iterable[dict], output_path: str, chunk_size: int) -> None:
    if pq is None or pa is None:
        raise RuntimeError("pyarrow is required for streaming parquet output.")

    schema = pa.schema(
        [
            ("timestamp", pa.float64()),
            ("topic", pa.string()),
            ("frame_id", pa.string()),
            ("pos_x", pa.float64()),
            ("pos_y", pa.float64()),
            ("pos_z", pa.float64()),
            ("ori_x", pa.float64()),
            ("ori_y", pa.float64()),
            ("ori_z", pa.float64()),
            ("ori_w", pa.float64()),
            ("vec_x", pa.float64()),
            ("vec_y", pa.float64()),
            ("vec_z", pa.float64()),
            ("value_float", pa.float64()),
            ("value_bool", pa.bool_()),
        ]
    )

    writer = None
    batch = []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= chunk_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema)
                writer.write_table(table)
                batch = []

        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _write_parquet_pandas(rows: List[dict], output_path: str) -> None:
    if pd is None:
        raise RuntimeError("pandas is required for parquet output when pyarrow is unavailable.")
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)


def convert_bag_to_sparse_parquet(
    bag_path: str,
    output_dir: str,
    topic: str,
    background_value: int = 127,
    fallback_height: Optional[int] = None,
    fallback_width: Optional[int] = None,
    max_messages: Optional[int] = None,
    chunk_size: int = 1000,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(bag_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_sparse.parquet")

    rows = _iter_sparse_rows(
        bag_path,
        topic,
        background_value,
        fallback_height,
        fallback_width,
        max_messages,
    )

    if pq is not None and pa is not None:
        _write_parquet_stream(rows, output_path, chunk_size)
    else:
        _write_parquet_pandas(list(rows), output_path)

    return output_path


def convert_bag_to_optitrack_parquet(
    bag_path: str,
    output_dir: str,
    topics: List[str],
    max_messages: Optional[int] = None,
    chunk_size: int = 1000,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(bag_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_optitrack.parquet")

    rows_iter = _iter_optitrack_rows(bag_path, topics, max_messages)
    rows_iter = iter(rows_iter)
    first_row = next(rows_iter, None)
    if first_row is None:
        return None

    rows = chain([first_row], rows_iter)

    if pq is not None and pa is not None:
        _write_optitrack_parquet_stream(rows, output_path, chunk_size)
    else:
        _write_parquet_pandas(list(rows), output_path)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert evetac image data to sparse vectors and export optitrack topics from ROS bags."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="raw_data",
        help="Bag file or directory containing .bag files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="sparse_data",
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--topic",
        default="/evetac_msg0",
        help="ROS topic containing the image messages.",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=127,
        help="Pixel value treated as background (not stored).",
    )
    parser.add_argument("--height", type=int, default=None, help="Fallback image height.")
    parser.add_argument("--width", type=int, default=None, help="Fallback image width.")
    parser.add_argument("--max-messages", type=int, default=None, help="Limit number of frames.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows per parquet write when using pyarrow streaming.",
    )
    parser.add_argument(
        "--optitrack-topics",
        default=",".join(OPTITRACK_DEFAULT_TOPICS),
        help="Comma-separated list of optitrack topics to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bag_files = _find_bag_files(args.input)
    if not bag_files:
        raise SystemExit(f"No .bag files found in {args.input}")

    for bag_path in bag_files:
        output_path = convert_bag_to_sparse_parquet(
            bag_path=bag_path,
            output_dir=args.output_dir,
            topic=args.topic,
            background_value=args.background_value,
            fallback_height=args.height,
            fallback_width=args.width,
            max_messages=args.max_messages,
            chunk_size=args.chunk_size,
        )
        print(f"Wrote {output_path}")

        optitrack_topics = [t.strip() for t in args.optitrack_topics.split(",") if t.strip()]
        optitrack_path = convert_bag_to_optitrack_parquet(
            bag_path=bag_path,
            output_dir=args.output_dir,
            topics=optitrack_topics,
            max_messages=args.max_messages,
            chunk_size=args.chunk_size,
        )
        if optitrack_path:
            print(f"Wrote {optitrack_path}")


if __name__ == "__main__":
    main()
