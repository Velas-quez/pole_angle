import os
from typing import Dict, Iterable, List, Optional

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

OPTITRACK_TOPICS = [
    "/optitrack/base/pose",
    "/optitrack/base/euler",
]

MARKERS_TOPIC = "/optitrack/pole/markers"

EVTAC_TOPIC = "/evetac_msg0"


def _msg_timestamp(msg, t):
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "stamp"):
        return header.stamp.to_sec()
    return t.to_sec()


def _collect_optitrack_rows(
    bag_path: str, topics: List[str], max_messages: Optional[int] = None
) -> Iterable[Dict]:
    """
    Reads the pose and euler topics (published at the same timestamp) and merges them
    into a single row containing timestamp plus nested fields:
    - pos: {x, y, z}
    - orientation: {x, y, z, w}
    - vector: {x, y, z}  (from /euler topic)
    """
    pose_topic = topics[0]
    euler_topic = topics[1]

    pose_by_ts: Dict[float, Dict] = {}
    vec_by_ts: Dict[float, Dict] = {}

    bag = rosbag.Bag(bag_path)
    count = 0
    try:
        for msg_topic, msg, t in bag.read_messages(topics=[pose_topic, euler_topic]):
            ts = msg.header.stamp.to_sec()

            if msg_topic == pose_topic and hasattr(msg, "pose"):
                pose = msg.pose
                pose_by_ts[ts] = {
                    "timestamp": ts,
                    "pos": {
                        "x": pose.position.x,
                        "y": pose.position.y,
                        "z": pose.position.z,
                    },
                    "orientation": {
                        "x": pose.orientation.x,
                        "y": pose.orientation.y,
                        "z": pose.orientation.z,
                        "w": pose.orientation.w,
                    },
                }
            elif msg_topic == euler_topic and hasattr(msg, "vector"):
                vec = msg.vector
                vec_by_ts[ts] = {
                    "vector": {
                        "x": vec.x,
                        "y": vec.y,
                        "z": vec.z,
                    }
                }

            count += 1
            if max_messages is not None and count >= max_messages:
                break
    finally:
        bag.close()

    # Merge by exact timestamp (topics are emitted together according to user note)
    timestamps = sorted(set(pose_by_ts.keys()) & set(vec_by_ts.keys()))
    for ts in timestamps:
        row = {**pose_by_ts[ts], **vec_by_ts[ts]}
        yield row


def _extract_image_message(msg):
    if hasattr(msg, "image"):
        return msg.image
    return msg


def _get_image_shape(
    image_msg, fallback_height: Optional[int], fallback_width: Optional[int]
):
    height = getattr(image_msg, "height", None)
    width = getattr(image_msg, "width", None)

    if height is None:
        height = fallback_height
    if width is None:
        width = fallback_width

    if height is None or width is None:
        raise ValueError(
            "Image height/width not found; provide fallback_height and fallback_width."
        )

    return int(height), int(width)


def _collect_evtac_rows(
    bag_path: str,
    topic: str,
    background_value: int = 127,
    fallback_height: Optional[int] = None,
    fallback_width: Optional[int] = None,
    max_messages: Optional[int] = None,
) -> Iterable[Dict]:
    """
    Converts image-like evtac messages into sparse vectors (indices + values) keeping only non-background pixels.
    Mirrors the logic previously used in data_sparsing.
    """
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


def _collect_markers_rows(
    bag_path: str, topic: str, max_messages: Optional[int] = None
) -> Iterable[Dict]:
    """
    Reads the markers topic and stores the raw marker vector exactly as published.
    """
    bag = rosbag.Bag(bag_path)
    count = 0
    try:
        for msg_topic, msg, t in bag.read_messages(topics=[topic]):
            data = getattr(msg, "data", None)
            if data is None:
                continue

            yield {
                "timestamp": msg.header.stamp.to_sec()
                if hasattr(msg, "header")
                else t.to_sec(),
                "markers": list(data),
            }

            count += 1
            if max_messages is not None and count >= max_messages:
                break
    finally:
        bag.close()


def _write_or_append_parquet(rows: List[Dict], output_path: str, schema=None) -> str:
    """
    Writes rows to a parquet file, appending to it if it already exists.
    Uses pyarrow when available; falls back to pandas otherwise.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not rows:
        return output_path

    if pq is not None and pa is not None:
        table = pa.Table.from_pylist(rows, schema=schema)
        if os.path.exists(output_path):
            existing = pq.read_table(output_path)
            table = pa.concat_tables([existing, table], promote=True)
        pq.write_table(table, output_path)
    elif pd is not None:
        df_new = pd.DataFrame(rows)
        if os.path.exists(output_path):
            df_existing = pd.read_parquet(output_path)
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        df_new.to_parquet(output_path, index=False)
    else:  # pragma: no cover - extremely unlikely given requirements.txt
        raise RuntimeError(
            "Neither pyarrow nor pandas is available to write parquet files."
        )

    return output_path


def _write_optitrack_parquet(rows: List[Dict], output_path: str) -> str:
    schema = None
    if pq is not None and pa is not None:
        schema = pa.schema(
            [
                ("timestamp", pa.float64()),
                (
                    "pos",
                    pa.struct(
                        [
                            ("x", pa.float64()),
                            ("y", pa.float64()),
                            ("z", pa.float64()),
                        ]
                    ),
                ),
                (
                    "orientation",
                    pa.struct(
                        [
                            ("x", pa.float64()),
                            ("y", pa.float64()),
                            ("z", pa.float64()),
                            ("w", pa.float64()),
                        ]
                    ),
                ),
                (
                    "vector",
                    pa.struct(
                        [
                            ("x", pa.float64()),
                            ("y", pa.float64()),
                            ("z", pa.float64()),
                        ]
                    ),
                ),
            ]
        )

    return _write_or_append_parquet(rows, output_path, schema)


def _write_markers_parquet(rows: List[Dict], output_path: str) -> str:
    schema = None
    if pq is not None and pa is not None:
        schema = pa.schema(
            [
                ("timestamp", pa.float64()),
                ("markers", pa.list_(pa.float64())),
            ]
        )

    return _write_or_append_parquet(rows, output_path, schema)


def _write_evtac_parquet(rows: List[Dict], output_path: str) -> str:
    schema = None
    if pq is not None and pa is not None:
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

    return _write_or_append_parquet(rows, output_path, schema)


def preprocess_data(
    bag_path: str,
    output_dir: str = "processed_data",
    max_messages: Optional[int] = None,
    background_value: int = 127,
    fallback_height: Optional[int] = None,
    fallback_width: Optional[int] = None,
    use_base_name: bool = True,
) -> Dict[str, str]:
    """
    Reads a rosbag and writes parquet outputs for each group of topics.

    When use_base_name is False, writes/append to shared files:
    - optitrack.parquet
    - markers.parquet
    - evtac.parquet
    """
    base_name = os.path.splitext(os.path.basename(bag_path))[0]
    optitrack_rows = list(
        _collect_optitrack_rows(bag_path, OPTITRACK_TOPICS, max_messages)
    )
    markers_rows = list(_collect_markers_rows(bag_path, MARKERS_TOPIC, max_messages))
    evtac_rows = list(
        _collect_evtac_rows(
            bag_path,
            EVTAC_TOPIC,
            background_value=background_value,
            fallback_height=fallback_height,
            fallback_width=fallback_width,
            max_messages=max_messages,
        )
    )

    outputs: Dict[str, str] = {}
    if optitrack_rows:
        filename = (
            f"{base_name}_optitrack.parquet" if use_base_name else "optitrack.parquet"
        )
        opti_path = os.path.join(output_dir, filename)
        outputs["optitrack"] = _write_optitrack_parquet(optitrack_rows, opti_path)
    else:
        outputs["optitrack"] = ""

    if markers_rows:
        filename = (
            f"{base_name}_markers.parquet" if use_base_name else "markers.parquet"
        )
        markers_path = os.path.join(output_dir, filename)
        outputs["markers"] = _write_markers_parquet(markers_rows, markers_path)
    else:
        outputs["markers"] = ""

    if evtac_rows:
        filename = f"{base_name}_evtac.parquet" if use_base_name else "evtac.parquet"
        evtac_path = os.path.join(output_dir, filename)
        outputs["evtac"] = _write_evtac_parquet(evtac_rows, evtac_path)
    else:
        outputs["evtac"] = ""

    return outputs


def preprocess_directory(
    raw_data_dir: str = "raw_data",
    output_dir: str = "processed_data",
    max_messages: Optional[int] = None,
    background_value: int = 127,
    fallback_height: Optional[int] = None,
    fallback_width: Optional[int] = None,
) -> Dict[str, str]:
    """
    Processes every .bag in raw_data_dir and appends each dataset (optitrack, markers, evtac)
    into three aggregated parquet files stored under output_dir.
    """
    bag_files = sorted(
        os.path.join(raw_data_dir, f)
        for f in os.listdir(raw_data_dir)
        if f.endswith(".bag")
    )

    results: Dict[str, str] = {"optitrack": "", "markers": "", "evtac": ""}
    if not bag_files:
        return results

    for bag_path in bag_files:
        bag_outputs = preprocess_data(
            bag_path,
            output_dir=output_dir,
            max_messages=max_messages,
            background_value=background_value,
            fallback_height=fallback_height,
            fallback_width=fallback_width,
            use_base_name=False,
        )
        for key in results.keys():
            if bag_outputs.get(key):
                results[key] = bag_outputs[key]

    return results


if __name__ == "__main__":
    raw_dir = os.path.join("data", "test", "raw")
    output_dir = os.path.join("data", "test", "preprocessed")
    result_paths = preprocess_directory(
        raw_data_dir=raw_dir,
        output_dir=output_dir,
        max_messages=None,
    )

    for kind, path in result_paths.items():
        if path:
            print(f"Wrote {kind} parquet: {path}")
        else:
            print(f"No data found for {kind}; file not written.")
