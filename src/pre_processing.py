import os
from typing import Dict, Iterable, List, Optional
import numpy as np
import rosbag

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    import pandas as pd
except ImportError:
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
    bag_name = os.path.basename(bag_path)
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
                    "bag_name": bag_name,
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

    timestamps = sorted(set(pose_by_ts.keys()) & set(vec_by_ts.keys()))
    for ts in timestamps:
        row = {**pose_by_ts[ts], **vec_by_ts[ts]}
        yield row


def _collect_evtac_rows(
    bag_path: str,
    topic: str,
    background_value: int = 127,
    fallback_height: Optional[int] = None,
    fallback_width: Optional[int] = None,
    max_messages: Optional[int] = None,
) -> Iterable[Dict]:
    bag_name = os.path.basename(bag_path)
    bag = rosbag.Bag(bag_path)
    count = 0
    try:
        for msg_topic, msg, t in bag.read_messages(topics=[topic]):
            image_msg = _extract_image_message(msg)
            height, width = _get_image_shape(image_msg, fallback_height, fallback_width)

            data = np.frombuffer(image_msg.data, dtype=np.uint8)
            flat = data.ravel()
            mask = flat != background_value
            indices = np.flatnonzero(mask)
            values = flat[indices]

            yield {
                "timestamp": _msg_timestamp(msg, t),
                "bag_name": bag_name,
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
    bag_name = os.path.basename(bag_path)
    bag = rosbag.Bag(bag_path)
    count = 0
    try:
        for msg_topic, msg, t in bag.read_messages(topics=[topic]):
            data = getattr(msg, "data", None)
            if data is None:
                continue

            yield {
                "timestamp": (
                    msg.header.stamp.to_sec() if hasattr(msg, "header") else t.to_sec()
                ),
                "bag_name": bag_name,
                "markers": list(data),
            }

            count += 1
            if max_messages is not None and count >= max_messages:
                break
    finally:
        bag.close()


# --- Helpers e Funções de Escrita com Schema Atualizado ---


def _extract_image_message(msg):
    return msg.image if hasattr(msg, "image") else msg


def _get_image_shape(image_msg, fallback_height, fallback_width):
    h = getattr(image_msg, "height", fallback_height)
    w = getattr(image_msg, "width", fallback_width)
    if h is None or w is None:
        raise ValueError("Image dimensions missing.")
    return int(h), int(w)


def _write_or_append_parquet(rows: List[Dict], output_path: str, schema=None) -> str:
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
            df_ex = pd.read_parquet(output_path)
            df_new = pd.concat([df_ex, df_new], ignore_index=True)
        df_new.to_parquet(output_path, index=False)
    return output_path


def _write_optitrack_parquet(rows: List[Dict], output_path: str) -> str:
    schema = None
    if pq and pa:
        schema = pa.schema(
            [
                ("timestamp", pa.float64()),
                ("bag_name", pa.string()),
                (
                    "pos",
                    pa.struct(
                        [("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())]
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
                        [("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())]
                    ),
                ),
            ]
        )
    return _write_or_append_parquet(rows, output_path, schema)


def _write_markers_parquet(rows: List[Dict], output_path: str) -> str:
    schema = None
    if pq and pa:
        schema = pa.schema(
            [
                ("timestamp", pa.float64()),
                ("bag_name", pa.string()),
                ("markers", pa.list_(pa.float64())),
            ]
        )
    return _write_or_append_parquet(rows, output_path, schema)


def _write_evtac_parquet(rows: List[Dict], output_path: str) -> str:
    schema = None
    if pq and pa:
        schema = pa.schema(
            [
                ("timestamp", pa.float64()),
                ("bag_name", pa.string()),
                ("height", pa.int32()),
                ("width", pa.int32()),
                ("indices", pa.list_(pa.int32())),
                ("values", pa.list_(pa.uint8())),
                ("nnz", pa.int32()),
            ]
        )
    return _write_or_append_parquet(rows, output_path, schema)


# --- Funções de Orquestração ---
def preprocess_data(bag_path, output_dir, **kwargs) -> Dict[str, str]:
    # Extraímos o use_base_name para que ele não vá para os 'collects'
    use_base = kwargs.pop("use_base_name", True)
    max_msgs = kwargs.get("max_messages")

    base = os.path.splitext(os.path.basename(bag_path))[0]

    # Agora o **kwargs não contém mais o 'use_base_name'
    opti_rows = list(_collect_optitrack_rows(bag_path, OPTITRACK_TOPICS, max_msgs))
    mark_rows = list(_collect_markers_rows(bag_path, MARKERS_TOPIC, max_msgs))
    evta_rows = list(_collect_evtac_rows(bag_path, EVTAC_TOPIC, **kwargs))

    res = {}
    for rows, name, func in [
        (opti_rows, "optitrack", _write_optitrack_parquet),
        (mark_rows, "markers", _write_markers_parquet),
        (evta_rows, "evtac", _write_evtac_parquet),
    ]:
        if rows:
            fname = f"{base}_{name}.parquet" if use_base else f"{name}.parquet"
            path = os.path.join(output_dir, fname)
            res[name] = func(rows, path)
        else:
            res[name] = ""
    return res


def preprocess_directory(raw_dir, output_dir, **kwargs) -> Dict[str, str]:
    if not os.path.exists(raw_dir):
        print(f"Erro: Diretorio {raw_dir} nao encontrado.")
        return {}

    bags = sorted(
        [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".bag")]
    )
    res = {"optitrack": "", "markers": "", "evtac": ""}

    for b in bags:
        print(f"Processando bag: {os.path.basename(b)}")
        # Passamos use_base_name=False para agrupar tudo em um único arquivo
        out = preprocess_data(b, output_dir, use_base_name=False, **kwargs)
        for k in res:
            if out.get(k):
                res[k] = out[k]
    return res


if __name__ == "__main__":
    mode = "train"  # "train"/"test"
    raw_dir = os.path.join("data", mode, "raw_babs_new")
    output_dir = os.path.join("data", mode, "preprocessed_babs_new")
    preprocess_directory(raw_dir, output_dir)
