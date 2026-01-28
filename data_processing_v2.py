import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict, Any


def get_event_hu_moments(x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """Calculates the 7 log-transformed Hu Moments for sparse event coordinates."""
    n_events: int = len(x_coords)
    if n_events < 10:
        return np.zeros(7)

    x, y = x_coords.astype(float), y_coords.astype(float)
    cx, cy = np.mean(x), np.mean(y)
    dx, dy = x - cx, y - cy

    mu20, mu02, mu11 = np.sum(dx**2), np.sum(dy**2), np.sum(dx * dy)
    mu30, mu03, mu21, mu12 = (
        np.sum(dx**3),
        np.sum(dy**3),
        np.sum((dx**2) * dy),
        np.sum(dx * (dy**2)),
    )

    inv_m00_2 = 1.0 / (float(n_events) ** 2)
    inv_m00_2_5 = 1.0 / (float(n_events) ** 2.5)

    e20, e02, e11 = mu20 * inv_m00_2, mu02 * inv_m00_2, mu11 * inv_m00_2
    e30, e03, e21, e12 = (
        mu30 * inv_m00_2_5,
        mu03 * inv_m00_2_5,
        mu21 * inv_m00_2_5,
        mu12 * inv_m00_2_5,
    )

    h1 = e20 + e02
    h2 = (e20 - e02) ** 2 + 4 * e11**2
    h3 = (e30 - 3 * e12) ** 2 + (3 * e21 - e03) ** 2
    h4 = (e30 + e12) ** 2 + (e21 + e03) ** 2
    h5 = (e30 - 3 * e12) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) + (
        3 * e21 - e03
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)
    h6 = (e20 - e02) * ((e30 + e12) ** 2 - (e21 + e03) ** 2) + 4 * e11 * (e30 + e12) * (
        e21 + e03
    )
    h7 = (3 * e21 - e03) * (e30 + e12) * ((e30 + e12) ** 2 - 3 * (e21 + e03) ** 2) - (
        e30 - 3 * e12
    ) * (e21 + e03) * (3 * (e30 + e12) ** 2 - (e21 + e03) ** 2)

    moments = np.array([h1, h2, h3, h4, h5, h6, h7])
    return -1 * np.sign(moments) * np.log10(np.abs(moments) + 1e-7)


def process_single_evtac_row(row: pd.Series) -> List[float]:
    """Reconstructs 2D coordinates from sparse indices and calculates row statistics."""
    indices: np.ndarray = np.array(row["indices"])
    values: np.ndarray = np.array(row["values"])
    width = row["width"]

    # Reconstruct 2D coordinates from 1D flat indices
    y_coords: np.ndarray = indices // width
    x_coords: np.ndarray = indices % width

    # Separate polarities (255 = positive, 0 = negative)
    pos_mask: np.ndarray = values == 255
    neg_mask: np.ndarray = values == 0

    # Base spatial features
    stats = [
        row["timestamp"],
        np.mean(x_coords),
        np.mean(x_coords[pos_mask]) if any(pos_mask) else 0.0,
        np.mean(x_coords[neg_mask]) if any(neg_mask) else 0.0,
        np.mean(y_coords),
        np.mean(y_coords[pos_mask]) if any(pos_mask) else 0.0,
        np.mean(y_coords[neg_mask]) if any(neg_mask) else 0.0,
        np.var(x_coords),
        np.var(y_coords),
        float(np.sum(pos_mask)),
        float(np.sum(neg_mask)),
    ]

    # Append Hu Moments for both polarities
    stats.extend(get_event_hu_moments(x_coords[pos_mask], y_coords[pos_mask]).tolist())
    stats.extend(get_event_hu_moments(x_coords[neg_mask], y_coords[neg_mask]).tolist())

    return stats


def estimate_pole_vector_from_row(row: pd.Series) -> np.ndarray:
    """Extracts the principal axis of pole markers via SVD."""
    pts: np.ndarray = np.array(row["markers"]).reshape(-1, 3)
    if len(pts) < 2:
        return np.array([row["timestamp"], 0.0, 0.0, 1.0])  # Default Z-up

    centered: np.ndarray = pts - np.mean(pts, axis=0)
    _, _, Vh = np.linalg.svd(centered)
    vec: np.ndarray = Vh[0]

    if vec[2] < 0:
        vec = -vec  # Ensure vector points towards Z+
    return np.array([row["timestamp"], vec[0], vec[1], vec[2]])


def transform_to_relative_angles(
    v_world: np.ndarray, q_base: np.ndarray
) -> Tuple[float, float]:
    """Transforms world vector to base local frame and returns Roll/Pitch."""
    r_base: R = R.from_quat(q_base)
    # v_local = R_inv * v_world
    v_local: np.ndarray = r_base.inv().apply(v_world)
    v_local /= np.linalg.norm(v_local)

    # Calculate angles relative to Base Z-axis [0,0,1]
    pitch: float = np.arccos(np.clip(v_local[2], -1.0, 1.0))
    roll: float = np.arctan2(v_local[1], v_local[0])
    return roll, pitch


def extract_and_prepare_parquet(
    evtac_path: str,
    markers_path: str,
    opti_path: str,
    output_name: str = "final_training_data.parquet",
) -> None:
    """Main pipeline to synchronize and process pre-processed Parquet files."""

    # 1. Load Data
    df_evtac: pd.DataFrame = pd.read_parquet(evtac_path)
    df_markers: pd.DataFrame = pd.read_parquet(markers_path)
    df_opti: pd.DataFrame = pd.read_parquet(opti_path)

    # 2. Process Evtac Features
    processed_evtac: List[List[float]] = [
        process_single_evtac_row(row)
        for _, row in df_evtac.iterrows()
        if row["nnz"] > 10
    ]
    evtac_feat_arr: np.ndarray = np.array(processed_evtac)

    # 3. Process Marker Vectors
    processed_markers: List[np.ndarray] = [
        estimate_pole_vector_from_row(row) for _, row in df_markers.iterrows()
    ]
    marker_vec_arr: np.ndarray = np.array(processed_markers)

    # 4. Interpolation and Synchronization
    # Convert nested dict 'orientation' to numpy [qx, qy, qz, qw]
    base_quats: np.ndarray = np.array(
        [[d["x"], d["y"], d["z"], d["w"]] for d in df_opti["orientation"]]
    )

    f_base_q = interp1d(
        df_opti["timestamp"],
        base_quats,
        axis=0,
        fill_value="extrapolate",  # type: ignore
    )
    f_marker_v = interp1d(
        marker_vec_arr[:, 0],
        marker_vec_arr[:, 1:],
        axis=0,
        fill_value="extrapolate",  # type: ignore
    )

    target_timestamps: np.ndarray = evtac_feat_arr[:, 0]
    sync_q_base: np.ndarray = f_base_q(target_timestamps)
    sync_v_world: np.ndarray = f_marker_v(target_timestamps)

    # 5. Transform and Generate Final Targets
    final_targets: List[Tuple[float, float]] = []
    base_eulers: List[np.ndarray] = []

    for i in range(len(target_timestamps)):
        r, p = transform_to_relative_angles(sync_v_world[i], sync_q_base[i])
        final_targets.append((r, p))
        base_eulers.append(R.from_quat(sync_q_base[i]).as_euler("xyz", degrees=True))

    # 6. Assemble and Export
    cols: List[str] = [
        "mu_x",
        "mu_x_pos",
        "mu_x_neg",
        "mu_y",
        "mu_y_pos",
        "mu_y_neg",
        "var_x",
        "var_y",
        "n_pos",
        "n_neg",
    ]
    cols += [f"hu_pos_{i}" for i in range(7)] + [f"hu_neg_{i}" for i in range(7)]

    df_final: pd.DataFrame = pd.DataFrame(evtac_feat_arr[:, 1:], columns=cols)
    df_final[["base_roll", "base_pitch", "base_yaw"]] = np.array(base_eulers)
    df_final[["target_roll", "target_pitch"]] = np.array(final_targets)

    df_final.to_parquet(output_name, index=False)
    print(f"Final Parquet created: {output_name} with {len(df_final)} rows.")


if __name__ == "__main__":
    # Example execution:
    extract_and_prepare_parquet(
        "preprocessed_evtac.parquet",
        "preprocessed_markers.parquet",
        "preprocessed_optitrack.parquet",
    )
    pass
