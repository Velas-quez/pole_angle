from __future__ import annotations

from typing import List, Optional

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from ..base_processing import BaseStrategy


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

    inv_m00_2 = 1.0 / (float(n_events) ** 2 + 1e-7)
    inv_m00_2_5 = 1.0 / (float(n_events) ** 2.5 + 1e-7)

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


def process_single_evtac_row(row: pd.Series, quad_w: int, quad_h: int) -> List[float]:
    """Calculates statistics for each quadrant of the EveTac sensor."""
    indices: np.ndarray = np.array(row["indices"])
    values: np.ndarray = np.array(row["values"])
    width, height = row["width"], row["height"]

    y_coords_all: np.ndarray = indices // width
    x_coords_all: np.ndarray = indices % width

    all_stats = [row["timestamp"]]

    for y_start in range(0, height, quad_h):
        for x_start in range(0, width, quad_w):
            in_quad = (
                (x_coords_all >= x_start)
                & (x_coords_all < x_start + quad_w)
                & (y_coords_all >= y_start)
                & (y_coords_all < y_start + quad_h)
            )

            x_q, y_q, v_q = (
                x_coords_all[in_quad],
                y_coords_all[in_quad],
                values[in_quad],
            )
            pos_mask, neg_mask = v_q == 255, v_q == 0

            quad_feats = [
                np.mean(x_q) if len(x_q) > 0 else 0.0,
                np.mean(y_q) if len(x_q) > 0 else 0.0,
                np.var(x_q) if len(x_q) > 0 else 0.0,
                np.var(y_q) if len(x_q) > 0 else 0.0,
                float(np.sum(pos_mask)),
                float(np.sum(neg_mask)),
            ]

            quad_feats.extend(
                get_event_hu_moments(x_q[pos_mask], y_q[pos_mask]).tolist()
            )
            quad_feats.extend(
                get_event_hu_moments(x_q[neg_mask], y_q[neg_mask]).tolist()
            )
            all_stats.extend(quad_feats)

    return all_stats


def estimate_pole_vector_from_row(row: pd.Series) -> np.ndarray:
    pts: np.ndarray = np.array(row["markers"]).reshape(-1, 3)
    if len(pts) < 2:
        return np.array([row["timestamp"], 0.0, 0.0, 1.0])

    centered: np.ndarray = pts - np.mean(pts, axis=0)
    _, _, Vh = np.linalg.svd(centered)
    vec: np.ndarray = Vh[0]
    if vec[2] < 0:
        vec = -vec
    return np.array([row["timestamp"], vec[0], vec[1], vec[2]])


def extract_and_prepare_parquet(
    evtac_path: str,
    markers_path: str,
    opti_path: str,
    quad_width: float,
    quad_height: float,
    output_name: Optional[str] = None,
) -> pd.DataFrame:
    # 1. Load Data
    df_evtac_full = pd.read_parquet(evtac_path)
    df_markers_full = pd.read_parquet(markers_path)
    df_opti_full = pd.read_parquet(opti_path)

    for df in [df_evtac_full, df_markers_full, df_opti_full]:
        df.rename(inplace=True, columns={"bagfile": "bag_name"})

    sensor_w = int(df_evtac_full.iloc[0]["width"])
    sensor_h = int(df_evtac_full.iloc[0]["height"])
    qw = int(quad_width) if quad_width >= 1 else int(sensor_w * quad_width)
    qh = int(quad_height) if quad_height >= 1 else int(sensor_h * quad_height)

    all_episodes_results = []
    bags = [b for b in df_evtac_full["bag_name"].unique() if pd.notna(b)]
    print(f"Iniciando processamento de {len(bags)} episódios sem Slerp...")

    for bag in bags:
        print(f"-> Processando bag: {bag}")
        df_evtac = df_evtac_full[df_evtac_full["bag_name"] == bag].copy()
        df_markers = df_markers_full[df_markers_full["bag_name"] == bag].copy()
        df_opti = df_opti_full[df_opti_full["bag_name"] == bag].copy()

        if df_opti.empty or df_evtac.empty:
            continue

        # Time Crop (2s)
        t_start = (
            max(
                df_evtac["timestamp"].min(),
                df_markers["timestamp"].min(),
                df_opti["timestamp"].min(),
            )
            + 2.0
        )
        t_end = (
            min(
                df_evtac["timestamp"].max(),
                df_markers["timestamp"].max(),
                df_opti["timestamp"].max(),
            )
            - 2.0
        )

        if t_start >= t_end:
            continue

        mask_fn = lambda df: df[
            (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
        ]
        df_evtac, df_markers, df_opti = (
            mask_fn(df_evtac),
            mask_fn(df_markers),
            mask_fn(df_opti),
        )

        # 3. Processamento de Features (Evtac)
        processed_evtac = [
            process_single_evtac_row(row, qw, qh)
            for _, row in df_evtac.iterrows()
            if row["nnz"] > 10
        ]
        if not processed_evtac:
            continue
        evtac_feat_arr = np.array(processed_evtac)
        target_timestamps = evtac_feat_arr[:, 0]

        # 4. Processamento de Marcadores (Pole Vector)
        processed_markers = [
            estimate_pole_vector_from_row(row) for _, row in df_markers.iterrows()
        ]
        marker_vec_arr = np.array(processed_markers)
        _, u_idx = np.unique(marker_vec_arr[:, 0], return_index=True)
        marker_vec_arr = marker_vec_arr[u_idx]

        # 5. INTERPOLAÇÃO SEM SLERP (Usando interp1d com extrapolação para tudo)
        df_opti = df_opti.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        base_quats = np.array(
            [[d["x"], d["y"], d["z"], d["w"]] for d in df_opti["orientation"]]
        )

        # Interpolação dos Quaternions (Linear)
        f_quat = interp1d(
            df_opti["timestamp"].values, base_quats, axis=0, fill_value="extrapolate"
        )
        interp_quats = f_quat(target_timestamps)

        # NORMALIZAÇÃO MANUAL: Essencial após interpolação linear de quaternions
        norms = np.linalg.norm(interp_quats, axis=1, keepdims=True)
        interp_quats = interp_quats / (norms + 1e-9)

        # Converte para objetos de rotação do SciPy
        sync_rots_base = R.from_quat(interp_quats)

        # Interpolação dos Marcadores
        f_marker_v = interp1d(
            marker_vec_arr[:, 0],
            marker_vec_arr[:, 1:],
            axis=0,
            fill_value="extrapolate",
        )
        sync_v_world = f_marker_v(target_timestamps)

        # 6. Transformação para Vetor Unitário Local
        final_vec_targets, base_eulers = [], []
        for i in range(len(target_timestamps)):
            v_local = sync_rots_base[i].inv().apply(sync_v_world[i])
            v_unit = v_local / (np.linalg.norm(v_local) + 1e-7)
            final_vec_targets.append(v_unit)
            base_eulers.append(sync_rots_base[i].as_euler("xyz", degrees=True))

        # 7. Colunas e DataFrame
        n_qx, n_qy = sensor_w // qw, sensor_h // qh
        cols = []
        for qy in range(n_qy):
            for qx in range(n_qx):
                s = f"q_{qx}_{qy}"
                cols += [
                    f"mu_x_{s}",
                    f"mu_y_{s}",
                    f"var_x_{s}",
                    f"var_y_{s}",
                    f"n_pos_{s}",
                    f"n_neg_{s}",
                ]
                cols += [f"hu_pos_{i}_{s}" for i in range(7)] + [
                    f"hu_neg_{i}_{s}" for i in range(7)
                ]

        df_bag = pd.DataFrame(evtac_feat_arr[:, 1:], columns=cols)
        df_bag[["base_roll", "base_pitch", "base_yaw"]] = np.array(base_eulers)
        df_bag[["target_x", "target_y", "target_z"]] = np.array(final_vec_targets)
        df_bag["source_bag"] = bag
        all_episodes_results.append(df_bag)

    if not all_episodes_results:
        print("Nenhum dado processado.")
        return pd.DataFrame()

    df_final = pd.concat(all_episodes_results, ignore_index=True)

    if output_name:
        df_final.to_parquet(output_name, index=False)
        print(f"\nFinal Parquet criado com sucesso: {output_name}")

    return df_final


class Strategy(BaseStrategy):
    def __init__(self, quad_width: float = 0.25, quad_height: float = 0.25):
        self.quad_width = quad_width
        self.quad_height = quad_height

    def transform(
        self,
        evtac_path: str,
        markers_path: str,
        opti_path: str,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        return extract_and_prepare_parquet(
            evtac_path=evtac_path,
            markers_path=markers_path,
            opti_path=opti_path,
            quad_width=self.quad_width,
            quad_height=self.quad_height,
            output_name=output_path,
        )
