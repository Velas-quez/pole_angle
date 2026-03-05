from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from src.models.datasets import TactileSequenceDataset, TactileSparseSequenceDataset
from src.processing.strategies.base_strategy import BaseStrategy
from src.processing.strategies.statistical_multi_quads_strategy import (
    estimate_pole_vector_from_row,
)
from src.tactile_types import TactileData


def _normalize_quaternion(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)


def _vector_to_quaternion(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Quaternion (x, y, z, w) that rotates z-axis [0,0,1] to unit vector v.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = v / (np.linalg.norm(v) + eps)
    dot = float(np.dot(z_axis, v))

    if dot > 1.0 - 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if dot < -1.0 + 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axis = np.cross(z_axis, v)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + dot], dtype=np.float64)
    q = q / (np.linalg.norm(q) + eps)
    return q.astype(np.float32)


class TemporalEvtacPoseQuatStrategy(BaseStrategy):
    """
    Strategy for temporal CNN/TCN training with:
      - sparse EvTAC with optional materialization:
        * dense mode (materialize=True): saves (N, T, 2, H, W) -> faster training but larger disk usage
        * sparse mode (materialize=False): saves object sequences and densifies in Dataset -> slower training but smaller disk usage
      - base pose sequence (N, T, 7) where 7=(x,y,z,qx,qy,qz,qw)
      - target quaternion at current step (N, 4)
    """

    name: str = "temporal_evtac_pose_quat"

    def __init__(
        self,
        window_size: int = 10,
        downsample_factor: int = 4,
        crop_seconds: float = 2.0,
        min_nnz: int = 1,
        materialize_dense_evtac: bool = False,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")
        self.window_size = int(window_size)
        self.downsample_factor = int(downsample_factor)
        self.crop_seconds = float(crop_seconds)
        self.min_nnz = int(min_nnz)
        self.materialize_dense_evtac = bool(materialize_dense_evtac)

    def get_dirname(self) -> str:
        return (
            f"{self.name}_w{self.window_size}_ds{self.downsample_factor}"
            f"_minnnz{self.min_nnz}_sigdens2ch_"
            f"{'dense' if self.materialize_dense_evtac else 'sparse'}"
        )

    def transform(
        self, evtac_path: str, markers_path: str, opti_path: str, output_base_dir: str
    ) -> None:
        df_evtac_full = pd.read_parquet(evtac_path)
        df_markers_full = pd.read_parquet(markers_path)
        df_opti_full = pd.read_parquet(opti_path)

        for df in (df_evtac_full, df_markers_full, df_opti_full):
            if "bagfile" in df.columns:
                df.rename(inplace=True, columns={"bagfile": "bag_name"})

        bags = [b for b in df_evtac_full["bag_name"].unique() if pd.notna(b)]
        all_evtac_seq: List[np.ndarray] = []
        all_base_seq: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        print(
            f"🔄 Processing {len(bags)} bags with temporal windows (T={self.window_size}, ds={self.downsample_factor})..."
        )
        for bag in bags:
            df_evtac = (
                df_evtac_full[df_evtac_full["bag_name"] == bag]
                .copy()
                .sort_values("timestamp")
            )
            df_markers = (
                df_markers_full[df_markers_full["bag_name"] == bag]
                .copy()
                .sort_values("timestamp")
            )
            df_opti = (
                df_opti_full[df_opti_full["bag_name"] == bag]
                .copy()
                .sort_values("timestamp")
            )

            bag_evtac_seq, bag_base_seq, bag_targets = self._build_bag_sequences(
                df_evtac=df_evtac, df_markers=df_markers, df_opti=df_opti
            )
            if len(bag_targets) == 0:
                continue

            all_evtac_seq.append(bag_evtac_seq)
            all_base_seq.append(bag_base_seq)
            all_targets.append(bag_targets)

        output_dir = Path(output_base_dir) / self.get_dirname()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "data.npz"

        if not all_targets:
            np.savez_compressed(
                output_path,
                evetac=self._empty_evtac_array(),
                base=np.empty((0, self.window_size, 7), dtype=np.float32),
                target=np.empty((0, 4), dtype=np.float32),
                evetac_is_sparse=np.array(
                    [0 if self.materialize_dense_evtac else 1], dtype=np.int8
                ),
            )
            print(f"⚠️ No valid samples. Empty dataset saved to: {output_path}")
            return

        evetac_arr = self._concat_evtac_batches(all_evtac_seq)
        base_arr = np.concatenate(all_base_seq, axis=0).astype(np.float32)
        target_arr = np.concatenate(all_targets, axis=0).astype(np.float32)

        np.savez_compressed(
            output_path,
            evetac=evetac_arr,
            base=base_arr,
            target=target_arr,
            evetac_is_sparse=np.array(
                [0 if self.materialize_dense_evtac else 1], dtype=np.int8
            ),
        )
        print(
            f"✅ Saved temporal dataset in {output_path} | "
            f"evetac={evetac_arr.shape}, base={base_arr.shape}, target={target_arr.shape}"
        )

    def _build_bag_sequences(
        self, df_evtac: pd.DataFrame, df_markers: pd.DataFrame, df_opti: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if df_evtac.empty or df_markers.empty or df_opti.empty:
            return self._empty_bag_arrays()

        t_start = (
            max(
                df_evtac["timestamp"].min(),
                df_markers["timestamp"].min(),
                df_opti["timestamp"].min(),
            )
            + self.crop_seconds
        )
        t_end = (
            min(
                df_evtac["timestamp"].max(),
                df_markers["timestamp"].max(),
                df_opti["timestamp"].max(),
            )
            - self.crop_seconds
        )
        if t_start >= t_end:
            return self._empty_bag_arrays()

        mask = lambda df: df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)]
        df_evtac = mask(df_evtac)
        df_markers = mask(df_markers)
        df_opti = mask(df_opti)
        if df_evtac.empty or df_markers.empty or df_opti.empty:
            return self._empty_bag_arrays()

        # Marker direction interpolation in world frame
        marker_vec_arr = np.array(
            [estimate_pole_vector_from_row(row) for _, row in df_markers.iterrows()]
        )
        if marker_vec_arr.size == 0:
            return self._empty_bag_arrays()
        _, unique_idx = np.unique(marker_vec_arr[:, 0], return_index=True)
        marker_vec_arr = marker_vec_arr[unique_idx]

        marker_interp = interp1d(
            marker_vec_arr[:, 0],
            marker_vec_arr[:, 1:],
            axis=0,
            fill_value="extrapolate",
        )

        df_opti = df_opti.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        opti_ts = df_opti["timestamp"].to_numpy(dtype=np.float64)
        if opti_ts.size < 2:
            return self._empty_bag_arrays()

        base_pos = np.array(
            [[d["x"], d["y"], d["z"]] for d in df_opti["pos"]], dtype=np.float64
        )
        base_quat = np.array(
            [[d["x"], d["y"], d["z"], d["w"]] for d in df_opti["orientation"]],
            dtype=np.float64,
        )
        # Base orientation in x, y, z, considering the quaterion to rotate the vector (0,0,1)
        base_vec = R.from_quat(base_quat).apply(
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        pos_interp = interp1d(opti_ts, base_pos, axis=0, fill_value="extrapolate")
        quat_interp = interp1d(opti_ts, base_quat, axis=0, fill_value="extrapolate")
        basevec_interp = interp1d(opti_ts, base_vec, axis=0, fill_value="extrapolate")

        frame_evtac: List[Any] = []
        frame_base: List[np.ndarray] = []
        frame_target_quat: List[np.ndarray] = []

        for _, row in df_evtac.iterrows():
            if int(row.get("nnz", 0)) < self.min_nnz:
                continue

            ts = float(row["timestamp"])
            if self.materialize_dense_evtac:
                evtac_entry: Any = self._sparse_evtac_to_downsampled_image(row)
            else:
                evtac_entry = self._extract_sparse_evtac_frame(row)
            pos_t = np.asarray(pos_interp(ts), dtype=np.float64)
            quat_t = _normalize_quaternion(
                np.asarray(quat_interp(ts), dtype=np.float64)
            )
            basevec_t = np.asarray(basevec_interp(ts), dtype=np.float64)
            rot_base = R.from_quat(quat_t)

            world_vec = np.asarray(marker_interp(ts), dtype=np.float64)
            world_norm = np.linalg.norm(world_vec)
            if world_norm < 1e-8:
                continue
            world_vec = world_vec / world_norm
            local_vec = rot_base.inv().apply(world_vec)
            local_norm = np.linalg.norm(local_vec)
            if local_norm < 1e-8:
                continue
            local_vec = local_vec / local_norm
            target_quat = _vector_to_quaternion(local_vec)

            frame_evtac.append(evtac_entry)
            frame_base.append(
                np.concatenate(
                    [
                        # pos_t,
                        quat_t,
                        basevec_t,
                    ],
                    axis=0,
                ).astype(np.float32)
            )
            frame_target_quat.append(target_quat)

        n_frames = len(frame_evtac)
        if n_frames < self.window_size:
            return self._empty_bag_arrays()

        seq_evtac: List[Any] = []
        seq_base: List[np.ndarray] = []
        seq_target: List[np.ndarray] = []
        start_idx = self.window_size - 1

        for i in range(start_idx, n_frames):
            if self.materialize_dense_evtac:
                seq_evtac.append(np.stack(frame_evtac[i - start_idx : i + 1], axis=0))
            else:
                seq_evtac.append(frame_evtac[i - start_idx : i + 1])
            seq_base.append(np.stack(frame_base[i - start_idx : i + 1], axis=0))
            seq_target.append(frame_target_quat[i])

        if self.materialize_dense_evtac:
            evetac_out = np.stack(seq_evtac, axis=0).astype(np.float32)
        else:
            evetac_out = np.asarray(seq_evtac, dtype=object)
        return (
            evetac_out,
            np.stack(seq_base, axis=0).astype(np.float32),
            np.stack(seq_target, axis=0).astype(np.float32),
        )

    def _extract_sparse_evtac_frame(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "height": int(row["height"]),
            "width": int(row["width"]),
            "indices": np.asarray(row["indices"], dtype=np.int64),
            "values": np.asarray(row["values"], dtype=np.uint8),
        }

    def _sparse_evtac_to_downsampled_image(self, row: pd.Series) -> np.ndarray:
        """
        Converts sparse events into a 2-channel downsampled image:
          channel 0 (signal): mean signed event in each cell, in [-1, 1]
          channel 1 (density): normalized event density in [0, 1]
        """
        h = int(row["height"])
        w = int(row["width"])
        ds = self.downsample_factor
        h_ds = (h + ds - 1) // ds
        w_ds = (w + ds - 1) // ds

        signal = np.zeros((h_ds, w_ds), dtype=np.float32)
        density = np.zeros((h_ds, w_ds), dtype=np.float32)

        indices = np.asarray(row["indices"], dtype=np.int64)
        values = np.asarray(row["values"], dtype=np.uint8)
        if indices.size == 0:
            return np.stack([signal, density], axis=0)

        y = indices // w
        x = indices % w
        y_ds = np.clip(y // ds, 0, h_ds - 1)
        x_ds = np.clip(x // ds, 0, w_ds - 1)

        signs = np.zeros_like(values, dtype=np.float32)
        signs[values == 255] = 1.0
        signs[values == 0] = -1.0

        signed_sum = np.zeros((h_ds, w_ds), dtype=np.float32)
        counts = np.zeros((h_ds, w_ds), dtype=np.float32)
        np.add.at(signed_sum, (y_ds, x_ds), signs)
        np.add.at(counts, (y_ds, x_ds), 1.0)

        nonzero = counts > 0
        signal[nonzero] = signed_sum[nonzero] / counts[nonzero]

        # Fixed normalization by maximum number of pixels per ds cell.
        max_cell_events = float(ds * ds)
        density = np.clip(counts / max_cell_events, 0.0, 1.0)

        return np.stack([signal, density], axis=0)

    def _empty_bag_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._empty_evtac_array(),
            np.empty((0, self.window_size, 7), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )

    def _empty_evtac_array(self) -> np.ndarray:
        if self.materialize_dense_evtac:
            return np.empty((0, self.window_size, 2, 1, 1), dtype=np.float32)
        return np.empty((0, self.window_size), dtype=object)

    def _concat_evtac_batches(self, batches: List[np.ndarray]) -> np.ndarray:
        if self.materialize_dense_evtac:
            return np.concatenate(batches, axis=0).astype(np.float32)
        return np.concatenate(batches, axis=0).astype(object)

    def load_processed_data(
        self, base_path: str, target_cols: List[str]
    ) -> TactileData:
        if target_cols and len(target_cols) != 4:
            raise ValueError(
                "Temporal quaternion strategy expects 4 target columns "
                "(target_qx, target_qy, target_qz, target_qw)."
            )

        work_dir = Path(base_path) / self.get_dirname()
        data_path = work_dir / "data.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {data_path}")

        data = np.load(data_path, allow_pickle=True)
        return TactileData(
            evetac=data["evetac"],
            base=data["base"],
            target=data["target"],
        )

    def get_dataset(self, data_dict: Dict[str, np.ndarray]) -> Dataset:
        if self.materialize_dense_evtac:
            return TactileSequenceDataset(data_dict)
        return TactileSparseSequenceDataset(
            data_dict=data_dict, downsample_factor=self.downsample_factor
        )
