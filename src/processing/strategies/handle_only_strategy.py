from typing import List, Dict
from pathlib import Path
import numpy as np
import pandas as pd

# Missing Scientific Imports
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from src.tactile_types import TactileData
from .base_strategy import BaseStrategy
from src.models.datasets import TactileVectorDataset

# Import helper functions from your other strategy
from .statistical_multi_quads_strategy import estimate_pole_vector_from_row


class HandleOnlyStrategy(BaseStrategy):
    """
    Strategy that uses ONLY the robot handle orientation (Pitch/Yaw) as input.
    Ignores tactile data entirely to test baseline physics correlation.
    """

    name: str = "handle_only"

    def get_dirname(self) -> str:
        return self.name

    def _extract_handle_only_parquet(
        self,
        evtac_path: str,
        markers_path: str,
        opti_path: str,
    ) -> pd.DataFrame:
        # 1. Load only necessary columns to save RAM (Crucial to avoid 'Killed' error)
        df_evtac_full = pd.read_parquet(
            evtac_path, columns=["timestamp", "bag_name", "user"]
        )
        df_markers_full = pd.read_parquet(markers_path)
        df_opti_full = pd.read_parquet(opti_path)

        for df in [df_evtac_full, df_markers_full, df_opti_full]:
            if "bagfile" in df.columns:
                df.rename(inplace=True, columns={"bagfile": "bag_name"})

        all_episodes_results = []
        bags = [b for b in df_evtac_full["bag_name"].unique() if pd.notna(b)]

        for bag in bags:
            print(f"📦 Processing Handle data for bag: {bag}")

            df_evtac = df_evtac_full[df_evtac_full["bag_name"] == bag].copy()
            df_markers = df_markers_full[df_markers_full["bag_name"] == bag].copy()
            df_opti = df_opti_full[df_opti_full["bag_name"] == bag].copy()

            if df_opti.empty or df_evtac.empty:
                continue

            # 2. Time Synchronization
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

            mask = (df_evtac["timestamp"] >= t_start) & (df_evtac["timestamp"] <= t_end)
            target_timestamps = df_evtac["timestamp"][mask].values
            users = deepcopy(df_evtac["user"][mask].values)

            # 3. Physics Interpolation
            processed_markers = [
                estimate_pole_vector_from_row(row) for _, row in df_markers.iterrows()
            ]
            marker_vec_arr = np.array(processed_markers)
            _, u_idx = np.unique(marker_vec_arr[:, 0], return_index=True)
            marker_vec_arr = marker_vec_arr[u_idx]

            df_opti = df_opti.drop_duplicates(subset=["timestamp"]).sort_values(
                "timestamp"
            )
            base_quats = np.array(
                [[d["x"], d["y"], d["z"], d["w"]] for d in df_opti["orientation"]]
            )

            # Interpolators
            f_quat = interp1d(
                df_opti["timestamp"].values,
                base_quats,
                axis=0,
                fill_value="extrapolate",
            )
            interp_quats = f_quat(target_timestamps)
            interp_quats /= np.linalg.norm(interp_quats, axis=1, keepdims=True)
            sync_rots_base = R.from_quat(interp_quats)

            f_marker_v = interp1d(
                marker_vec_arr[:, 0],
                marker_vec_arr[:, 1:],
                axis=0,
                fill_value="extrapolate",
            )
            sync_v_world = f_marker_v(target_timestamps)

            # 4. Coordinate Transformation
            final_vec_targets, base_eulers, base_vectors = [], [], []
            for i in range(len(target_timestamps)):
                v_local = sync_rots_base[i].inv().apply(sync_v_world[i])
                final_vec_targets.append(v_local / (np.linalg.norm(v_local) + 1e-7))
                base_eulers.append(sync_rots_base[i].as_euler("xyz", degrees=True))
                base_vectors.append(sync_rots_base[i].apply(np.array([0, 0, 1])))

            # Lightweight DataFrame
            df_bag = pd.DataFrame()
            df_bag[["base_roll", "base_pitch", "base_yaw"]] = np.array(base_eulers)
            df_bag[["base_vec_x", "base_vec_y", "base_vec_z"]] = np.array(base_vectors)
            df_bag[["target_x", "target_y", "target_z"]] = np.array(final_vec_targets)
            df_bag["user"] = users

            all_episodes_results.append(df_bag)
            print(f"   ✅ Added {len(df_bag)} samples.")

        return (
            pd.concat(all_episodes_results, ignore_index=True)
            if all_episodes_results
            else pd.DataFrame()
        )

    def transform(
        self, evtac_path: str, markers_path: str, opti_path: str, output_base_dir: str
    ) -> None:
        # Call the lightweight internal method instead of the heavy one
        df_final = self._extract_handle_only_parquet(
            evtac_path, markers_path, opti_path
        )

        output_dir = Path(output_base_dir) / self.get_dirname()
        output_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_dir / "data.parquet", index=False)
        print(f"✅ Handle-only data saved in: {output_dir}")

    def load_processed_data(
        self, base_path: str, target_cols: List[str]
    ) -> TactileData:
        work_dir = Path(base_path) / self.get_dirname()
        df = pd.read_parquet(work_dir / "data.parquet")

        input_cols = ["base_pitch", "base_yaw"]

        # Consistent dummy array for the tactile branch of the model
        evetac_dummy = np.zeros((len(df), 1))
        labelencoder = LabelEncoder()
        users = labelencoder.fit_transform(df["user"])
        return TactileData(
            user=np.array(users),
            evetac=evetac_dummy,
            base=df[input_cols].values,
            target=df[target_cols].values,
        )

    def get_dataset(self, data_dict: Dict[str, np.ndarray]) -> Dataset:
        return TactileVectorDataset(data_dict)
