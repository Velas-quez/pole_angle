from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.tactile_types import TactileData
from src.processing.strategies.base_strategy import BaseStrategy
from src.models.datasets import TactileVectorDataset

# Importing the heavy lifting from your original file
from .statistical_multi_quads_strategy import extract_and_prepare_parquet


class SlidingWindowQuadsStrategy(BaseStrategy):
    """
    Temporal strategy that concatenates features from the current time (t)
    and the two previous steps (t-1, t-2).
    """

    name: str = "sliding_window_quads"

    def __init__(
        self, quad_width: float = 1, quad_height: float = 1, window_size: int = 3
    ):
        self.quad_width = quad_width
        self.quad_height = quad_height
        self.window_size = window_size

    def get_dirname(self) -> str:
        return f"{self.name}_w{self.quad_width}_h{self.quad_height}"

    def transform(
        self, evtac_path: str, markers_path: str, opti_path: str, output_base_dir: str
    ) -> None:
        # 1. Run the standard extraction (Hu moments, interpolation, etc.)
        # Note: Your extract_and_prepare_parquet returns a DF with 'bag_name'
        df_all = extract_and_prepare_parquet(
            evtac_path, markers_path, opti_path, self.quad_width, self.quad_height
        )

        # 2. Apply Temporal Windowing per Bag to avoid data leakage
        temporal_results = []
        bags = df_all["bag_name"].unique()

        print(
            f"🔄 Applying temporal windowing (size {self.window_size}) across {len(bags)} bags..."
        )

        for bag in bags:
            df_bag = df_all[df_all["bag_name"] == bag].copy()

            if len(df_bag) < self.window_size:
                continue

            # Define column groups
            target_cols = ["target_x", "target_y", "target_z"]
            # Features are everything else (evetac stats + base orientation)
            feature_cols = [
                c
                for c in df_bag.columns
                if c.startswith("evetac_") or c.startswith("base_")
            ]

            # Create the shifts
            # t0 = current, t1 = previous (t-1), t2 = two back (t-2)
            df_t0 = df_bag[feature_cols].reset_index(drop=True)
            df_t1 = df_bag[feature_cols].shift(1).reset_index(drop=True)
            df_t2 = df_bag[feature_cols].shift(2).reset_index(drop=True)

            # Rename shifted columns to keep them distinct
            df_t1.columns = [f"{c}_t-1" for c in df_t1.columns]
            df_t2.columns = [f"{c}_t-2" for c in df_t2.columns]

            # Concatenate features: order [t-2, t-1, t]
            df_temporal = pd.concat([df_t2, df_t1, df_t0], axis=1)

            # Assign targets from current time (t0)
            df_temporal[target_cols] = df_bag[target_cols].reset_index(drop=True)

            # 3. Drop the first 2 rows (they contain NaNs because they have no history)
            df_temporal = df_temporal.dropna().reset_index(drop=True)

            temporal_results.append(df_temporal)

        df_final = pd.concat(temporal_results, ignore_index=True)

        # 4. Save
        output_dir = Path(output_base_dir) / self.get_dirname()
        output_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_dir / "data.parquet", index=False)
        print(f"✅ Saved temporal data (Samples: {len(df_final)}) in: {output_dir}")

    def load_processed_data(
        self, base_path: str, target_cols: List[str]
    ) -> TactileData:
        work_dir = Path(base_path) / self.get_dirname()
        df = pd.read_parquet(work_dir / "data.parquet")

        # Automatically select all temporal versions of evetac and base columns
        evetac_cols = [c for c in df.columns if "evetac_" in c]
        base_cols = [c for c in df.columns if "base_" in c]

        return TactileData(
            evetac=df[evetac_cols].values,
            base=df[base_cols].values,
            target=df[target_cols].values,
        )

    def get_dataset(self, data_dict: Dict[str, np.ndarray]) -> Dataset:
        return TactileVectorDataset(data_dict)
