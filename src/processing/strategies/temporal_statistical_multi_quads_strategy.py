from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import LabelEncoder  # <-- Importação do Encoder
from torch.utils.data import Dataset

from src.models.datasets import TactileSequenceDataset
from src.processing.strategies.base_strategy import BaseStrategy
from src.processing.strategies.statistical_multi_quads_strategy import (
    extract_and_prepare_parquet,
)
from src.tactile_types import TactileData


class TemporalStatisticalMultiQuadsStrategy(BaseStrategy):
    """
    Temporal variant of multi-quads statistics.

    It reuses frame-level statistical extraction and then builds sliding windows:
      - evetac: (N, T, F_evtac)
      - base:   (N, T, F_base)
      - target: (N, 3) from the last timestep in each window
    """

    name: str = "temporal_statistical_multi_quads"

    def __init__(
        self,
        quad_width: float = 0.25,
        quad_height: float = 0.25,
        window_size: int = 10,
        stride: int = 1,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.quad_width = quad_width
        self.quad_height = quad_height
        self.window_size = int(window_size)
        self.stride = int(stride)

    def get_dirname(self) -> str:
        return (
            f"{self.name}_w{self.quad_width}_h{self.quad_height}"
            f"_T{self.window_size}_s{self.stride}"
        )

    def transform(
        self, evtac_path: str, markers_path: str, opti_path: str, output_base_dir: str
    ) -> None:
        df_all = extract_and_prepare_parquet(
            evtac_path, markers_path, opti_path, self.quad_width, self.quad_height
        )

        output_dir = Path(output_base_dir) / self.get_dirname()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "data.npz"

        if df_all.empty:
            np.savez_compressed(
                output_path,
                evetac=np.empty((0, self.window_size, 1), dtype=np.float32),
                base=np.empty((0, self.window_size, 1), dtype=np.float32),
                target=np.empty((0, 3), dtype=np.float32),
                user=np.empty((0,), dtype=object),  # <-- Array vazio para user
            )
            print(f"⚠️ No valid frame-level data. Empty dataset saved to: {output_path}")
            return

        evetac_cols = [c for c in df_all.columns if c.startswith("evetac_")]
        base_cols = [c for c in df_all.columns if c.startswith("base_")]
        target_cols = ["target_x", "target_y", "target_z"]

        seq_evtac: List[np.ndarray] = []
        seq_base: List[np.ndarray] = []
        seq_target: List[np.ndarray] = []
        seq_user: List[str] = []  # <-- Lista para capturar os usuários

        bags = [b for b in df_all["bag_name"].unique() if b is not None]
        print(
            f"🔄 Building temporal windows (T={self.window_size}, stride={self.stride}) for {len(bags)} bags..."
        )

        for bag in bags:
            df_bag = df_all[df_all["bag_name"] == bag].reset_index(drop=True)
            if len(df_bag) < self.window_size:
                continue

            evetac_arr = df_bag[evetac_cols].to_numpy(dtype=np.float32)
            base_arr = df_bag[base_cols].to_numpy(dtype=np.float32)
            target_arr = df_bag[target_cols].to_numpy(dtype=np.float32)
            user_arr = df_bag["user"].to_numpy(
                dtype=object
            )  # <-- Extrai a coluna de usuários

            for end_idx in range(self.window_size - 1, len(df_bag), self.stride):
                start_idx = end_idx - self.window_size + 1
                seq_evtac.append(evetac_arr[start_idx : end_idx + 1])
                seq_base.append(base_arr[start_idx : end_idx + 1])
                seq_target.append(target_arr[end_idx])
                seq_user.append(
                    user_arr[end_idx]
                )  # <-- Pega o user correspondente ao final da janela

        if not seq_target:
            evetac_out = np.empty(
                (0, self.window_size, len(evetac_cols)), dtype=np.float32
            )
            base_out = np.empty((0, self.window_size, len(base_cols)), dtype=np.float32)
            target_out = np.empty((0, 3), dtype=np.float32)
            user_out = np.empty((0,), dtype=object)  # <-- Fallback
        else:
            evetac_out = np.stack(seq_evtac, axis=0).astype(np.float32)
            base_out = np.stack(seq_base, axis=0).astype(np.float32)
            target_out = np.stack(seq_target, axis=0).astype(np.float32)
            user_out = np.array(seq_user, dtype=object)  # <-- Converte a lista em array

        np.savez_compressed(
            output_path,
            evetac=evetac_out,
            base=base_out,
            target=target_out,
            user=user_out,  # <-- Salva no arquivo npz
        )
        print(
            f"✅ Saved temporal statistical dataset to {output_path} | "
            f"evetac={evetac_out.shape}, base={base_out.shape}, target={target_out.shape}, user={user_out.shape}"
        )

    def load_processed_data(
        self, base_path: str, target_cols: List[str]
    ) -> TactileData:
        expected_targets = ["target_x", "target_y", "target_z"]
        if target_cols and target_cols != expected_targets:
            raise ValueError(
                "Temporal statistical strategy expects target columns "
                f"{expected_targets}, got {target_cols}."
            )

        work_dir = Path(base_path) / self.get_dirname()
        data_path = work_dir / "data.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {data_path}")

        # allow_pickle=True é obrigatório aqui porque o array 'user' contém strings (objetos)
        data = np.load(data_path, allow_pickle=True)

        # Encoder dos usuários
        raw_users = data["user"]
        labelencoder = LabelEncoder()
        users_encoded = labelencoder.fit_transform(raw_users)

        return TactileData(
            user=np.array(users_encoded),  # <-- Injeta na dataclass
            evetac=data["evetac"],
            base=data["base"],
            target=data["target"],
        )

    def get_dataset(self, data_dict: Dict[str, np.ndarray]) -> Dataset:
        return TactileSequenceDataset(data_dict)
