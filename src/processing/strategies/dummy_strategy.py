from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset
from src.tactile_types import TactileData
from .base_strategy import BaseStrategy
from src.models.datasets import TactileVectorDataset
from .statistical_multi_quads_strategy import estimate_pole_vector_from_row


class DummyStrategy(BaseStrategy):
    """
    Estratégia Dummy que não processa inputs.
    Gera arrays vazios para 'evetac' e 'base' para economizar RAM e tempo.
    Calcula apenas os targets rigorosos para uso do DummyModel baseline.
    """

    name: str = "dummy_baseline"

    def get_dirname(self) -> str:
        return self.name

    def transform(
        self, evtac_path: str, markers_path: str, opti_path: str, output_base_dir: str
    ) -> None:
        """Extrai apenas a quantidade de timestamps e calcula os targets correspondentes."""
        # 1. Carrega apenas as colunas vitais para não estourar a RAM
        df_evtac = pd.read_parquet(evtac_path, columns=["timestamp", "bag_name"])
        df_markers = pd.read_parquet(markers_path)
        df_opti = pd.read_parquet(opti_path)

        for df in [df_evtac, df_markers, df_opti]:
            if "bagfile" in df.columns:
                df.rename(inplace=True, columns={"bagfile": "bag_name"})

        bags = [b for b in df_evtac["bag_name"].unique() if pd.notna(b)]
        all_episodes_results = []

        for bag in bags:
            print(f"📦 Extraindo Targets (Dummy) da bag: {bag}")
            evtac_bag = df_evtac[df_evtac["bag_name"] == bag]
            markers_bag = df_markers[df_markers["bag_name"] == bag]
            opti_bag = df_opti[df_opti["bag_name"] == bag]

            if opti_bag.empty or evtac_bag.empty or markers_bag.empty:
                continue

            # Sincronização de tempo básica (mantendo a margem de 2.0s do original)
            t_start = (
                max(
                    evtac_bag["timestamp"].min(),
                    markers_bag["timestamp"].min(),
                    opti_bag["timestamp"].min(),
                )
                + 2.0
            )

            t_end = (
                min(
                    evtac_bag["timestamp"].max(),
                    markers_bag["timestamp"].max(),
                    opti_bag["timestamp"].max(),
                )
                - 2.0
            )

            if t_start >= t_end:
                continue

            # Esses são os frames que VÃO virar targets reais
            mask = (evtac_bag["timestamp"] >= t_start) & (
                evtac_bag["timestamp"] <= t_end
            )
            target_timestamps = evtac_bag["timestamp"][mask].values

            if len(target_timestamps) == 0:
                continue

            # --- PROCESSAMENTO APENAS DOS TARGETS (Ground Truth) ---
            # 1. Marker Vectors
            processed_markers = [
                estimate_pole_vector_from_row(row) for _, row in markers_bag.iterrows()
            ]
            marker_vec_arr = np.array(processed_markers)
            _, u_idx = np.unique(marker_vec_arr[:, 0], return_index=True)
            marker_vec_arr = marker_vec_arr[u_idx]

            # 2. Base Quaternions
            opti_bag = opti_bag.drop_duplicates(subset=["timestamp"]).sort_values(
                "timestamp"
            )
            base_quats = np.array(
                [[d["x"], d["y"], d["z"], d["w"]] for d in opti_bag["orientation"]]
            )

            # Interpoladores
            f_quat = interp1d(
                opti_bag["timestamp"].values,
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

            # 3. Transformação para o Frame Local
            final_vec_targets = []
            for i in range(len(target_timestamps)):
                v_local = sync_rots_base[i].inv().apply(sync_v_world[i])
                final_vec_targets.append(v_local / (np.linalg.norm(v_local) + 1e-7))

            # Guarda os targets. Não geramos inputs aqui, isso é feito no load.
            df_bag = pd.DataFrame(
                final_vec_targets, columns=["target_x", "target_y", "target_z"]
            )
            all_episodes_results.append(df_bag)

        df_final = (
            pd.concat(all_episodes_results, ignore_index=True)
            if all_episodes_results
            else pd.DataFrame()
        )

        output_dir = Path(output_base_dir) / self.get_dirname()
        output_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(output_dir / "data.parquet", index=False)
        print(f"✅ Dummy strategy guardada em: {output_dir}")

    def load_processed_data(
        self, base_path: str, target_cols: List[str]
    ) -> TactileData:
        work_dir = Path(base_path) / self.get_dirname()
        df = pd.read_parquet(work_dir / "data.parquet")

        num_samples = len(df)

        # Cria inputs 100% inúteis e leves. O DummyModel não vai olhar pra eles.
        # Shape (N, 1) é suficiente para o PyTorch criar os batches sem estourar.
        evetac_dummy = np.zeros((num_samples, 1), dtype=np.float32)
        base_dummy = np.zeros((num_samples, 1), dtype=np.float32)

        return TactileData(
            evetac=evetac_dummy,
            base=base_dummy,
            target=df[target_cols].values.astype(np.float32),
        )

    def get_dataset(self, data_dict: Dict[str, Any]) -> Dataset:
        return TactileVectorDataset(data_dict)
