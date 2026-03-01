import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Dict, Tuple
from src.tactile_types import TactileData, TactileInput


class TactileVectorDataset(Dataset):
    """
    Dataset para estratégias que resultam em vetores 1D (ex: MLP com Hu Moments).
    Espera que 'evetac' no data_dict tenha o shape (N, Features).
    """

    def __init__(self, data_dict: TactileData):
        # Conversão direta para tensores
        self.evetac = torch.as_tensor(data_dict["evetac"], dtype=torch.float32)
        self.base = torch.as_tensor(data_dict["base"], dtype=torch.float32)
        self.targets = torch.as_tensor(data_dict["target"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[TactileInput, torch.Tensor]:
        # Monta o TypedDict para que o modelo saiba onde ler
        x: TactileInput = {"evetac": self.evetac[idx], "base": self.base[idx]}
        return x, self.targets[idx]


class TactileImageDataset(Dataset):
    """
    Dataset para estratégias que resultam em matrizes 2D (ex: CNN com pixels).
    Transforma (N, H, W) em (N, C, H, W) adicionando a dimensão de canal (C=1).
    """

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        # Se os dados vierem como (N, H, W), fazemos o unsqueeze para (N, 1, H, W)
        evetac_raw = torch.as_tensor(data_dict["evetac"], dtype=torch.float32)

        if evetac_raw.ndim == 3:
            self.evetac = evetac_raw.unsqueeze(1)
        else:
            self.evetac = evetac_raw

        self.base = torch.as_tensor(data_dict["base"], dtype=torch.float32)
        self.targets = torch.as_tensor(data_dict["target"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[TactileInput, torch.Tensor]:
        x: TactileInput = {"evetac": self.evetac[idx], "base": self.base[idx]}
        return x, self.targets[idx]


class TactileSequenceDataset(Dataset):
    """
    Dataset para modelos temporais (ex: LSTM/Transformers).
    Lida com janelas de tempo: (N, Window_Size, Features).
    """

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.evetac = torch.as_tensor(data_dict["evetac"], dtype=torch.float32)
        self.base = torch.as_tensor(data_dict["base"], dtype=torch.float32)
        self.targets = torch.as_tensor(data_dict["target"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[TactileInput, torch.Tensor]:
        x: TactileInput = {"evetac": self.evetac[idx], "base": self.base[idx]}
        return x, self.targets[idx]


class TactileSparseSequenceDataset(Dataset):
    """
    Dataset temporal para EvETAC esparso.
    Espera `evetac` como array object com shape (N, T), onde cada elemento
    contém dict com: height, width, indices, values.
    Densifica on-the-fly para (T, 2, H_ds, W_ds).
    """

    def __init__(self, data_dict: Dict[str, np.ndarray], downsample_factor: int):
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")
        self.evetac_sparse = data_dict["evetac"]
        self.base = torch.as_tensor(data_dict["base"], dtype=torch.float32)
        self.targets = torch.as_tensor(data_dict["target"], dtype=torch.float32)
        self.downsample_factor = int(downsample_factor)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[TactileInput, torch.Tensor]:
        sparse_seq = self.evetac_sparse[idx]
        dense_frames = [
            self._sparse_frame_to_downsampled_2ch(frame) for frame in sparse_seq
        ]
        evetac = torch.stack(dense_frames, dim=0)  # (T, 2, H_ds, W_ds)
        x: TactileInput = {"evetac": evetac, "base": self.base[idx]}
        return x, self.targets[idx]

    def _sparse_frame_to_downsampled_2ch(self, frame: Dict[str, Any]) -> torch.Tensor:
        h = int(frame["height"])
        w = int(frame["width"])
        ds = self.downsample_factor
        h_ds = (h + ds - 1) // ds
        w_ds = (w + ds - 1) // ds

        signal = np.zeros((h_ds, w_ds), dtype=np.float32)
        density = np.zeros((h_ds, w_ds), dtype=np.float32)

        indices = np.asarray(frame["indices"], dtype=np.int64)
        values = np.asarray(frame["values"], dtype=np.uint8)
        if indices.size == 0:
            return torch.from_numpy(np.stack([signal, density], axis=0))

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

        density = np.clip(counts / float(ds * ds), 0.0, 1.0)
        return torch.from_numpy(np.stack([signal, density], axis=0))
