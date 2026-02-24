import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple
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
