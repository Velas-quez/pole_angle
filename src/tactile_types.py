from typing import TypedDict
import torch
import numpy as np


class TactileInput(TypedDict):
    evetac: torch.Tensor  # Pode ser (B, H, W) ou (B, Features)
    base: torch.Tensor  # Costuma ser (B, 2) para os ângulos Alpha/Beta


class TactileData(TypedDict):
    user: np.ndarray
    evetac: np.ndarray  # Dados do EvETAC, pode ser (N, H, W) ou (N, Features)
    base: np.ndarray  # Dados da Base, geralmente (N, 2) para
    target: (
        np.ndarray
    )  # O que queremos prever, pode ser (B, 2) para direção ou (B,) para classe
