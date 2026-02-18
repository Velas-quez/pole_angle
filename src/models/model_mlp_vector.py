from typing_extensions import Any
import pandas as pd
from src.models.base_model import BaseTactileModel
import joblib
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from typing import Optional, Tuple, List
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass


# --- 0. Hyperparameters ---
@dataclass
class MLPParams:
    num_layers: int
    neurons: List[int]
    activation: DictConfig  # Contém o _target_ para o instantiate
    dropout: float = 0.1
    normalization: Optional[DictConfig] = None


# --- 2. Model Definition with L2 Normalization ---
class TactileMLPVector(BaseTactileModel[MLPParams]):
    def __init__(self, input_dims: int, output_dim: int, net_params: DictConfig):
        super().__init__(input_dims, output_dim, net_params, MLPParams)

    def build_model(self) -> nn.Module:
        """
        Constrói a rede dinamicamente. O Neovim agora reconhece self.model_cfg
        como uma instância de MLPParams graças ao Generics na Base.
        """
        layers = []
        neurons = self.net_params.neurons
        curr_in = self.input_dim

        # Iitera sobre a lista de neurônios para criar as camadas ocultas
        for h_dim in neurons:
            # 1. Camada Linear
            layers.append(nn.Linear(curr_in, h_dim))

            # 2. Normalização (Batch ou Layer Norm via Hydra)
            if self.net_params.normalization is not None:
                # Instancia o target (ex: torch.nn.BatchNorm1d) passando o tamanho da camada
                norm_layer = hydra.utils.instantiate(
                    self.net_params.normalization, h_dim
                )
                layers.append(norm_layer)

            # 3. Função de Ativação (instancia uma nova para cada camada)
            layers.append(hydra.utils.instantiate(self.net_params.activation))

            # 4. Dropout para regularização
            if self.net_params.dropout > 0:
                layers.append(nn.Dropout(self.net_params.dropout))

            curr_in = h_dim

        # Camada final de saída
        layers.append(nn.Linear(curr_in, self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # Normalização L2 na última camada: garante que o vetor de saída seja unitário
        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8)


# --- 3. Helper Functions ---
def get_angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates angular error in degrees between unit vectors."""
    # Como as saídas já estão normalizadas, o dot product é a própria similaridade de cosseno
    cos_theta = torch.sum(pred * target, dim=1)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos_theta))
