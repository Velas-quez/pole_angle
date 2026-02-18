import torch
import torch.nn as nn
import mlflow
import joblib
import hydra
from abc import ABC, abstractmethod
from functools import cached_property
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline


class BaseTactileModel(ABC):
    def __init__(self, input_dim: int, output_dim: int, model_cfg: DictConfig):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_cfg = model_cfg
        self.device = torch.device(
            model_cfg.params.device if torch.cuda.is_available() else "cpu"
        )

        # Objetos que serão populados durante o ciclo de vida
        self.model = self.build_model(input_dim=input_dim, output_dim=output_dim).to(
            self.device
        )

    @abstractmethod
    def build_model(self, input_dim: int, output_dim) -> nn.Module:
        """Deve retornar a arquitetura PyTorch."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encaminha os dados pelo modelo. Pode ser sobrescrito para lógica customizada."""
        return self.model(x)
