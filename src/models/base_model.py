from typing import Generic, TypeVar, Type
import torch
import torch.nn as nn
import mlflow
import joblib
import hydra
from abc import ABC, abstractmethod
from functools import cached_property
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline


T = TypeVar("T")


class BaseTactileModel(nn.Module, ABC, Generic[T]):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        net_params: DictConfig,
        params_schema: Type[T],
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_params: T = OmegaConf.to_object(
            OmegaConf.merge(params_schema, net_params)
        )
        self.device = torch.device(
            net_params.params.device if torch.cuda.is_available() else "cpu"
        )

        # Objetos que serão populados durante o ciclo de vida
        self.model = self.build_model().to(self.device)

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Deve retornar a arquitetura PyTorch."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encaminha os dados pelo modelo. Pode ser sobrescrito para lógica customizada."""
        return self.model(x)
