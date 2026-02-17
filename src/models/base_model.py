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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.params.device if torch.cuda.is_available() else "cpu"
        )

        # Objetos que serão populados durante o ciclo de vida
        self.model = self.build_model(input_dim=cfg.model.input_dim)

    @abstractmethod
    def build_model(self, input_dim: int) -> nn.Module:
        """Deve retornar a arquitetura PyTorch."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encaminha os dados pelo modelo. Pode ser sobrescrito para lógica customizada."""
        return self.model(x)

    @cached_property
    def criterion(self) -> nn.Module:
        """Pode ser sobrescrito para perdas customizadas."""
        return hydra.utils.instantiate(self.cfg.criterion)

    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        """Instanciado via Hydra após o modelo estar no device correto."""
        if self.model is None:
            raise RuntimeError("Modelo deve ser inicializado antes do optimizer.")
        return hydra.utils.instantiate(
            self.cfg.optimizer, params=self.model.parameters()
        )

    @cached_property
    def scheduler(self):
        """Scheduler para ajuste de Learning Rate."""
        return hydra.utils.instantiate(self.cfg.scheduler, optimizer=self.optimizer)

    def train_step(self, loader):
        """Lógica de um passo de treino."""
        # ... implementaremos o loop aqui
        pass

    def fit(self, parquet_path: str):
        """
        O fluxo principal:
        1. Carrega e Splita (Temporal)
        2. Fit do Pipeline (apenas no Train)
        3. Build do Modelo
        4. Loop de Treino via MLflow
        """
        # ... implementaremos o gerenciamento de estados aqui
        pass

