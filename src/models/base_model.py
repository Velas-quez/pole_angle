import torch
import torch.nn as nn
import mlflow
import joblib
import hydra
from abc import ABC, abstractmethod
from functools import cached_property
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

class BaseTactileTrainer(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.params.device if torch.cuda.is_available() else "cpu")
        
        # Objetos que serão populados durante o ciclo de vida
        self.model = None
        self.pipeline = None 

    @abstractmethod
    def build_model(self, input_dim: int) -> nn.Module:
        """Deve retornar a arquitetura PyTorch."""
        pass

    @cached_property
    def criterion(self) -> nn.Module:
        """Pode ser sobrescrito para perdas customizadas."""
        return hydra.utils.instantiate(self.cfg.criterion)

    @cached_property
    def optimizer(self) -> torch.optim.Optimizer:
        """Instanciado via Hydra após o modelo estar no device correto."""
        if self.model is None:
            raise RuntimeError("Modelo deve ser inicializado antes do optimizer.")
        return hydra.utils.instantiate(self.cfg.optimizer, params=self.model.parameters())

    @cached_property
    def scheduler(self):
        """Scheduler para ajuste de Learning Rate."""
        return hydra.utils.instantiate(self.cfg.scheduler, optimizer=self.optimizer)

    @abstractmethod
    def create_pipeline(self) -> Pipeline:
        """Define o Pipeline do Sklearn (Scaler, Selectors, etc.)."""
        pass

    def save_artifacts(self, path: str = "artifacts"):
        """Salva localmente e loga no MLflow."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Salva o Pipeline (o .pkl mais importante)
        pipeline_path = os.path.join(path, "pipeline.pkl")
        joblib.dump(self.pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path, "preprocessing")
        
        # Loga o modelo PyTorch
        mlflow.pytorch.log_model(self.model, "model")

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