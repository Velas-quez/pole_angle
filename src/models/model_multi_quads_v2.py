from src.models.base_model import BaseTactileModel
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig

# --- 0. Hyperparameters ---


# --- 2. Model Definition with L2 Normalization ---
class TactileMLP(BaseTactileModel):
    def __init__(self, input_dims: int, output_dim: int, model_cfg: DictConfig):
        super().__init__(input_dims, output_dim, model_cfg)

    def build_model(self, input_dim: int, output_dim) -> nn.Module:
        layers = []
        num_layers = self.model_cfg.num_layers
        neurons = self.model_cfg.neurons
        activation_fn = hydra.utils.instantiate(self.model_cfg.activation)

        layers.append(nn.Linear(input_dim, neurons[0]))
        for i in range(1, num_layers):
            if self.model_cfg.normalization is not None:
                normalization_layer = hydra.utils.instantiate(
                    self.model_cfg.normalization, neurons[i]
                )
                layers.append(normalization_layer)
            layers.append(activation_fn)
            layers.append(nn.Dropout(self.model_cfg.dropout))
            layers.append(nn.Linear(neurons[i - 1], neurons[i]))

        layers.append(nn.Linear(neurons[-1], output_dim))
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
