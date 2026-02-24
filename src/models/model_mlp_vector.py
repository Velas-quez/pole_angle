from typing import List, Optional, Dict, Tuple, Any
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from src.models.base_model import BaseTactileModel
from src.tactile_types import TactileInput

from typing import Callable


class TactileMLPVector(BaseTactileModel):
    def __init__(
        self,
        input_shape_dict: Dict[str, Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        neurons: List[int],
        activation: Callable,  # Mudado para Any porque o partial não é um DictConfig
        dropout: float = 0.1,
        normalization: Optional[Callable] = None,  # Mudado para Any
    ):
        super().__init__(input_shape_dict, output_shape)

        # Cálculo das dimensões achatadas
        self.evetac_dim: int = int(
            torch.prod(torch.tensor(input_shape_dict["evetac"])).item()
        )
        self.base_dim: int = int(
            torch.prod(torch.tensor(input_shape_dict["base"])).item()
        )

        total_in = self.evetac_dim + self.base_dim
        total_out = output_shape[0]

        # Construção da MLP
        layers = []
        curr_in = total_in
        for h_dim in neurons:
            layers.append(nn.Linear(curr_in, h_dim))

            if normalization is not None:
                layers.append(normalization(h_dim))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_in = h_dim

        layers.append(nn.Linear(curr_in, total_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: TactileInput) -> torch.Tensor:
        evetac = x["evetac"].view(x["evetac"].size(0), -1)
        base = x["base"].view(x["base"].size(0), -1)

        combined = torch.cat((evetac, base), dim=1)
        out = self.net(combined)

        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8)
