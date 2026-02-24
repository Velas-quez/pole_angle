from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
import torch.nn as nn
from src.tactile_types import TactileInput


class BaseTactileModel(nn.Module, ABC):
    def __init__(
        self,
        input_shape_dict: Dict[str, Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ):
        super().__init__()
        self.input_shape_dict = input_shape_dict
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, x: TactileInput) -> torch.Tensor:
        """
        Agora 'x' é um TypedDict. O LSP saberá que x['evetac'] e x['base'] existem.
        """
        pass
