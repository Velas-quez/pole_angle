import torch
from torch import nn
from typing import Dict, Tuple
from src.models.base_model import BaseTactileModel
from src.tactile_types import TactileInput


class DummyModel(BaseTactileModel):
    def __init__(
        self,
        input_shape_dict: Dict[str, Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ):
        """
        A baseline model that always outputs the Z-axis vector (0, 0, 1).
        """
        super().__init__(input_shape_dict, output_shape)
        # Add a dummy parameter to make .parameters() valid
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: TactileInput) -> torch.Tensor:
        """
        Return a fixed tensor for the Z-axis vector (0, 0, 1), repeated for the batch size.
        """
        batch_size = x["evetac"].size(
            0
        )  # Retrieve the batch size from the tactile input.
        z_vector = torch.tensor(
            [0.0, 0.0, 1.0], device=x["evetac"].device, requires_grad=True
        )
        return z_vector.unsqueeze(0).repeat(batch_size, 1)
