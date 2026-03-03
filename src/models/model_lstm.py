from typing import List, Optional, Dict, Tuple, Callable
import torch
import torch.nn as nn
from src.models.base_model import BaseTactileModel
from src.tactile_types import TactileInput


class TactileLSTMVector(BaseTactileModel):
    def __init__(
        self,
        input_shape_dict: Dict[str, Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        window_size: int,
        lstm_hidden_dim: int,
        neurons: List[int],
        activation: Callable,
        dropout: float = 0.1,
        normalization: Optional[Callable] = None,
    ):
        super().__init__(input_shape_dict, output_shape)
        self.window_size = window_size

        # Determine the number of features per time step
        # Original shape is (window_size * features), we need just 'features'
        self.evetac_feats = input_shape_dict["evetac"][0] // window_size
        self.base_feats = input_shape_dict["base"][0] // window_size

        # 1. Temporal Processor (LSTM)
        # We process the combined features of evetac and base per step
        self.lstm = nn.LSTM(
            input_size=self.evetac_feats + self.base_feats,
            hidden_size=lstm_hidden_dim,
            num_layers=1,  # Fixed to 1 as requested
            batch_first=True,
        )

        # 2. Decision Maker (MLP)
        layers = []
        curr_in = lstm_hidden_dim
        for h_dim in neurons:
            layers.append(nn.Linear(curr_in, h_dim))
            if normalization is not None:
                layers.append(normalization(h_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_in = h_dim

        # Final output layer for [x, y, z]
        layers.append(nn.Linear(curr_in, output_shape[0]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: TactileInput) -> torch.Tensor:
        # x["evetac"] shape: (Batch, window_size * evetac_feats)
        batch_size = x["evetac"].size(0)

        # Reshape to 3D: (Batch, Sequence_Length, Features)
        evetac_seq = x["evetac"].view(batch_size, self.window_size, self.evetac_feats)
        base_seq = x["base"].view(batch_size, self.window_size, self.base_feats)

        # Combine sensors at each time step
        combined_seq = torch.cat((evetac_seq, base_seq), dim=2)

        # Pass through LSTM
        # hn is the last hidden state: (num_layers, Batch, lstm_hidden_dim)
        _, (hn, _) = self.lstm(combined_seq)

        # Use the hidden state from the last (and only) layer
        out = self.net(hn[-1])

        # L2 Normalization to ensure a unit vector output
        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8)
