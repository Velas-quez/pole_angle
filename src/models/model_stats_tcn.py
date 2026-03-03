from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseTactileModel
from src.models.model_cnn_tcn import TemporalConvNet
from src.tactile_types import TactileInput


class TactileStatsTCN(BaseTactileModel):
    """
    Pure TCN model for temporal statistical features.

    Expected input:
      - evetac: (B, T, F_evtac) or (B, F_evtac) for single-step fallback
      - base:   (B, T, F_base) or (B, F_base) for single-step fallback
    """

    def __init__(
        self,
        input_shape_dict: Dict[str, Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        evetac_hidden_dim: int = 128,
        evetac_embedding_dim: int = 96,
        base_hidden_dim: int = 64,
        base_embedding_dim: int = 32,
        tcn_channels: Sequence[int] = (128, 128, 128),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        head_hidden_dims: Sequence[int] = (128, 64),
        temporal_window: Optional[int] = None,
        normalize_output: bool = True,
    ):
        super().__init__(input_shape_dict, output_shape)
        self.normalize_output = normalize_output
        self.output_dim = output_shape[0]

        evetac_shape = input_shape_dict["evetac"]
        base_shape = input_shape_dict["base"]
        if len(evetac_shape) < 1:
            raise ValueError("`evetac` shape must contain feature dimension.")
        if len(base_shape) < 1:
            raise ValueError("`base` shape must contain feature dimension.")

        evetac_in_dim = evetac_shape[-1]
        base_in_dim = base_shape[-1]

        self.evetac_encoder = nn.Sequential(
            nn.Linear(evetac_in_dim, evetac_hidden_dim),
            nn.ReLU(),
            nn.Linear(evetac_hidden_dim, evetac_embedding_dim),
            nn.ReLU(),
        )
        self.base_encoder = nn.Sequential(
            nn.Linear(base_in_dim, base_hidden_dim),
            nn.ReLU(),
            nn.Linear(base_hidden_dim, base_embedding_dim),
            nn.ReLU(),
        )

        temporal_in_dim = evetac_embedding_dim + base_embedding_dim
        self.temporal_model = TemporalConvNet(
            in_channels=temporal_in_dim,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )

        if temporal_window is not None:
            rf = self.temporal_model.receptive_field()
            if rf < temporal_window:
                raise ValueError(
                    f"TCN receptive field ({rf}) is smaller than temporal window "
                    f"({temporal_window}). Increase channels/depth or kernel size."
                )

        head_layers = []
        head_in = self.temporal_model.out_channels
        for h_dim in head_hidden_dims:
            head_layers.extend([nn.Linear(head_in, h_dim), nn.ReLU()])
            head_in = h_dim
        head_layers.append(nn.Linear(head_in, self.output_dim))
        self.regression_head = nn.Sequential(*head_layers)

    @staticmethod
    def _ensure_sequence_feature_shape(
        evetac: torch.Tensor, base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # evetac -> (B, T, F)
        if evetac.ndim == 2:
            evetac = evetac.unsqueeze(1)
        elif evetac.ndim != 3:
            raise ValueError(
                f"Unexpected `evetac` ndim={evetac.ndim}. Expected 2D or 3D tensor."
            )

        # base -> (B, T, F)
        if base.ndim == 2:
            base = base.unsqueeze(1)
        elif base.ndim != 3:
            raise ValueError(
                f"Unexpected `base` ndim={base.ndim}. Expected 2D or 3D tensor."
            )

        if evetac.shape[1] != base.shape[1]:
            raise ValueError(
                "Temporal length mismatch between `evetac` and `base`: "
                f"{evetac.shape[1]} vs {base.shape[1]}"
            )
        return evetac, base

    def forward(self, x: TactileInput) -> torch.Tensor:
        evetac = x["evetac"].float()
        base = x["base"].float()

        evetac, base = self._ensure_sequence_feature_shape(evetac, base)
        batch_size, time_steps, _ = evetac.shape

        evetac_feat = self.evetac_encoder(evetac.reshape(batch_size * time_steps, -1))
        base_feat = self.base_encoder(base.reshape(batch_size * time_steps, -1))

        temporal_features = torch.cat([evetac_feat, base_feat], dim=-1)
        temporal_features = temporal_features.reshape(batch_size, time_steps, -1)
        temporal_features = temporal_features.transpose(1, 2)

        tcn_out = self.temporal_model(temporal_features)
        last_timestep_feat = tcn_out[:, :, -1]

        out = self.regression_head(last_timestep_feat)
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=1, eps=1e-8)
        return out
