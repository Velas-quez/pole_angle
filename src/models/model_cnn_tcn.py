from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseTactileModel
from src.tactile_types import TactileInput


def quaternion_geodesic_loss(
    pred_q: torch.Tensor, target_q: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Geodesic loss for unit quaternions:
        loss = 1 - (dot(q_pred, q_gt))^2
    """
    pred_q = F.normalize(pred_q, p=2, dim=1, eps=eps)
    target_q = F.normalize(target_q, p=2, dim=1, eps=eps)
    dot = torch.sum(pred_q * target_q, dim=1)
    return (1.0 - dot.pow(2)).mean()


class QuaternionGeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        return quaternion_geodesic_loss(pred_q, target_q, eps=self.eps)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.resample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        residual = x if self.resample is None else self.resample(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not channels:
            raise ValueError("`channels` for TCN must have at least one entry.")

        blocks = []
        prev_channels = in_channels
        for i, out_channels in enumerate(channels):
            dilation = 2**i
            blocks.append(
                TCNResidualBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            prev_channels = out_channels

        self.network = nn.Sequential(*blocks)
        self.out_channels = prev_channels
        self.kernel_size = kernel_size
        self.num_blocks = len(channels)

    def receptive_field(self) -> int:
        field = 1
        for i in range(self.num_blocks):
            dilation = 2**i
            field += 2 * (self.kernel_size - 1) * dilation
        return field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: Sequence[int],
        embedding_dim: int,
    ):
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError("`conv_channels` must have exactly 3 entries.")

        layers = []
        prev_c = in_channels
        for out_c in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(prev_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            prev_c = out_c

        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(prev_c, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        return self.proj(x)


class TactileCNNTCN(BaseTactileModel):
    def __init__(
        self,
        input_shape_dict: Dict[str, Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        img_embedding_dim: int = 128,
        base_embedding_dim: int = 32,
        base_hidden_dim: int = 64,
        cnn_channels: Sequence[int] = (16, 32, 64),
        tcn_channels: Sequence[int] = (128, 128, 128),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        head_hidden_dims: Sequence[int] = (128, 64),
        temporal_window: Optional[int] = None,
        normalize_quaternion_output: bool = True,
        default_img_in_channels: int = 2,
    ):
        super().__init__(input_shape_dict, output_shape)

        self.normalize_quaternion_output = normalize_quaternion_output
        self.output_dim = output_shape[0]

        evetac_shape = input_shape_dict["evetac"]
        if len(evetac_shape) == 4:
            img_in_channels = evetac_shape[1]
        elif len(evetac_shape) == 1:
            # Sparse temporal processed data may expose only window size here
            # (e.g., evetac shape [T]); channels are defined by Dataset densification.
            img_in_channels = default_img_in_channels
        elif len(evetac_shape) >= 2:
            img_in_channels = 1
        else:
            raise ValueError(
                f"Unexpected `evetac` shape {evetac_shape}. Expected image-like input "
                "or sparse temporal shape [T]."
            )

        self.image_encoder = CNNFeatureExtractor(
            in_channels=img_in_channels,
            conv_channels=cnn_channels,
            embedding_dim=img_embedding_dim,
        )

        self.base_encoder = nn.Sequential(
            nn.Linear(input_shape_dict["base"][-1], base_hidden_dim),
            nn.ReLU(),
            nn.Linear(base_hidden_dim, base_embedding_dim),
            nn.ReLU(),
        )

        temporal_in_dim = img_embedding_dim + base_embedding_dim
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
    def _map_tactile_values(evetac: torch.Tensor) -> torch.Tensor:
        """
        Map tactile values:
            127 -> 0
              0 -> -1
            255 -> +1
        """
        mapped = torch.zeros_like(evetac, dtype=torch.float32)
        mapped = torch.where(evetac == 0, torch.full_like(mapped, -1.0), mapped)
        mapped = torch.where(evetac == 255, torch.full_like(mapped, 1.0), mapped)
        return mapped

    @staticmethod
    def _looks_like_discrete_evtac_codes(evetac: torch.Tensor) -> bool:
        """
        Heuristic: checks if single-channel tensor resembles discrete {0,127,255}.
        """
        if evetac.shape[2] != 1:
            return False
        flat = evetac.reshape(-1)
        if flat.numel() == 0:
            return False
        stride = max(1, flat.numel() // 4096)
        sample = flat[::stride]
        valid = (sample == 0) | (sample == 127) | (sample == 255)
        return bool(torch.all(valid).item())

    @staticmethod
    def _ensure_sequence_image_shape(
        evetac: torch.Tensor, base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # evetac -> (B, T, C, H, W)
        if evetac.ndim == 5:
            pass
        elif evetac.ndim == 4:
            if base.ndim == 3 and evetac.shape[1] == base.shape[1]:
                evetac = evetac.unsqueeze(2)
            else:
                evetac = evetac.unsqueeze(1)
        elif evetac.ndim == 3:
            evetac = evetac.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(
                f"Unexpected `evetac` ndim={evetac.ndim}. Expected 3D/4D/5D tensor."
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

        evetac, base = self._ensure_sequence_image_shape(evetac, base)
        if self._looks_like_discrete_evtac_codes(evetac):
            evetac = self._map_tactile_values(evetac)

        batch_size, time_steps, channels, height, width = evetac.shape
        evetac_flat = evetac.reshape(batch_size * time_steps, channels, height, width)
        base_flat = base.reshape(batch_size * time_steps, -1)

        img_feat = self.image_encoder(evetac_flat).reshape(batch_size, time_steps, -1)
        base_feat = self.base_encoder(base_flat).reshape(batch_size, time_steps, -1)

        temporal_features = torch.cat([img_feat, base_feat], dim=-1)
        temporal_features = temporal_features.transpose(1, 2)

        tcn_out = self.temporal_model(temporal_features)
        last_timestep_feat = tcn_out[:, :, -1]

        out = self.regression_head(last_timestep_feat)
        if self.normalize_quaternion_output and self.output_dim == 4:
            out = F.normalize(out, p=2, dim=1, eps=1e-8)
        return out
