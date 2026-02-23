import torch
import torch.nn.functional as F


def identity_transform(x):
    return F.normalize(x, p=2, dim=1)


def euler_to_vector(angles):
    # Exemplo: Azimute e Elevação para Vetor Unitário
    az, el = angles[:, 0], angles[:, 1]
    x = torch.cos(el) * torch.cos(az)
    y = torch.cos(el) * torch.sin(az)
    z = torch.sin(el)
    return torch.stack([x, y, z], dim=1)
