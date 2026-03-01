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


def quaternion_to_vector(quat):
    """
    Converts quaternion (x,y,z,w) into the direction of the rotated z-axis.
    """
    q = F.normalize(quat, p=2, dim=1)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    vx = 2.0 * (x * z + w * y)
    vy = 2.0 * (y * z - w * x)
    vz = 1.0 - 2.0 * (x * x + y * y)
    v = torch.stack([vx, vy, vz], dim=1)
    return F.normalize(v, p=2, dim=1)


def quaternion_geodesic_deg(pred_q, target_q, eps: float = 1e-8):
    pred = F.normalize(pred_q, p=2, dim=1, eps=eps)
    target = F.normalize(target_q, p=2, dim=1, eps=eps)
    dot = torch.sum(pred * target, dim=1).abs().clamp(-1.0 + eps, 1.0 - eps)
    angle_rad = 2.0 * torch.acos(dot)
    return torch.rad2deg(angle_rad)
