import torch
from datetime import datetime


def compute_translation_matrix(translation):
    N = translation.shape[0]
    device = translation.device
    matrix = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1) # Shape (N, 4, 4)
    matrix[:, 0, 3] = translation[:, 0].squeeze(1)
    matrix[:, 1, 3] = translation[:, 1].squeeze(1)
    matrix[:, 2, 3] = translation[:, 2].squeeze(1)
    return matrix


def compute_rotation_matrix(angles):
    N = angles.shape[0]
    device = angles.device
    alpha = angles[:, 0].squeeze(1) # Rotation around Z
    beta = angles[:, 1].squeeze(1)  # Rotation around Y
    gamma = angles[:, 2].squeeze(1) # Rotation around X
    
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    matrix = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1) # Shape (N, 4, 4)

    matrix[:, 0, 0] = cos_alpha * cos_beta
    matrix[:, 0, 1] = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma
    matrix[:, 0, 2] = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma

    matrix[:, 1, 0] = sin_alpha * cos_beta
    matrix[:, 1, 1] = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
    matrix[:, 1, 2] = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma

    matrix[:, 2, 0] = -sin_beta
    matrix[:, 2, 1] = cos_beta * sin_gamma
    matrix[:, 2, 2] = cos_beta * cos_gamma
    return matrix

def datetime_from_seconds(seconds):
    zerodatetime = datetime.fromtimestamp(0)
    nowdatetime = datetime.fromtimestamp(seconds)
    return nowdatetime - zerodatetime