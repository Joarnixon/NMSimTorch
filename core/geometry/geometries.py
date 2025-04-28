import torch
from abc import ABC, abstractmethod
from core.numba_backend.ray_cast import numba_ray_casting
from torch import inf
import numpy as np
import time

def torch_check_inside(position, half_size):
    return torch.max(torch.abs(position) - half_size, dim=-1)[0] <= 0

@torch.jit.script
def torch_ray_casting(position, direction, half_size, distance_epsilon: float):
    inside = torch.max(torch.abs(position) - half_size, dim=1)[0] <= 0
    norm_pos = -position/direction
    norm_size = torch.abs(half_size/direction)
    tmin = torch.max(norm_pos - norm_size, dim=1)[0]
    tmax = torch.min(norm_pos + norm_size, dim=1)[0]
    distance = torch.where(tmax > tmin, tmin, torch.full_like(tmin, float('inf')))
    distance[inside] = tmax[inside]
    distance[distance < 0] = torch.tensor(float('inf'), device=distance.device)
    distance += distance_epsilon
    return distance, inside

class Geometry(ABC):
    def __init__(self, size, device=None, dtype=torch.float32):
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        elif device is None:
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid device type: {type(device)}")
        self.size = size.to(self.device).to(dtype)
        self.dtype = dtype

    @property
    def half_size(self):
        return self.size/2

    @property
    def quarter_size(self):
        return self.size/4

    @abstractmethod
    def check_outside(self, position):
        pass

    @abstractmethod
    def check_inside(self, position):
        pass

    @abstractmethod
    def cast_path(self, position, direction):
        pass
    

class Box(Geometry):
    def __init__(self, size, **kwds):
        """
        Size: Tensor with 3 dimension shapes. Batched geometry not implemented.
        """
        super().__init__(size, **kwds)
        self.distance_method = 'ray_casting'
        dtype = torch.get_default_dtype()
        if (dtype == torch.float16) or (dtype == torch.bfloat16):
            self.distance_epsilon = 0.5
        else:
            self.distance_epsilon = 0.001   # don't change that. 5 hours of debugging for that
        args = [
            'distance_method',
            'distance_epsilon'
        ]

        for arg in args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

    def check_outside(self, position):
        return torch.max(torch.abs(position) - self.half_size, dim=1)[0] > 0

    def check_inside(self, position):
        return torch_check_inside(position, self.half_size)

    def cast_path(self, position, direction):
        return getattr(self, self.distance_method)(position, direction)

    def ray_marching(self, position, *args):
        """
        Not tested
        """
        q = torch.abs(position) - self.half_size
        maxXYZ = torch.max(q, dim=1)[0]
        lengthq = torch.norm(torch.where(q > 0, q, torch.zeros_like(q)), dim=1)
        distance = lengthq + torch.where(maxXYZ < 0, maxXYZ, torch.zeros_like(maxXYZ))
        inside = distance < 0
        distance = torch.abs(distance) + self.distance_epsilon
        return distance, inside

    def ray_casting(self, position, direction):
        if self.device.type == 'cpu':
            # basically unbeatable on cpu
            distance, inside = numba_ray_casting(position.to(torch.float32).numpy(), direction.to(torch.float32).cpu().numpy(), np.array(self.half_size.to(torch.float32)), self.distance_epsilon)
            distance = torch.from_numpy(distance).to(self.dtype)
            inside = torch.from_numpy(inside).to(torch.bool)
            return distance, inside
        else:
            return torch_ray_casting(position, direction, self.half_size, self.distance_epsilon)