import logging
import threading as mt
import time
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
import contextlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import queue
from math import log
from functools import cache, partial
from itertools import count
from pathlib import Path
from signal import SIGINT, signal
from typing import Iterable, TypeVar, Union
import numba
import numpy as np
import torch
import torch.nn.functional as F
from cProfile import runctx
from h5py import File, string_dtype
from hepunits import *
from tqdm import tqdm
from torch import Tensor, inf


atomic_number = {
    'H':    1,
    'He':   2,
    'Li':   3,
    'Be':   4,
    'B':    5,
    'C':    6,
    'N':    7,
    'O':    8,
    'F':    9,
    'Ne':   10,
    'Na':   11,
    'Mg':   12,
    'Al':   13,
    'Si':   14,
    'P':    15,
    'S':    16,
    'Cl':   17,
    'Ar':   18,
    'K':    19,
    'Ca':   20,
    'Sc':   21,
    'Ti':   22,
    'V':    23,
    'Cr':   24,
    'Mn':   25,
    'Fe':   26,
    'Co':   27,
    'Ni':   28,
    'Cu':   29,
    'Zn':   30,
    'Ga':   31,
    'Ge':   32,
    'As':   33,
    'Se':   34,
    'Br':   35,
    'Kr':   36,
    'Rb':   37,
    'Sr':   38,
    'Y':    39,
    'Zr':   40,
    'Nb':   41,
    'Mo':   42,
    'Tc':   43,
    'Ru':   44,
    'Rh':   45,
    'Pd':   46,
    'Ag':   47,
    'Cd':   48,
    'In':   49,
    'Sn':   50,
    'Sb':   51,
    'Te':   52,
    'I':    53,
    'Xe':   54,
    'Cs':   55,
    'Ba':   56,
    'La':   57,
    'Ce':   58,
    'Pr':   59,
    'Nd':   60,
    'Pm':   61,
    'Sm':   62,
    'Eu':   63,
    'Gd':   64,
    'Tb':   65,
    'Dy':   66,
    'Ho':   67,
    'Er':   68,
    'Tm':   69,
    'Yb':   70,
    'Lu':   71,
    'Hf':   72,
    'Ta':   73,
    'W':    74,
    'Re':   75,
    'Os':   76,
    'Ir':   77,
    'Pt':   78,
    'Au':   79,
    'Hg':   80,
    'Tl':   81,
    'Pb':   82,
    'Bi':   83,
    'Po':   84,
    'At':   85,
    'Rn':   86,
    'Fr':   87,
    'Ra':   88,
    'Ac':   89,
    'Th':   90,
    'Pa':   91,
    'U':    92
}

element_symbol = {value: name for name, value in atomic_number.items()}
elements = atomic_number.keys()

@numba.njit(parallel=True)
def numba_ray_casting(position, direction, half_size, distance_epsilon):
    """
    Numba-accelerated implementation of ray casting.
    
    Parameters:
    -----------
    position : numpy.ndarray
        Array of shape (n, 3) containing the positions
    direction : numpy.ndarray
        Array of shape (n, 3) containing the directions
    half_size : numpy.ndarray
        Array of shape (3,) containing the half-sizes of the box
    distance_epsilon : float
        Small epsilon value to avoid numerical issues
        
    Returns:
    --------
    distance : numpy.ndarray
        Array of shape (n,) containing the distances
    inside : numpy.ndarray
        Array of shape (n,) containing boolean values indicating if the position is inside
    """
    n = position.shape[0]
    distance = np.full(n, np.inf)
    inside = np.zeros(n, dtype=numba.boolean)

    for i in numba.prange(n):
        pos = position[i]
        dir = direction[i]

        is_inside = True
        for j in range(3):
            if abs(pos[j]) > half_size[j]:
                is_inside = False
                break

        inside[i] = is_inside

        t_min = -np.inf
        t_max = np.inf
        
        for j in range(3):
            if abs(dir[j]) < 1e-10:
                if pos[j] > half_size[j] or pos[j] < -half_size[j]:
                    t_max = -np.inf
                    break
            else:
                inv_dir = 1.0 / dir[j]
                t1 = (-half_size[j] - pos[j]) * inv_dir
                t2 = (half_size[j] - pos[j]) * inv_dir

                if t1 > t2:
                    t1, t2 = t2, t1

                t_min = max(t_min, t1)
                t_max = min(t_max, t2)

                if t_min > t_max:
                    t_max = -np.inf
                    break

        if is_inside:
            distance[i] = t_max + distance_epsilon
        else:
            if t_max > 0 and t_min < t_max:
                distance[i] = max(0.0, t_min) + distance_epsilon

    return distance, inside

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

class NonuniqueArray:
    """
    A wrapper class for storing elements by reference indices in a PyTorch tensor
    """
    
    def __init__(self, shape, device=None, element_list=None):
        self.data = torch.zeros(shape, dtype=torch.uint8, device=device)
        self.element_list = element_list if element_list is not None else [None]
        self.device = device

    @property
    def shape(self):
        return self.data.shape
    
    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        return self
    
    @classmethod
    def from_tensor(cls, tensor, element_list=None):
        obj = cls(tensor.shape, tensor.device, element_list)
        obj.data = tensor.clone()
        return obj
    
    def clone(self):
        """Create a copy of this NonuniqueArray"""
        new_obj = NonuniqueArray(self.data.shape, self.data.device, self.element_list)
        new_obj.data = self.data.clone()
        return new_obj
    
    def to(self, device):
        """Move to a different device"""
        self.data = self.data.to(device)
        self.device = device
        return self
    
    def __contains__(self, value):
        return value in self.element_list
    
    def __getitem__(self, key):
        sliced_data = self.data[key]
        new_obj = self.__class__(sliced_data.shape, device=self.data.device)
        new_obj.data = sliced_data
        new_obj.element_list = self.element_list
        return new_obj
    
    def __setitem__(self, key, value):
        if isinstance(value, NonuniqueArray):
            # Получаем подмассив целевых позиций
            subarray = self.data[key]
            for element, mask in value.inverse_indices.items():
                if element not in self.element_list:
                    self.element_list.append(element)
                element_index = self.element_list.index(element)

                # Выставляем элемент только в тех позициях, где mask == True
                subarray[mask] = element_index

            self.data[key] = subarray
            return

        if value not in self.element_list:
            self.element_list.append(value)
        element_index = self.element_list.index(value)
        self.data[key] = element_index
        
    def get_value(self, key):
        """Get indices at specified positions"""
        return self.data[key]
    
    def restore(self):
        """Convert indices back to actual elements"""
        result = []
        for i in range(len(self.element_list)):
            mask = self.data == i
            if mask.any():
                result.append((mask, self.element_list[i]))
        return result
    
    def type_matching(self, type_cls):
        """Return mask where elements match the given type"""
        match = torch.zeros_like(self.data, dtype=torch.bool)
        for index, element in enumerate(self.element_list):
            if isinstance(element, type_cls):
                match |= (self.data == index)
        return match
    
    def matching(self, value):
        """Return mask where elements match the given value"""
        if value not in self:
            return torch.zeros_like(self.data, dtype=torch.bool)
        index = self.element_list.index(value)
        return self.data == index
    
    @property
    def inverse_indices(self):
        """Return dictionary of {element: mask} pairs"""
        inverse_dict = {}
        for index, element in enumerate(self.element_list):
            mask = (self.data == index)
            inverse_dict[element] = mask
        return inverse_dict

@dataclass(eq=True, frozen=True)
class Material:
    """ Класс материала """
    name: str = 'Vacuum'
    type: str = ''
    density: float = 0.4*(10**(-29))*g/cm3
    composition: namedtuple = namedtuple('composition', ['H'])(H=1.)
    ZtoA_ratio: float = 0.
    ID: int = 0

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density < other.Zeff*other.density
        return False

    def __le__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density <= other.Zeff*other.density
        return False

    def __gt__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density > other.Zeff*other.density
        return True

    def __ge__(self, other):
        if isinstance(other, Material):
            return self.Zeff*self.density >= other.Zeff*other.density
        return True

    @property
    @cache
    def Zeff(self):
        Zeff = 0
        for element, weight in self.composition_dict.items():
            Zeff += atomic_number[element]*weight
        return Zeff
    
    @property
    @cache
    def composition_dict(self):
        return self.composition._asdict()
    
    @property
    @cache
    def composition_array(self):
        composition_array = np.zeros(shape=93, dtype=float)
        for element, weight in self.composition_dict.items():
            Z = atomic_number[element]
            composition_array[Z] = weight
        return composition_array


class MaterialArray(NonuniqueArray):
    """ 
    Array containing material at each voxel.
    Ensure that number of materials is within range of uint8.
    """
    
    def __init__(self, shape, device="cpu", element_list=None):
        if element_list is None:
            element_list = [Material()]
        super().__init__(shape, device, element_list)
        self.device = device
        self.dtype = torch.get_default_dtype()
        if self.dtype == torch.float16:
            self.dtype = torch.float32  # density is out of range of float16, not recommended.
    
    @property
    def material_list(self):
        return self.element_list
    
    @property
    def Zeff(self):
        """Get Zeff, dtype=uint8
        
        Works not supreme efficiently, needs more work.
        """
        
        # Or like in original - zeros_like, resulting in NonuniqueArray?
        Zeff = torch.zeros(self.data.shape, dtype=self.dtype, device=self.device)
        
        for material, indices in self.inverse_indices.items():
            Zeff = Zeff.masked_fill(indices, material.Zeff)

        return Zeff

    @property
    def density(self):
        """Get densities for each material in a volume, dtype=float32
        
        Works not supreme efficiently, needs more work.
        """
        
        # Or like in original - zeros_like, resulting in NonuniqueArray?
        density = torch.zeros(self.data.shape, dtype=self.dtype, device=self.device)
        
        for material, indices in self.inverse_indices.items():
            density = density.masked_fill(indices, material.density)
        
        return density
    
    def get_materials_at_indices(self, indices):
        num_materials = len(self.element_list)
        material_masks = torch.stack([
            (self.data == i) for i in range(num_materials)
        ], dim=0)
        return MaterialArray.from_tensor(self._get_materials_at_indices(material_masks, indices, indices.shape[0], indices.shape[1]), self.element_list)

    @staticmethod
    @torch.jit.script
    def _get_materials_at_indices(masks, indices, batch_size: int, num_indices: int):
        x_coords = indices[..., 0]
        y_coords = indices[..., 1]
        z_coords = indices[..., 2]
        material_values = masks[:, x_coords, y_coords, z_coords].to(torch.uint8)

        indices_tensor = torch.arange(material_values.shape[0], device=masks.device, dtype=torch.uint8)
        weighted_sum = torch.mul(material_values, indices_tensor.view(-1, 1)).sum(dim=0, dtype=torch.uint8)
        return weighted_sum

class MaterialDataBase(dict):
    """ Класс базы данных материалов """

    counter = count(1)

    def __init__(self, base_name = 'NIST Materials'):
        self._base_name = base_name
        material = Material()
        self.update({material.name: material})
        self._load_materials()
        
    @property
    def base_name(self):
        return self._base_name
    
    @base_name.setter
    def base_name(self, value):
        self._base_name = value
        self._load_materials()

    def _load_materials(self):
        file = File(f'tables/{self._base_name}.h5', 'r')
        for group_name, material_type_group in file.items():
            for material_name, material_group in material_type_group.items():
                type = group_name
                density = float(np.copy(material_group['Density']))
                composition_dict = {}
                if group_name == 'Elemental media':
                    Z = int(np.copy(material_group['Z']))
                    composition_dict.update({element_symbol[Z]: 1.})
                else:
                    for element, weight in material_group['Composition'].items():
                        composition_dict.update({element: float(np.copy(weight))})
                try:
                    ZtoA_ratio = float(np.copy(material_group['Z\\A']))
                except:
                    # print(f'Для {material_name} отсутствует Z\\A')
                    ZtoA_ratio = 0.5
                ID = next(self.counter)
                composition_dict = namedtuple('composition', composition_dict)(**composition_dict)
                material = Material(material_name, type, density, composition_dict, ZtoA_ratio, ID)
                self.update({material_name: material})

class Interp1d():
    @staticmethod
    def apply(x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            v[name] = vec
            is_flat[name] = v[name].shape[0] == 1

        device = x.device

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes 
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        is_flat['slopes'] = is_flat['x']
        
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)
        return ynew


class AttenuationFunction(dict):
    """Attenuation function class"""

    def __init__(self, process, attenuation_database, device):
        self.__class__.__name__ = self.__class__.__name__ + 'Of' + process.name
        self.__class__.__qualname__ = self.__class__.__qualname__ + 'Of' + process.name
        dtype = torch.get_default_dtype()
        if dtype == torch.float16:
            dtype = torch.float32  # density is out of range of float16, not recommended.
        self.dtype = dtype

        for material, attenuation_data in attenuation_database.items():
            energy = torch.from_numpy(np.copy(attenuation_data['Energy'])).to(dtype).to(device)
            attenuation_coefficient = torch.tensor(
                attenuation_data['Coefficient'][process.name] * material.density, 
                dtype=dtype,
                device=device
            )

            lower_limit = torch.searchsorted(energy, process.energy_range[0].clone(), side='left')
            upper_limit = torch.searchsorted(energy, process.energy_range[1].clone(), side='right')
            
            
            energy = energy[lower_limit:upper_limit]
            attenuation_coefficient = attenuation_coefficient[lower_limit:upper_limit]
            attenuation_function = partial(
                Interp1d.apply,
                energy.unsqueeze(0),
                attenuation_coefficient.unsqueeze(0)
            )
            
            self.update({material.name: attenuation_function})
    
    def __call__(self, material, energy):
        """Get linear attenuation coefficient"""
        mass_coefficient = torch.zeros_like(energy, dtype=self.dtype)        
        for material_type, mask in material.inverse_indices.items():
            if torch.any(mask):
                mass_coefficient = mass_coefficient.masked_scatter(mask, self[material_type.name](torch.masked_select(energy, mask)))
        return mass_coefficient
    
class ParticleProperties(ABC):
    """Базовый класс для свойств частиц"""


    @property
    def type(self) -> Tensor:
        pass

    @property
    def position(self) -> Tensor:
        pass

    @property
    def direction(self) -> Tensor:
        pass

    @property
    def energy(self) -> Tensor:
        pass

    @property
    def emission_time(self) -> Tensor:
        pass

    @property
    def emission_position(self) -> Tensor:
        pass

    @property
    def distance_traveled(self) -> Tensor:
        pass

    @property
    def ID(self) -> Tensor:
        pass
    
    @property
    def valid(self) -> Tensor:
        pass

class ParticleBatch(ParticleProperties):
    """Класс для работы с батчами частиц"""
    _count = 0

    def __init__(self,
                 type: Tensor,
                 position: Tensor,
                 direction: Tensor,
                 energy: Tensor,
                 emission_time: Tensor = None,
                 emission_position: Tensor = None,
                 distance_traveled: Tensor = None,
                 ID: Tensor = None,
                 valid: Tensor = None):
        self._type = type
        self._position = position
        self._direction = direction
        self._energy = energy
        self.size = energy.size()
        self.batch_size, self.num_particles = energy.shape[:2]
        
        self._emission_time = emission_time if emission_time is not None else torch.zeros_like(energy)
        self._emission_position = position.clone() if emission_position is None else emission_position
        self._distance_traveled = torch.zeros_like(energy) if distance_traveled is None else distance_traveled
        self._valid = valid if valid is not None else torch.ones_like(energy, dtype=torch.bool)
        
        if ID is None:
            self._ID = torch.zeros((self.batch_size, self.num_particles), dtype=torch.int64, device=energy.device)
            
            total_particles = self.batch_size * self.num_particles
            self._ID = torch.arange(ParticleBatch._count, 
                                  ParticleBatch._count + total_particles,
                                  dtype=torch.int64, device=energy.device).reshape(self.batch_size, self.num_particles)
            ParticleBatch._count += total_particles
        else:
            self._ID = ID

    @property
    def type(self) -> Tensor:
        return self._type
    
    @property
    def position(self) -> Tensor:
        return self._position
    
    @property
    def direction(self) -> Tensor:
        return self._direction
    
    @property
    def energy(self) -> Tensor:
        return self._energy
    
    @property
    def emission_time(self) -> Tensor:
        return self._emission_time
    
    @property
    def emission_position(self) -> Tensor:
        return self._emission_position
    
    @property
    def distance_traveled(self) -> Tensor:
        return self._distance_traveled
    
    @property
    def ID(self) -> Tensor:
        return self._ID
    
    @property
    def valid(self) -> Tensor:
        return self._valid

    def __len__(self):
        return self._energy.shape[0]

    def __getitem__(self, idx):
        return ParticleBatch(
            type=self._type[idx].unsqueeze(0),
            position=self._position[idx].unsqueeze(0),
            direction=self._direction[idx].unsqueeze(0),
            energy=self._energy[idx].unsqueeze(0),
            emission_time=self._emission_time[idx].unsqueeze(0),
            emission_position=self._emission_position[idx].unsqueeze(0),
            distance_traveled=self._distance_traveled[idx].unsqueeze(0),
            ID=self._ID[idx].unsqueeze(0),
            valid=self._valid[idx].unsqueeze(0)
        )

    def move(self, distance: Tensor):
        """Move particles"""
        self._distance_traveled += distance
        self._position += self._direction * distance.unsqueeze(-1)

    def change_energy(self, delta_energy: Tensor):
        """Change energy of particles"""
        self._energy -= delta_energy

    def rotate(self, theta: Tensor, phi: Tensor):
        """
        Optimized version of rotate function.
        
        [theta] = radian
        [phi] = radian
        """
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        delta1 = sin_theta * cos_phi
        delta2 = sin_theta * sin_phi
        
        z_sign = torch.sign(self._direction[:, :, 2])
        abs_z = torch.abs(self._direction[:, :, 2])
        
        b = self._direction[:, :, 0] * delta1 + self._direction[:, :, 1] * delta2
        tmp = cos_theta - b / (1 + abs_z)
        
        new_dir = torch.empty_like(self._direction)
        new_dir[:, :, 0] = self._direction[:, :, 0] * tmp + delta1
        new_dir[:, :, 1] = self._direction[:, :, 1] * tmp + delta2
        new_dir[:, :, 2] = self._direction[:, :, 2] * cos_theta - z_sign * b
        
        self._direction = new_dir

    def to(self, device: torch.device):
        """Перенос всех тензоров на указанное устройство"""
        return ParticleBatch(
            type=self._type.to(device),
            position=self._position.to(device),
            direction=self._direction.to(device),
            energy=self._energy.to(device),
            emission_time=self._emission_time.to(device),
            emission_position=self._emission_position.to(device),
            distance_traveled=self._distance_traveled.to(device),
            ID=self._ID.to(device),
            valid=self._valid.to(device)
        )
    
    def apply_mask(self, mask: Tensor) -> "ParticleBatch":
        """Выбирает частицы по маске и возвращает новый ParticleBatch"""
        return ParticleBatch(
            type=self._type[mask],
            position=self._position[mask],
            direction=self._direction[mask],
            energy=self._energy[mask],
            emission_time=self._emission_time[mask],
            emission_position=self._emission_position[mask],
            distance_traveled=self._distance_traveled[mask],
            ID=self._ID[mask],
            valid=self._valid[mask]
        )

    def replace_with_new(self, mask: Tensor, new_particles: "ParticleBatch"):
        """Заменяет частицы по маске на новые"""
        self._type[mask] = new_particles.type
        self._position[mask] = new_particles.position
        self._direction[mask] = new_particles.direction
        self._energy[mask] = new_particles.energy
        self._emission_time[mask] = new_particles.emission_time
        self._emission_position[mask] = new_particles.emission_position
        self._distance_traveled[mask] = new_particles.distance_traveled
        self._ID[mask] = new_particles.ID
        self._valid[mask] = new_particles.valid

    @classmethod
    def cat(cls, batches: list):
        """Конкатенация нескольких батчей"""
        return cls(
            type=torch.cat([b._type for b in batches], dim=1),
            position=torch.cat([b._position for b in batches], dim=1),
            direction=torch.cat([b._direction for b in batches], dim=1),
            energy=torch.cat([b._energy for b in batches], dim=1),
            emission_time=torch.cat([b._emission_time for b in batches], dim=1),
            emission_position=torch.cat([b._emission_position for b in batches], dim=1),
            distance_traveled=torch.cat([b._distance_traveled for b in batches], dim=1),
            ID=torch.cat([b._ID for b in batches], dim=1),
            valid=torch.cat([b._valid for b in batches], dim=1)
        )


class ParticleInteractionData:
    """Class for saving the interaction data of particles"""
    def __init__(self,
                 particle_batch: ParticleBatch,
                 process_name: str | list[str],
                 energy_deposit: Tensor,
                 scattering_angles: Tensor,
                 batch_indices: Tensor):

        self.particle_batch = particle_batch

        self.process_name = process_name
        self.energy_deposit = energy_deposit
        self.scattering_angles = scattering_angles
        self.batch_indices = batch_indices

    @property
    def size(self):
        return self.particle_batch.size

    @classmethod
    def cat(cls, interactions: list):
        """Concatenate interactions from list"""
        return cls(
            ParticleBatch.cat([interaction.particle_batch for interaction in interactions]),
            process_name=sum([list(b.process_name) for b in interactions], start=[]),
            energy_deposit=torch.cat([b.energy_deposit for b in interactions], dim=1),
            scattering_angles=torch.cat([b.scattering_angles for b in interactions], dim=2),
            batch_indices=torch.cat([b.batch_indices for b in interactions], dim=0))


class Source:
    """
    Class of a particle source
    """
    def __init__(self, distribution, batch_size, activity=None, voxel_size=4*mm, radiation_type='Gamma',
                 energy=140.5*keV, half_life=6*hour, device=None, rng=None):
        """
        Only same source across batch dimension supported.

        Args:
            distribution (torch.Tensor): must be of input (W, H, L) with no batch dimension.
            batch_size (int): The number of samples in a batch
            activity (float, optional): The initial activity of the source.
            voxel_size (float, optional): The size of each voxel in the distribution grid.
            radiation_type (str, optional): The type of radiation emitted by the source
            energy (float or list, optional): Pairs of energy values and their probabilities.
            half_life (float, optional): The half-life of the radioactive source.
            device (torch.device, optional): The device to use for computations.
            rng (torch.Generator, optional): A random number generator for sampling.
        """

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.distribution = torch.as_tensor(distribution, device=self.device, dtype=torch.float32)
        self.distribution /= torch.sum(self.distribution)
        self.initial_activity = torch.sum(distribution) if activity is None else torch.as_tensor(activity, device=self.device, dtype=torch.float64)
        self.voxel_size = voxel_size

        self.size = torch.tensor(self.distribution.shape, device=self.device) * self.voxel_size
        self.radiation_type = radiation_type
        
        if not isinstance(energy, list):
            energy = [[energy, 1.0]]
        energy = torch.tensor(energy, device=self.device)
        
        self.energy = {
            "energy": energy[:, 0],
            "probability": energy[:, 1] / torch.sum(energy[:, 1])
        }
        
        self.half_life = half_life
        self.batch_size = batch_size
        self.timer = torch.tensor([0.], dtype=torch.float64, device=self.device).expand(self.batch_size)
        self._generate_emission_table()
        
        self.transformation_matrix = torch.tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], device=self.device)
        
        self.rng = torch.default_generator if rng is None else rng

    def translate(self, x=0., y=0., z=0., in_local=False):
        """Move source. Not tested"""
        translation = torch.tensor([x, y, z], device=self.device)
        translation_matrix = compute_translation_matrix(translation)
        if in_local:
            self.transformation_matrix = self.transformation_matrix @ translation_matrix
        else:
            self.transformation_matrix = translation_matrix @ self.transformation_matrix

    def rotate(self, alpha=0., beta=0., gamma=0., rotation_center=[0., 0., 0.], in_local=False):
        """Rotate source. Not tested"""
        rotation_angles = torch.tensor([alpha, beta, gamma], device=self.device)
        rotation_center = torch.tensor(rotation_center, device=self.device)
        rotation_matrix = compute_translation_matrix(rotation_center)
        rotation_matrix = rotation_matrix @ compute_rotation_matrix(rotation_angles)
        rotation_matrix = rotation_matrix @ compute_translation_matrix(-rotation_center)
        if in_local:
            self.transformation_matrix = self.transformation_matrix @ rotation_matrix
        else:
            self.transformation_matrix = rotation_matrix @ self.transformation_matrix

    def convert_to_global_position(self, position):
        global_position = torch.concat([position, torch.ones(position.shape[0], 1, device=position.device)], dim=-1)
        transformed = global_position @ self.transformation_matrix.T
        return transformed[:, :3]

    def _generate_emission_table(self):
        shape = self.distribution.shape
        x = torch.linspace(0, self.size[-3], shape[-3], device=self.device)
        y = torch.linspace(0, self.size[-2], shape[-2], device=self.device)
        z = torch.linspace(0, self.size[-1], shape[-1], device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        position = torch.stack((grid_x, grid_y, grid_z), dim=3).reshape(-1, 3) - self.size/2
        
        probability = self.distribution.flatten()
        indices = torch.nonzero(probability).squeeze(1)

        self.emission_table = {
            'position': position[indices], 
            'probability': probability[indices]
        }

    @property
    def activity(self):
        return self.initial_activity * (2**(-self.timer/self.half_life))
    
    @property
    def nuclei_number(self):
        return self.activity * self.half_life / torch.log(torch.tensor(2.0, dtype=torch.float64, device=self.device))

    def set_state(self, timer):
        self.timer = torch.tensor(timer, dtype=torch.float64, device=self.device).expand((self.batch_size, ))

    def _generate_energy(self, n):
        indices = torch.multinomial(
            self.energy["probability"], 
            n, 
            replacement=True, 
            generator=self.rng
        )
        return self.energy["energy"][indices]

    def _generate_position(self, n):
        indices = torch.multinomial(
            self.emission_table['probability'], 
            n, 
            replacement=True, 
            generator=self.rng
        )
        position = self.emission_table['position'][indices]
        position += torch.rand(position.shape, device=self.device, generator=self.rng) * self.voxel_size
        position = self.convert_to_global_position(position)
        return position
    
    def _generate_emission_time(self, n_per_batch):        
        dt = torch.zeros_like(self.timer, dtype=torch.float64)
        
        log_2 = torch.log(torch.tensor(2.0, device=self.device, dtype=torch.float64))
        dt = torch.log((self.nuclei_number + n_per_batch.to(torch.float64)) / self.nuclei_number) * self.half_life / log_2
        a = 2**(-self.timer/self.half_life)
        b = 2**(-(self.timer + dt)/self.half_life)
        
        total = n_per_batch.sum().item()

        a_expanded = torch.repeat_interleave(a, n_per_batch)
        b_expanded = torch.repeat_interleave(b, n_per_batch)
        
        alpha = torch.rand(total, device=self.device, generator=self.rng, dtype=torch.float64) * (a_expanded - b_expanded) + b_expanded
        all_emission_times = -torch.log(alpha) * self.half_life / log_2
        return all_emission_times, dt

    def _generate_direction(self, n):
        a1 = torch.rand(n, device=self.device, generator=self.rng)
        a2 = torch.rand(n, device=self.device, generator=self.rng)
        
        cos_alpha = 1 - 2 * a1
        sq = torch.sqrt(1 - cos_alpha**2)
        
        cos_beta = sq * torch.cos(2 * torch.pi * a2)
        cos_gamma = sq * torch.sin(2 * torch.pi * a2)
        
        direction = torch.stack((cos_alpha, cos_beta, cos_gamma), dim=1)
        return direction

    def generate_particles(self, n: Union[torch.Tensor, int]):
        """
        Generates the given amount of particles

        If integer is provided returns a ParticleBatch with shape [batch, N]
        If Tensor is provided returns a ParticleBatch with shape [1, Total]
        """
        if isinstance(n, int):
            batch_size = self.batch_size
            total = n * batch_size
            n = torch.tensor([n], device=self.device, dtype=torch.int64).expand(batch_size)
        else:
            batch_size = 1
            total = n.sum(dim=0).item()  # total across all batches

        num_samples = total // batch_size
        energy = self._generate_energy(total)
        direction = self._generate_direction(total)
        position = self._generate_position(total)
        emission_time, dt = self._generate_emission_time(n)
        self.timer = self.timer + dt

        return ParticleBatch(
            torch.zeros_like(energy).reshape((batch_size, num_samples)),
            position.reshape((batch_size, num_samples, 3)),
            direction.reshape((batch_size, num_samples, 3)),
            energy.reshape((batch_size, num_samples)),
            emission_time.reshape((batch_size, num_samples)))

class Тс99m_MIBI(Source):
    """
    Source 99mТс-MIBI

    [position = (x, y, z)] = cm

    [activity] = Bq

    [distribution] = Tensor[:,:,:]

    [voxel_size] = cm
    """

    def __init__(self, distribution, batch_size, activity=None, voxel_size=4*mm, device=None, rng=None):
        radiation_type = 'Gamma'
        energy = 140.5*keV
        half_life = 6.*hour
        super().__init__(distribution, batch_size, activity, voxel_size, radiation_type, energy, half_life, device, rng)


class I123(Source):
    """
    Source I123

    [position = (x, y, z)] = cm

    [activity] = Bq

    [distribution] = Tensor[:,:,:]

    [voxel_size] = cm
    """

    def __init__(self, distribution, activity=None, voxel_size=4*mm, device=None, rng=None):
        radiation_type = 'Gamma'
        energy = [
            [158.97*keV, 83.0],
            [528.96*keV, 1.39],
            [440.02*keV, 0.428],
            [538.54*keV, 0.382],
            [505.33*keV, 0.316],
            [346.35*keV, 0.126],
        ]
        half_life = 13.27*hour
        super().__init__(distribution, activity, voxel_size, radiation_type, energy, half_life, device, rng)

class AttenuationDataBase(dict):
    """ Класс базы данных коэффициентов ослабления """
    
    def __init__(self, base_name = 'NIST XCOM Elements MAC'):
        self._base_name = base_name
        self._elements_MAC = {}
        self._load_elements_MAC()
    
    @property
    def base_name(self):
        return self._base_name
    
    @base_name.setter
    def base_name(self, value):
        self._base_name = value
        self._load_elements_MAC()

    def _load_elements_MAC(self):
        file = File(f'tables/{self._base_name}.h5', 'r')
        for element, element_group in file.items():
            processes_dict = {key: np.copy(value) for key, value in element_group.items()}
            energy = processes_dict.pop('Energy')
            processes_dict = {processes_names[key]: value for key, value in processes_dict.items() if key in processes_names}
            MAC = np.ndarray(energy.size, dtype=MAC_dtype)
            MAC['Energy'] = energy
            for process, value in processes_dict.items():
                MAC['Coefficient'][process] = value
            self._elements_MAC.update({element: MAC})
    
    def add_material(self, material):
        if isinstance(material, Material):
            self._add_material(material)
            return
        if isinstance(material, Iterable):
            for mat in material:
                self._add_material(mat)
            return
        raise ValueError('Неверный тип')
    
    def _add_material(self, material):
        assert isinstance(material, Material), ValueError('Неверный тип')
        list_of_energy = []
        list_of_shells = []
        list_of_MAC = []
        
        for element, weight in material.composition_dict.items():
            energy = np.copy(self._elements_MAC[element]['Energy'])
            MAC = np.copy(self._elements_MAC[element]['Coefficient'])
            for process in MAC.dtype.fields:
                MAC[process] *= weight
            
            _, indices, counts = np.unique(energy, return_index=True, return_counts=True)
            indices_of_shells = indices[counts > 1]
            energy[indices_of_shells] -= 1*eV
            
            list_of_energy.append(energy)
            list_of_shells.append(energy[indices_of_shells + 1])
            list_of_MAC.append(MAC)
        
        array_of_energy = np.concatenate(list_of_energy)
        array_of_energy = np.unique(array_of_energy)
        
        array_of_MAC = np.ndarray(array_of_energy.size, dtype=MAC_dtype)
        array_of_MAC['Energy'] = array_of_energy
        
        for process in MAC.dtype.fields:
            array_of_MAC['Coefficient'][process] = 0
            for energy, MAC in zip(list_of_energy, list_of_MAC):
                array_of_MAC['Coefficient'][process] += np.interp(array_of_energy, energy, MAC[process])

        self.update({material: array_of_MAC})
    
    
processes_names = {
    'Photoelectric absorption': 'PhotoelectricEffect',
    'Incoherent scattering': 'ComptonScattering',
    'Coherent scattering': 'CoherentScattering'
}
processes_dtype = np.dtype([(name, float) for name in processes_names.values()])
MAC_dtype = np.dtype([('Energy', float), ('Coefficient', processes_dtype)])

material_database = MaterialDataBase()
attenuation_database = AttenuationDataBase()
attenuation_database.add_material(material_database.values())

def initialize_cpu_coherent(rng):
    torch.jit.enable_onednn_fusion(True)
    torch.jit.warnings.filterwarnings('ignore')
    # PP_constants is a dictionary containing PP0-PP8 as tensors
    dtype = torch.get_default_dtype()
    if dtype == torch.float16:
        dtype = torch.float32  # PP3 is out of range of float16, not recommended.
    PP0 = torch.tensor([0, 
    0, 2., 5.21459, 10.2817, 3.66207, 3.63903, 3.71155, 36.5165, 3.43548, 3.40045,     # 1-10 
    2.87811, 3.35541, 3.21141, 2.95234, 3.02524, 126.146, 175.044, 162, 296.833, 300.994,     # 11-20 
    373.186, 397.823, 430.071, 483.293, 2.14885, 335.553, 505.422, 644.739, 737.017, 707.575,     # 21-30 
    3.8094, 505.957, 4.10347, 574.665, 15.5277, 10.0991, 4.95013, 16.3391, 6.20836, 3.52767,     # 31-40 
    2.7763, 2.19565, 12.2802, 965.741, 1011.09, 2.85583, 3.65673, 225.777, 1.95284, 15.775,     # 41-50 
    39.9006, 3.7927, 64.7339, 1323.91, 3.73723, 2404.54, 28.3408, 29.9869, 217.128, 71.7138,     # 51-60 
    255.42, 134.495, 3364.59, 425.326, 449.405, 184.046, 3109.04, 193.133, 3608.48, 152.967,     # 61-70 
    484.517, 422.591, 423.518, 393.404, 437.172, 432.356, 478.71, 455.097, 495.237, 417.8,     # 71-80 
    3367.95, 3281.71, 3612.56, 3368.73, 3407.46, 40.2866, 641.24, 826.44, 3579.13, 4916.44,     # 81-90 
    930.184, 887.945, 3490.96, 4058.6, 3068.1, 3898.32, 1398.34, 5285.18, 1, 872.368,     # 91-100 
    ], dtype=dtype)

    PP1 = torch.tensor([0, 
        1., 2., 3.7724, 2.17924, 11.9967, 17.7772, 23.5265, 23.797, 39.9937, 46.7748,     # 1-10 
        60, 68.6446, 81.7887, 98, 112, 128, 96.7939, 162, 61.5575, 96.4218,     # 11-20 
        65.4084, 83.3079, 96.2889, 90.123, 312, 338, 181.943, 94.3868, 54.5084, 132.819,     # 21-30 
        480, 512, 544, 578, 597.472, 647.993, 682.009, 722, 754.885, 799.974,     # 31-40 
        840, 882, 924, 968, 1012, 1058, 1104, 1151.95, 1199.05, 1250,     # 41-50 
        1300, 1352, 1404, 1458, 1512, 729.852, 1596.66, 1682, 1740, 1800,     # 51-60 
        1605.79, 1787.51, 603.151, 2048, 2112, 1993.95, 334.907, 2312, 885.149, 2337.19,     # 61-70 
        2036.48, 2169.41, 2241.49, 2344.6, 2812, 2888, 2964, 2918.04, 2882.97, 2938.74,     # 71-80 
        2716.13, 511.66, 581.475, 594.305, 672.232, 3657.71, 3143.76, 3045.56, 3666.7, 1597.84,     # 81-90 
        3428.87, 3681.22, 1143.31, 1647.17, 1444.9, 1894.33, 3309.12, 2338.59, 4900, 4856.61,     # 91-100 
    ], dtype=dtype)

    PP2 = torch.tensor([0, 
        0, 0, 0.0130091, 3.53906, 9.34125, 14.5838, 21.7619, 3.68644, 37.5709, 49.8248,     # 1-10 
        58.1219, 72, 83.9999, 95.0477, 109.975, 1.85351, 17.1623, 0, 2.60927, 2.58422,     # 11-20 
        2.4053, 2.86948, 2.63999, 2.58417, 310.851, 2.44683, 41.6348, 44.8739, 49.4746, 59.6053,     # 21-30 
        477.191, 6.04261, 540.897, 3.33531, 612, 637.908, 682.041, 705.661, 759.906, 796.498,     # 31-40 
        838.224, 879.804, 912.72, 2.25892, 1.90993, 1055.14, 1101.34, 926.275, 1200, 1234.23,     # 41-50 
        1261.1, 1348.21, 1340.27, 134.085, 1509.26, 1.60851, 1624, 1652.01, 1523.87, 1728.29,     # 51-60 
        1859.79, 1922, 1.25916, 1622.67, 1663.6, 2178, 1045.05, 2118.87, 267.371, 2409.84,     # 61-70 
        2520, 2592, 2664, 2738, 2375.83, 2455.64, 2486.29, 2710.86, 2862.79, 3043.46,     # 71-80 
        476.925, 2930.63, 2694.96, 3092.96, 3145.31, 3698, 3784, 3872, 675.166, 1585.71,     # 81-90 
        3921.95, 3894.83, 4014.73, 3130.23, 4512, 3423.35, 4701.53, 1980.23, 4900, 4271.02,     # 91-100 
    ], dtype=dtype)

    PP3 = torch.tensor([0, 
        1.53728e-16, 2.95909e-16, 1.95042e-15, 6.24521e-16, 4.69459e-17, 3.1394e-17, 2.38808e-17, 3.59428e-16, 1.2947e-17, 1.01182e-17,     # 1-10 
        6.99543e-18, 6.5138e-18, 5.24063e-18, 4.12831e-18, 4.22067e-18, 2.12802e-16, 3.27035e-16, 2.27705e-16, 1.86943e-15, 8.10577e-16,     # 11-20 
        1.80541e-15, 9.32266e-16, 5.93459e-16, 4.93049e-16, 5.03211e-19, 2.38223e-16, 4.5181e-16, 5.34468e-16, 5.16504e-16, 3.0641e-16,     # 21-30 
        1.24646e-18, 2.13805e-16, 1.21448e-18, 2.02122e-16, 5.91556e-18, 3.4609e-18, 1.39331e-18, 5.47242e-18, 1.71017e-18, 7.92438e-19,     # 31-40 
        4.72225e-19, 2.74825e-19, 4.02137e-18, 1.6662e-16, 1.68841e-16, 4.73202e-19, 7.28319e-19, 3.64382e-15, 1.53323e-19, 4.15409e-18,     # 41-50 
        7.91645e-18, 6.54036e-19, 1.04123e-17, 9.116e-17, 5.97268e-19, 1.23272e-15, 5.83259e-18, 5.42458e-18, 2.20137e-17, 1.19654e-17,     # 51-60 
        2.3481e-17, 1.53337e-17, 8.38225e-16, 3.40248e-17, 3.50901e-17, 1.95115e-17, 2.91803e-16, 1.98684e-17, 3.59425e-16, 1.54e-17,     # 61-70 
        3.04174e-17, 2.71295e-17, 2.6803e-17, 2.36469e-17, 2.56818e-17, 2.50364e-17, 2.6818e-17, 2.56229e-17, 2.7419e-17, 2.27442e-17,     # 71-80 
        1.38078e-15, 1.49595e-15, 1.20023e-16, 1.74446e-15, 1.82836e-15, 5.80108e-18, 3.02324e-17, 3.71029e-17, 1.01058e-16, 4.87707e-16,     # 81-90 
        4.18953e-17, 4.03182e-17, 1.11553e-16, 9.51125e-16, 2.57569e-15, 1.14294e-15, 2.98597e-15, 5.88714e-16, 1.46196e-20, 1.53226e-15,     # 91-100 
    ], dtype=dtype)

    PP4 = torch.tensor([0, 
        1.10561e-15, 3.50254e-16, 1.56836e-16, 7.86286e-15, 2.2706e-16, 7.28454e-16, 4.54123e-16, 8.03792e-17, 4.91833e-16, 1.45891e-16,     # 1-10 
        1.71829e-16, 3.90707e-15, 2.76487e-15, 4.345e-16, 6.80131e-16, 4.04186e-16, 8.95703e-17, 3.32136e-16, 1.3847e-17, 4.16869e-17,     # 11-20 
        1.37963e-17, 1.96187e-17, 2.93852e-17, 2.46581e-17, 4.49944e-16, 3.80311e-16, 1.62925e-15, 7.52449e-16, 9.45445e-16, 5.47652e-16,     # 21-30 
        6.89379e-16, 1.37078e-15, 1.22209e-15, 1.13856e-15, 9.06914e-16, 8.77868e-16, 9.70871e-16, 1.8532e-16, 1.69254e-16, 1.14059e-15,     # 31-40 
        7.90712e-16, 5.36611e-16, 8.27932e-16, 2.4329e-16, 5.82899e-16, 1.97595e-16, 1.96263e-16, 1.73961e-16, 1.62174e-16, 5.31143e-16,     # 41-50 
        5.29731e-16, 4.1976e-16, 4.91842e-16, 4.67937e-16, 4.32264e-16, 6.91046e-17, 1.62962e-16, 9.87241e-16, 1.04526e-15, 1.05819e-15,     # 51-60 
        1.10579e-16, 1.49116e-16, 4.61021e-17, 1.5143e-16, 1.53667e-16, 1.67844e-15, 2.7494e-17, 2.31253e-16, 2.27211e-15, 1.33401e-15,     # 61-70 
        9.02548e-16, 1.77743e-15, 1.76608e-15, 9.45054e-16, 1.06805e-16, 1.06085e-16, 1.01688e-16, 1.0226e-16, 7.7793e-16, 8.0166e-16,     # 71-80 
        9.18595e-17, 2.73428e-17, 3.01222e-17, 3.09814e-17, 3.39028e-17, 1.49653e-15, 1.19511e-15, 1.40408e-15, 2.37226e-15, 8.35973e-17,     # 81-90 
        1.4089e-15, 1.2819e-15, 4.96925e-17, 6.04886e-17, 7.39507e-17, 6.6832e-17, 1.09433e-16, 9.61804e-17, 1.38525e-16, 2.49104e-16,     # 91-100 
    ], dtype=dtype)

    PP5 = torch.tensor([0, 
        6.89413e-17, 2.11456e-17, 2.47782e-17, 7.01557e-17, 1.01544e-15, 1.76177e-16, 1.28191e-16, 1.80511e-17, 1.96803e-16, 3.16753e-16,     # 1-10 
        1.21362e-15, 6.6366e-17, 8.42625e-17, 1.01935e-16, 1.34162e-16, 1.87076e-18, 2.76259e-17, 1.2217e-16, 1.66059e-18, 1.76249e-18,     # 11-20 
        1.13734e-18, 1.58963e-18, 1.33987e-18, 1.18496e-18, 2.44536e-16, 6.69957e-19, 2.5667e-17, 2.62482e-17, 2.55816e-17, 2.6574e-17,     # 21-30 
        2.26522e-16, 2.17703e-18, 2.07434e-16, 8.8717e-19, 1.75583e-16, 1.81312e-16, 1.83716e-16, 2.58371e-15, 1.74416e-15, 1.7473e-16,     # 31-40 
        1.76817e-16, 1.74757e-16, 1.6739e-16, 2.68691e-19, 1.8138e-19, 1.60726e-16, 1.59441e-16, 1.36927e-16, 2.70127e-16, 1.63371e-16,     # 41-50 
        1.29776e-16, 1.49012e-16, 1.17301e-16, 1.67919e-17, 1.47596e-16, 1.14246e-19, 1.10392e-15, 1.58755e-16, 1.11706e-16, 1.80135e-16,     # 51-60 
        1.00213e-15, 9.44133e-16, 4.722e-20, 1.18997e-15, 1.16311e-15, 2.31716e-16, 1.86238e-15, 1.53632e-15, 2.45853e-17, 2.08069e-16,     # 61-70 
        1.08659e-16, 1.29019e-16, 1.24987e-16, 1.07865e-16, 1.03501e-15, 1.05211e-15, 9.38473e-16, 8.66912e-16, 9.3778e-17, 9.91467e-17,     # 71-80 
        2.58481e-17, 9.72329e-17, 9.77921e-16, 1.02928e-16, 1.01767e-16, 1.81276e-16, 1.07026e-16, 1.11273e-16, 3.25695e-17, 1.77629e-15,     # 81-90 
        1.18382e-16, 1.111e-16, 1.56996e-15, 8.45221e-17, 3.6783e-16, 1.20652e-16, 3.91104e-16, 3.52282e-15, 4.29979e-16, 1.28308e-16,     # 91-100 
    ], dtype=dtype)

    PP6 = torch.tensor([0, 
        6.57834, 3.91446, 7.59547, 10.707, 3.97317, 4.00593, 3.93206, 8.10644, 3.97743, 4.04641,     # 1-10 
        4.30202, 4.19399, 4.27399, 4.4169, 4.04829, 2.21745, 11.3523, 1.84976, 1.61905, 3.68297,     # 11-20 
        1.5704, 2.58852, 3.59827, 3.61633, 9.07174, 1.76738, 1.97272, 1.91032, 1.9838, 2.64286,     # 21-30 
        4.16296, 1.80149, 3.94257, 1.72731, 2.27523, 2.57383, 3.33453, 2.2361, 2.94376, 3.91332,     # 31-40 
        5.01832, 6.8016, 2.19508, 1.65926, 1.63781, 4.23097, 3.4399, 2.55583, 7.96814, 2.06573,     # 41-50 
        1.84175, 3.23516, 1.79129, 2.90259, 3.18266, 1.51305, 1.88361, 1.91925, 1.68033, 1.72078,     # 51-60 
        1.66246, 1.66676, 1.49394, 1.58924, 1.57558, 1.63307, 1.84447, 1.60296, 1.56719, 1.62166,     # 61-70 
        1.5753, 1.57329, 1.558, 1.57567, 1.55612, 1.54607, 1.53251, 1.51928, 1.50265, 1.52445,     # 71-80 
        1.4929, 1.51098, 2.52959, 1.42334, 1.41292, 2.0125, 1.45015, 1.43067, 2.6026, 1.39261,     # 81-90 
        1.38559, 1.37575, 2.53155, 2.51924, 1.32386, 2.31791, 2.47722, 1.33584, 9.60979, 6.84949,     # 91-100 
    ], dtype=dtype)

    PP7 = torch.tensor([0, 
        3.99983, 6.63093, 3.85593, 1.69342, 14.7911, 7.03995, 8.89527, 13.1929, 4.93354, 5.59461,     # 1-10 
        3.98033, 1.74578, 2.67629, 14.184, 8.88775, 13.1809, 4.51627, 13.7677, 9.53727, 4.04257,     # 11-20 
        7.88725, 5.78566, 4.08148, 4.18194, 7.96292, 8.38322, 3.31429, 13.106, 13.0857, 13.1053,     # 21-30 
        3.54708, 2.08567, 2.38131, 2.58162, 3.199, 3.20493, 3.19799, 1.88697, 1.80323, 3.15596,     # 31-40 
        4.10675, 5.68928, 3.93024, 11.2607, 4.86595, 12.1708, 12.2867, 9.29496, 1.61249, 5.0998,     # 41-50 
        5.25068, 6.67673, 5.82498, 6.12968, 6.94532, 1.71622, 1.63028, 3.34945, 2.84671, 2.66325,     # 51-60 
        2.73395, 1.93715, 1.72497, 2.74504, 2.71531, 1.52039, 1.58191, 1.61444, 2.67701, 1.51369,     # 61-70 
        2.60766, 1.46608, 1.49792, 2.49166, 2.84906, 2.80604, 2.92788, 2.76411, 2.59305, 2.5855,     # 71-80 
        2.80503, 1.4866, 1.46649, 1.45595, 1.44374, 1.54865, 2.45661, 2.43268, 1.35352, 1.35911,     # 81-90 
        2.26339, 2.26838, 1.35877, 1.37826, 1.3499, 1.36574, 1.33654, 1.33001, 1.37648, 4.28173,     # 91-100 
    ], dtype=dtype)

    PP8 = torch.tensor([0, 
        4, 4, 5.94686, 4.10265, 7.87177, 12.0509, 12.0472, 3.90597, 5.34338, 6.33072,     # 1-10 
        2.76777, 7.90099, 5.58323, 4.26372, 3.3005, 5.69179, 2.3698, 3.68167, 5.2807, 4.61212,     # 11-20 
        5.87809, 4.46207, 4.59278, 4.67584, 1.75212, 7.00575, 2.05428, 2.00415, 2.02048, 1.98413,     # 21-30 
        1.71725, 3.18743, 1.74231, 4.40997, 2.01626, 1.8622, 1.7544, 1.60332, 2.23338, 1.70932,     # 31-40 
        1.67223, 1.64655, 1.76198, 6.33416, 7.92665, 1.67835, 1.67408, 1.55895, 9.3642, 1.68776,     # 41-50 
        2.02167, 1.65401, 2.20616, 1.76498, 1.63064, 7.13771, 3.17033, 1.65236, 2.66943, 1.62703,     # 51-60 
        2.72469, 2.73686, 10.86, 2.76759, 2.69728, 1.62436, 2.76662, 1.48514, 1.57342, 1.61518,     # 61-70 
        3.18455, 2.73467, 2.72521, 2.786, 2.35611, 2.31574, 2.5787, 2.46877, 2.89052, 2.6478,     # 71-80 
        1.50419, 2.73998, 2.79809, 2.66207, 2.73089, 1.34835, 2.59656, 2.7006, 1.41867, 4.26255,     # 81-90 
        2.47985, 2.47126, 1.72573, 3.44856, 1.36451, 2.8715, 2.35731, 1.28196, 4.1224, 1.32633,     # 91-100 
    ], dtype=dtype)
    
    x = cm/(h_Planck*c_light)
    f_factor = 0.5*x*x
    
    def compute_w_values(x_b: torch.Tensor, n: torch.Tensor, numlim: float = 0.02):
        w_small = n * x_b * (1.0 - 0.5 * (n - 1.0) * x_b * (1.0 - (n - 2.0) * x_b / 3.0))
        w_large = 1.0 - torch.exp(-n * torch.log1p(x_b))
        return torch.where(x_b < numlim, w_small, w_large)
    
    def compute_x_value(y: torch.Tensor, n: torch.Tensor, numlim: float = 0.02):
        n_inv = 1.0 / n
        x_small = y * n_inv * (1.0 + 0.5 * (n_inv + 1.0) * y * (1.0 - (n_inv + 2.0) * y / 3.0))
        x_large = torch.exp(-n_inv * torch.log(1.0 - y)) - 1.0
        return torch.where(y < numlim, x_small, x_large)
    
    def theta_generator(energy: torch.Tensor,
                        Z: torch.Tensor,
                        PP0: torch.Tensor=PP0,
                        PP1: torch.Tensor=PP1,
                        PP2: torch.Tensor=PP2,
                        PP3: torch.Tensor=PP3,
                        PP4: torch.Tensor=PP4,
                        PP5: torch.Tensor=PP5,
                        PP6: torch.Tensor=PP6,
                        PP7: torch.Tensor=PP7,
                        PP8: torch.Tensor=PP8,
                        f_factor: float = f_factor,
                        generator: torch.Generator = rng) -> torch.Tensor:
        
        device = energy.device
        original_shape = energy.shape

        batch_size = energy.numel()
        energy_flat = energy.reshape(-1)
        Z_flat = Z.reshape(-1).long()
        
        xx = f_factor * energy_flat.square()
        
        n0 = PP6[Z_flat] - 1.0
        n1 = PP7[Z_flat] - 1.0
        n2 = PP8[Z_flat] - 1.0
        b0 = PP3[Z_flat]
        b1 = PP4[Z_flat]
        b2 = PP5[Z_flat]
        y0 = PP0[Z_flat]
        y1 = PP1[Z_flat]
        y2 = PP2[Z_flat]
        
        x_b0 = 2.0 * xx * b0
        x_b1 = 2.0 * xx * b1
        x_b2 = 2.0 * xx * b2
        
        w0 = compute_w_values(x_b0, n0)
        w1 = compute_w_values(x_b1, n1)
        w2 = compute_w_values(x_b2, n2)
        
        x0 = w0 * y0 / (b0 * n0)
        x1 = w1 * y1 / (b1 * n1)
        x2 = w2 * y2 / (b2 * n2)
        
        cost = torch.zeros(batch_size, device=device, dtype=PP3.dtype)
        
        need_sampling = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_max_iter = 10
        iters = 0
        x_sum = x0 + x1 + x2
        while need_sampling.any() and iters < num_max_iter:
            iters += 1
            indices = torch.where(need_sampling)[0]
            num_need_sampling = indices.shape[0]
            
            r1 = torch.rand(num_need_sampling, device=device, generator=generator)
            r2 = torch.rand(num_need_sampling, device=device, generator=generator)
            r3 = torch.rand(num_need_sampling, device=device, generator=generator)

            xx_need = xx[indices]
            x_sum_need = x_sum[indices]

            w = w0[indices].clone()
            n = n0[indices].clone()
            b = b0[indices].clone()
            
            x_rand = r1 * x_sum_need
            
            mask1 = x_rand > x0[indices]
            mask2 = (x_rand <= (x0[indices] + x1[indices])) & mask1
            mask3 = mask1 & ~mask2
            
            w[mask2] = w1[indices][mask2]
            n[mask2] = n1[indices][mask2]
            b[mask2] = b1[indices][mask2]
            
            w[mask3] = w2[indices][mask3]
            n[mask3] = n2[indices][mask3]
            b[mask3] = b2[indices][mask3]
            
            y = w * r2
            x_1 = compute_x_value(y, n)
            cost_sampled = 1.0 - x_1 / (b * xx_need)
            accepted = (2 * r3 < 1.0 + cost_sampled.square()) | (cost_sampled > -1.0)
            accepted_indices = indices[accepted]
            cost[accepted_indices] = cost_sampled[accepted]
            
            need_sampling[accepted_indices] = False
        
        cost = cost.reshape(original_shape)
        theta = torch.acos(cost)
        
        return theta
    
    return torch.compile(
        torch.jit.script(
            theta_generator,
        ),
        dynamic=True,
        fullgraph=True
    )


def initialize_cuda_coherent(rng, device='cuda'):
    """
    Precaution: please use only when batch_size >= 10 for speed gain.
    """
    dtype = torch.get_default_dtype()
    if dtype == torch.float16:
        dtype = torch.float32  # PP3 is out of range of float16, not recommended.
    PP0 = torch.tensor([0, 
    0, 2., 5.21459, 10.2817, 3.66207, 3.63903, 3.71155, 36.5165, 3.43548, 3.40045,     # 1-10 
    2.87811, 3.35541, 3.21141, 2.95234, 3.02524, 126.146, 175.044, 162, 296.833, 300.994,     # 11-20 
    373.186, 397.823, 430.071, 483.293, 2.14885, 335.553, 505.422, 644.739, 737.017, 707.575,     # 21-30 
    3.8094, 505.957, 4.10347, 574.665, 15.5277, 10.0991, 4.95013, 16.3391, 6.20836, 3.52767,     # 31-40 
    2.7763, 2.19565, 12.2802, 965.741, 1011.09, 2.85583, 3.65673, 225.777, 1.95284, 15.775,     # 41-50 
    39.9006, 3.7927, 64.7339, 1323.91, 3.73723, 2404.54, 28.3408, 29.9869, 217.128, 71.7138,     # 51-60 
    255.42, 134.495, 3364.59, 425.326, 449.405, 184.046, 3109.04, 193.133, 3608.48, 152.967,     # 61-70 
    484.517, 422.591, 423.518, 393.404, 437.172, 432.356, 478.71, 455.097, 495.237, 417.8,     # 71-80 
    3367.95, 3281.71, 3612.56, 3368.73, 3407.46, 40.2866, 641.24, 826.44, 3579.13, 4916.44,     # 81-90 
    930.184, 887.945, 3490.96, 4058.6, 3068.1, 3898.32, 1398.34, 5285.18, 1, 872.368,     # 91-100 
    ], dtype=dtype).to(device)

    PP1 = torch.tensor([0, 
        1., 2., 3.7724, 2.17924, 11.9967, 17.7772, 23.5265, 23.797, 39.9937, 46.7748,     # 1-10 
        60, 68.6446, 81.7887, 98, 112, 128, 96.7939, 162, 61.5575, 96.4218,     # 11-20 
        65.4084, 83.3079, 96.2889, 90.123, 312, 338, 181.943, 94.3868, 54.5084, 132.819,     # 21-30 
        480, 512, 544, 578, 597.472, 647.993, 682.009, 722, 754.885, 799.974,     # 31-40 
        840, 882, 924, 968, 1012, 1058, 1104, 1151.95, 1199.05, 1250,     # 41-50 
        1300, 1352, 1404, 1458, 1512, 729.852, 1596.66, 1682, 1740, 1800,     # 51-60 
        1605.79, 1787.51, 603.151, 2048, 2112, 1993.95, 334.907, 2312, 885.149, 2337.19,     # 61-70 
        2036.48, 2169.41, 2241.49, 2344.6, 2812, 2888, 2964, 2918.04, 2882.97, 2938.74,     # 71-80 
        2716.13, 511.66, 581.475, 594.305, 672.232, 3657.71, 3143.76, 3045.56, 3666.7, 1597.84,     # 81-90 
        3428.87, 3681.22, 1143.31, 1647.17, 1444.9, 1894.33, 3309.12, 2338.59, 4900, 4856.61,     # 91-100 
    ], dtype=dtype).to(device)

    PP2 = torch.tensor([0, 
        0, 0, 0.0130091, 3.53906, 9.34125, 14.5838, 21.7619, 3.68644, 37.5709, 49.8248,     # 1-10 
        58.1219, 72, 83.9999, 95.0477, 109.975, 1.85351, 17.1623, 0, 2.60927, 2.58422,     # 11-20 
        2.4053, 2.86948, 2.63999, 2.58417, 310.851, 2.44683, 41.6348, 44.8739, 49.4746, 59.6053,     # 21-30 
        477.191, 6.04261, 540.897, 3.33531, 612, 637.908, 682.041, 705.661, 759.906, 796.498,     # 31-40 
        838.224, 879.804, 912.72, 2.25892, 1.90993, 1055.14, 1101.34, 926.275, 1200, 1234.23,     # 41-50 
        1261.1, 1348.21, 1340.27, 134.085, 1509.26, 1.60851, 1624, 1652.01, 1523.87, 1728.29,     # 51-60 
        1859.79, 1922, 1.25916, 1622.67, 1663.6, 2178, 1045.05, 2118.87, 267.371, 2409.84,     # 61-70 
        2520, 2592, 2664, 2738, 2375.83, 2455.64, 2486.29, 2710.86, 2862.79, 3043.46,     # 71-80 
        476.925, 2930.63, 2694.96, 3092.96, 3145.31, 3698, 3784, 3872, 675.166, 1585.71,     # 81-90 
        3921.95, 3894.83, 4014.73, 3130.23, 4512, 3423.35, 4701.53, 1980.23, 4900, 4271.02,     # 91-100 
    ], dtype=dtype).to(device)

    PP3 = torch.tensor([0, 
        1.53728e-16, 2.95909e-16, 1.95042e-15, 6.24521e-16, 4.69459e-17, 3.1394e-17, 2.38808e-17, 3.59428e-16, 1.2947e-17, 1.01182e-17,     # 1-10 
        6.99543e-18, 6.5138e-18, 5.24063e-18, 4.12831e-18, 4.22067e-18, 2.12802e-16, 3.27035e-16, 2.27705e-16, 1.86943e-15, 8.10577e-16,     # 11-20 
        1.80541e-15, 9.32266e-16, 5.93459e-16, 4.93049e-16, 5.03211e-19, 2.38223e-16, 4.5181e-16, 5.34468e-16, 5.16504e-16, 3.0641e-16,     # 21-30 
        1.24646e-18, 2.13805e-16, 1.21448e-18, 2.02122e-16, 5.91556e-18, 3.4609e-18, 1.39331e-18, 5.47242e-18, 1.71017e-18, 7.92438e-19,     # 31-40 
        4.72225e-19, 2.74825e-19, 4.02137e-18, 1.6662e-16, 1.68841e-16, 4.73202e-19, 7.28319e-19, 3.64382e-15, 1.53323e-19, 4.15409e-18,     # 41-50 
        7.91645e-18, 6.54036e-19, 1.04123e-17, 9.116e-17, 5.97268e-19, 1.23272e-15, 5.83259e-18, 5.42458e-18, 2.20137e-17, 1.19654e-17,     # 51-60 
        2.3481e-17, 1.53337e-17, 8.38225e-16, 3.40248e-17, 3.50901e-17, 1.95115e-17, 2.91803e-16, 1.98684e-17, 3.59425e-16, 1.54e-17,     # 61-70 
        3.04174e-17, 2.71295e-17, 2.6803e-17, 2.36469e-17, 2.56818e-17, 2.50364e-17, 2.6818e-17, 2.56229e-17, 2.7419e-17, 2.27442e-17,     # 71-80 
        1.38078e-15, 1.49595e-15, 1.20023e-16, 1.74446e-15, 1.82836e-15, 5.80108e-18, 3.02324e-17, 3.71029e-17, 1.01058e-16, 4.87707e-16,     # 81-90 
        4.18953e-17, 4.03182e-17, 1.11553e-16, 9.51125e-16, 2.57569e-15, 1.14294e-15, 2.98597e-15, 5.88714e-16, 1.46196e-20, 1.53226e-15,     # 91-100 
    ], dtype=dtype).to(device)

    PP4 = torch.tensor([0, 
        1.10561e-15, 3.50254e-16, 1.56836e-16, 7.86286e-15, 2.2706e-16, 7.28454e-16, 4.54123e-16, 8.03792e-17, 4.91833e-16, 1.45891e-16,     # 1-10 
        1.71829e-16, 3.90707e-15, 2.76487e-15, 4.345e-16, 6.80131e-16, 4.04186e-16, 8.95703e-17, 3.32136e-16, 1.3847e-17, 4.16869e-17,     # 11-20 
        1.37963e-17, 1.96187e-17, 2.93852e-17, 2.46581e-17, 4.49944e-16, 3.80311e-16, 1.62925e-15, 7.52449e-16, 9.45445e-16, 5.47652e-16,     # 21-30 
        6.89379e-16, 1.37078e-15, 1.22209e-15, 1.13856e-15, 9.06914e-16, 8.77868e-16, 9.70871e-16, 1.8532e-16, 1.69254e-16, 1.14059e-15,     # 31-40 
        7.90712e-16, 5.36611e-16, 8.27932e-16, 2.4329e-16, 5.82899e-16, 1.97595e-16, 1.96263e-16, 1.73961e-16, 1.62174e-16, 5.31143e-16,     # 41-50 
        5.29731e-16, 4.1976e-16, 4.91842e-16, 4.67937e-16, 4.32264e-16, 6.91046e-17, 1.62962e-16, 9.87241e-16, 1.04526e-15, 1.05819e-15,     # 51-60 
        1.10579e-16, 1.49116e-16, 4.61021e-17, 1.5143e-16, 1.53667e-16, 1.67844e-15, 2.7494e-17, 2.31253e-16, 2.27211e-15, 1.33401e-15,     # 61-70 
        9.02548e-16, 1.77743e-15, 1.76608e-15, 9.45054e-16, 1.06805e-16, 1.06085e-16, 1.01688e-16, 1.0226e-16, 7.7793e-16, 8.0166e-16,     # 71-80 
        9.18595e-17, 2.73428e-17, 3.01222e-17, 3.09814e-17, 3.39028e-17, 1.49653e-15, 1.19511e-15, 1.40408e-15, 2.37226e-15, 8.35973e-17,     # 81-90 
        1.4089e-15, 1.2819e-15, 4.96925e-17, 6.04886e-17, 7.39507e-17, 6.6832e-17, 1.09433e-16, 9.61804e-17, 1.38525e-16, 2.49104e-16,     # 91-100 
    ], dtype=dtype).to(device)

    PP5 = torch.tensor([0, 
        6.89413e-17, 2.11456e-17, 2.47782e-17, 7.01557e-17, 1.01544e-15, 1.76177e-16, 1.28191e-16, 1.80511e-17, 1.96803e-16, 3.16753e-16,     # 1-10 
        1.21362e-15, 6.6366e-17, 8.42625e-17, 1.01935e-16, 1.34162e-16, 1.87076e-18, 2.76259e-17, 1.2217e-16, 1.66059e-18, 1.76249e-18,     # 11-20 
        1.13734e-18, 1.58963e-18, 1.33987e-18, 1.18496e-18, 2.44536e-16, 6.69957e-19, 2.5667e-17, 2.62482e-17, 2.55816e-17, 2.6574e-17,     # 21-30 
        2.26522e-16, 2.17703e-18, 2.07434e-16, 8.8717e-19, 1.75583e-16, 1.81312e-16, 1.83716e-16, 2.58371e-15, 1.74416e-15, 1.7473e-16,     # 31-40 
        1.76817e-16, 1.74757e-16, 1.6739e-16, 2.68691e-19, 1.8138e-19, 1.60726e-16, 1.59441e-16, 1.36927e-16, 2.70127e-16, 1.63371e-16,     # 41-50 
        1.29776e-16, 1.49012e-16, 1.17301e-16, 1.67919e-17, 1.47596e-16, 1.14246e-19, 1.10392e-15, 1.58755e-16, 1.11706e-16, 1.80135e-16,     # 51-60 
        1.00213e-15, 9.44133e-16, 4.722e-20, 1.18997e-15, 1.16311e-15, 2.31716e-16, 1.86238e-15, 1.53632e-15, 2.45853e-17, 2.08069e-16,     # 61-70 
        1.08659e-16, 1.29019e-16, 1.24987e-16, 1.07865e-16, 1.03501e-15, 1.05211e-15, 9.38473e-16, 8.66912e-16, 9.3778e-17, 9.91467e-17,     # 71-80 
        2.58481e-17, 9.72329e-17, 9.77921e-16, 1.02928e-16, 1.01767e-16, 1.81276e-16, 1.07026e-16, 1.11273e-16, 3.25695e-17, 1.77629e-15,     # 81-90 
        1.18382e-16, 1.111e-16, 1.56996e-15, 8.45221e-17, 3.6783e-16, 1.20652e-16, 3.91104e-16, 3.52282e-15, 4.29979e-16, 1.28308e-16,     # 91-100 
    ], dtype=dtype).to(device)

    PP6 = torch.tensor([0, 
        6.57834, 3.91446, 7.59547, 10.707, 3.97317, 4.00593, 3.93206, 8.10644, 3.97743, 4.04641,     # 1-10 
        4.30202, 4.19399, 4.27399, 4.4169, 4.04829, 2.21745, 11.3523, 1.84976, 1.61905, 3.68297,     # 11-20 
        1.5704, 2.58852, 3.59827, 3.61633, 9.07174, 1.76738, 1.97272, 1.91032, 1.9838, 2.64286,     # 21-30 
        4.16296, 1.80149, 3.94257, 1.72731, 2.27523, 2.57383, 3.33453, 2.2361, 2.94376, 3.91332,     # 31-40 
        5.01832, 6.8016, 2.19508, 1.65926, 1.63781, 4.23097, 3.4399, 2.55583, 7.96814, 2.06573,     # 41-50 
        1.84175, 3.23516, 1.79129, 2.90259, 3.18266, 1.51305, 1.88361, 1.91925, 1.68033, 1.72078,     # 51-60 
        1.66246, 1.66676, 1.49394, 1.58924, 1.57558, 1.63307, 1.84447, 1.60296, 1.56719, 1.62166,     # 61-70 
        1.5753, 1.57329, 1.558, 1.57567, 1.55612, 1.54607, 1.53251, 1.51928, 1.50265, 1.52445,     # 71-80 
        1.4929, 1.51098, 2.52959, 1.42334, 1.41292, 2.0125, 1.45015, 1.43067, 2.6026, 1.39261,     # 81-90 
        1.38559, 1.37575, 2.53155, 2.51924, 1.32386, 2.31791, 2.47722, 1.33584, 9.60979, 6.84949,     # 91-100 
    ], dtype=dtype).to(device)

    PP7 = torch.tensor([0, 
        3.99983, 6.63093, 3.85593, 1.69342, 14.7911, 7.03995, 8.89527, 13.1929, 4.93354, 5.59461,     # 1-10 
        3.98033, 1.74578, 2.67629, 14.184, 8.88775, 13.1809, 4.51627, 13.7677, 9.53727, 4.04257,     # 11-20 
        7.88725, 5.78566, 4.08148, 4.18194, 7.96292, 8.38322, 3.31429, 13.106, 13.0857, 13.1053,     # 21-30 
        3.54708, 2.08567, 2.38131, 2.58162, 3.199, 3.20493, 3.19799, 1.88697, 1.80323, 3.15596,     # 31-40 
        4.10675, 5.68928, 3.93024, 11.2607, 4.86595, 12.1708, 12.2867, 9.29496, 1.61249, 5.0998,     # 41-50 
        5.25068, 6.67673, 5.82498, 6.12968, 6.94532, 1.71622, 1.63028, 3.34945, 2.84671, 2.66325,     # 51-60 
        2.73395, 1.93715, 1.72497, 2.74504, 2.71531, 1.52039, 1.58191, 1.61444, 2.67701, 1.51369,     # 61-70 
        2.60766, 1.46608, 1.49792, 2.49166, 2.84906, 2.80604, 2.92788, 2.76411, 2.59305, 2.5855,     # 71-80 
        2.80503, 1.4866, 1.46649, 1.45595, 1.44374, 1.54865, 2.45661, 2.43268, 1.35352, 1.35911,     # 81-90 
        2.26339, 2.26838, 1.35877, 1.37826, 1.3499, 1.36574, 1.33654, 1.33001, 1.37648, 4.28173,     # 91-100 
    ], dtype=dtype).to(device)

    PP8 = torch.tensor([0, 
        4, 4, 5.94686, 4.10265, 7.87177, 12.0509, 12.0472, 3.90597, 5.34338, 6.33072,     # 1-10 
        2.76777, 7.90099, 5.58323, 4.26372, 3.3005, 5.69179, 2.3698, 3.68167, 5.2807, 4.61212,     # 11-20 
        5.87809, 4.46207, 4.59278, 4.67584, 1.75212, 7.00575, 2.05428, 2.00415, 2.02048, 1.98413,     # 21-30 
        1.71725, 3.18743, 1.74231, 4.40997, 2.01626, 1.8622, 1.7544, 1.60332, 2.23338, 1.70932,     # 31-40 
        1.67223, 1.64655, 1.76198, 6.33416, 7.92665, 1.67835, 1.67408, 1.55895, 9.3642, 1.68776,     # 41-50 
        2.02167, 1.65401, 2.20616, 1.76498, 1.63064, 7.13771, 3.17033, 1.65236, 2.66943, 1.62703,     # 51-60 
        2.72469, 2.73686, 10.86, 2.76759, 2.69728, 1.62436, 2.76662, 1.48514, 1.57342, 1.61518,     # 61-70 
        3.18455, 2.73467, 2.72521, 2.786, 2.35611, 2.31574, 2.5787, 2.46877, 2.89052, 2.6478,     # 71-80 
        1.50419, 2.73998, 2.79809, 2.66207, 2.73089, 1.34835, 2.59656, 2.7006, 1.41867, 4.26255,     # 81-90 
        2.47985, 2.47126, 1.72573, 3.44856, 1.36451, 2.8715, 2.35731, 1.28196, 4.1224, 1.32633,     # 91-100 
    ], dtype=dtype).to(device)
    
    x = cm/(h_Planck*c_light)
    f_factor = torch.tensor(0.5*x*x, device=device)
    
    def compute_w_values(x_b: torch.Tensor, n: torch.Tensor, numlim: float = 0.02):
        w_small = n * x_b * (1.0 - 0.5 * (n - 1.0) * x_b * (1.0 - (n - 2.0) * x_b / 3.0))
        w_large = 1.0 - torch.exp(-n * torch.log1p(x_b))
        return torch.where(x_b < numlim, w_small, w_large)
    
    def compute_x_value(y: torch.Tensor, n: torch.Tensor, numlim: float = 0.02):
        n_inv = 1.0 / n
        x_small = y * n_inv * (1.0 + 0.5 * (n_inv + 1.0) * y * (1.0 - (n_inv + 2.0) * y / 3.0))
        x_large = torch.exp(-n_inv * torch.log(1.0 - y)) - 1.0
        return torch.where(y < numlim, x_small, x_large)

    def theta_generator(energy: torch.Tensor,
                        Z: torch.Tensor,
                        PP0: torch.Tensor=PP0,
                        PP1: torch.Tensor=PP1,
                        PP2: torch.Tensor=PP2,
                        PP3: torch.Tensor=PP3,
                        PP4: torch.Tensor=PP4,
                        PP5: torch.Tensor=PP5,
                        PP6: torch.Tensor=PP6,
                        PP7: torch.Tensor=PP7,
                        PP8: torch.Tensor=PP8,
                        f_factor: torch.Tensor = f_factor,
                        generator: torch.Generator = rng) -> torch.Tensor:
        device = energy.device
        original_shape = energy.shape

        batch_size = energy.numel()
        energy_flat = energy.reshape(-1)
        Z_flat = Z.reshape(-1).long()
        
        xx = f_factor * energy_flat.square()
        
        n0 = PP6[Z_flat] - 1.0
        n1 = PP7[Z_flat] - 1.0
        n2 = PP8[Z_flat] - 1.0
        b0 = PP3[Z_flat]
        b1 = PP4[Z_flat]
        b2 = PP5[Z_flat]
        y0 = PP0[Z_flat]
        y1 = PP1[Z_flat]
        y2 = PP2[Z_flat]
        
        x_b0 = 2.0 * xx * b0
        x_b1 = 2.0 * xx * b1
        x_b2 = 2.0 * xx * b2
        
        w0 = compute_w_values(x_b0, n0)
        w1 = compute_w_values(x_b1, n1)
        w2 = compute_w_values(x_b2, n2)
        
        x0 = w0 * y0 / (b0 * n0)
        x1 = w1 * y1 / (b1 * n1)
        x2 = w2 * y2 / (b2 * n2)
        
        cost = torch.zeros(batch_size, device=device, dtype=PP3.dtype)

        need_sampling = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_max_iter = 10
        iters = 0
        x_sum = x0 + x1 + x2

        while need_sampling.any() and iters < num_max_iter:
            iters += 1
            indices = torch.where(need_sampling)[0]
            num_need_sampling = indices.shape[0]

            r1 = torch.rand(num_need_sampling, device=device, generator=generator)
            r2 = torch.rand(num_need_sampling, device=device, generator=generator)
            r3 = torch.rand(num_need_sampling, device=device, generator=generator)
            
            xx_need = xx[indices]
            x_sum_need = x_sum[indices]

            w = w0[indices].clone()
            n = n0[indices].clone()
            b = b0[indices].clone()
            
            x_rand = r1 * x_sum_need
            
            mask1 = x_rand > x0[indices]
            mask2 = (x_rand <= (x0[indices] + x1[indices])) & mask1
            mask3 = mask1 & ~mask2
            
            w[mask2] = w1[indices][mask2]
            n[mask2] = n1[indices][mask2]
            b[mask2] = b1[indices][mask2]
            
            w[mask3] = w2[indices][mask3]
            n[mask3] = n2[indices][mask3]
            b[mask3] = b2[indices][mask3]
            
            y = w * r2

            x_1 = compute_x_value(y, n)
            
            cost_sampled = 1.0 - x_1 / (b * xx_need)
            
            accepted = (2 * r3 < 1.0 + cost_sampled.square()) | (cost_sampled > -1.0)
            
            accepted_indices = indices[accepted]
            cost[accepted_indices] = cost_sampled[accepted]
            
            need_sampling[accepted_indices] = False
        
        cost = cost.reshape(original_shape)
        theta = torch.acos(cost)
        
        return theta
    return theta_generator

def initialize_cpu_compton(rng):
    dtype = torch.get_default_dtype()
    if dtype == torch.float16:
        dtype = torch.float32  # fit param is out of range of float16, not recommended
    scat_func_fit_param = torch.tensor([
        [0,              0.,             0.,              0.,              0.,              0.,               0.,              0.,               0.,              0.,               0.,              0.,               0.,              0.,               0.,              0.],
        [1, 6.000000000e+00, 7.087999300e+00, 1.499680000e+08, -1.435559123e+01, 2.000000043e+00, -3.925518125e+02, 2.434944521e+02, -5.784393623e+01, 6.160181204e+00, -2.461326602e-01, -1.649463594e+03, 8.121933215e+02, -1.498313316e+02, 1.227279742e+01, -3.765996345e-01],
        [2, 6.000000000e+00, 7.199000403e+00, 2.500350000e+08, -1.430103027e+01, 2.000000041e+00, 3.574019365e+02, -1.978574937e+02, 3.971327838e+01, -3.443224867e+00, 1.091825227e-01, -4.009960832e+02, 1.575831469e+02, -2.174763446e+01, 1.185163045e+00, -1.814503741e-02],
        [3, 6.000000000e+00, 7.301000136e+00, 3.999450000e+08, -1.357675458e+01, 2.000000074e+00, 7.051635443e+02, -4.223841786e+02, 9.318729225e+01, -9.002642767e+00, 3.220625771e-01, 1.524679907e+03, -7.851479582e+02, 1.509941052e+02, -1.285477984e+01, 4.089348830e-01],
        [4, 6.000000000e+00, 7.349500202e+00, 5.000350000e+08, -1.375202671e+01, 1.999999994e+00, -1.832909604e+02, 1.193997722e+02, -3.034328318e+01, 3.471545044e+00, -1.484222463e-01, 1.397476657e+03, -7.026416933e+02, 1.320720559e+02, -1.099824430e+01, 3.424610532e-01],
        [5, 6.000000000e+00, 7.388999972e+00, 5.997910000e+08, -1.380548571e+01, 2.000000004e+00, -2.334197545e+02, 1.467013466e+02, -3.574851109e+01, 3.925047955e+00, -1.616186492e-01, 6.784713308e+02, -3.419562074e+02, 6.433945831e+01, -5.354244209e+00, 1.663784966e-01],
        [6, 6.000000000e+00, 7.422500001e+00, 6.998420000e+08, -1.388639003e+01, 1.999999863e+00, -2.460254935e+02, 1.516613633e+02, -3.622024219e+01, 3.900099543e+00, -1.576557530e-01, -1.610185428e+02, 7.010907070e+01, -1.142375397e+01, 8.303365180e-01, -2.273786010e-02],
        [7, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.388605429e+01, 1.999999612e+00, -3.054540719e+02, 1.877740247e+02, -4.440273010e+01, 4.718886370e+00, -1.881615004e-01, -2.263864349e+02, 1.017885461e+02, -1.716982752e+01, 1.292954622e+00, -3.668301946e-02],
        [8, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.395860675e+01, 1.999999906e+00, -3.877174895e+02, 2.345831969e+02, -5.431822300e+01, 5.643262324e+00, -2.200840540e-01, -7.949384302e+02, 3.757293602e+02, -6.661741851e+01, 5.256265086e+00, -1.556986777e-01],
        [9, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.400000063e+01, 2.000000106e+00, -2.939854827e+02, 1.784214589e+02, -4.168473845e+01, 4.377669850e+00, -1.724300716e-01, -1.169326170e+03, 5.545642014e+02, -9.863024948e+01, 7.801721240e+00, -2.315522357e-01],
        [10, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.404575854e+01, 2.000000178e+00, -2.615701853e+02, 1.582596311e+02, -3.698114811e+01, 3.889093901e+00, -1.533613504e-01, -1.275287356e+03, 6.022076554e+02, -1.066410301e+02, 8.398773148e+00, -2.481899800e-01],
        [11, 6.000000000e+00, 7.500000000e+00, 1.000000000e+09, -1.344369665e+01, 1.999999860e+00, 1.112662501e+03, -6.807056448e+02, 1.545837472e+02, -1.548462180e+01, 5.785425068e-01, -1.007702307e+03, 4.699937040e+02, -8.220352105e+01, 6.396099420e+00, -1.867816054e-01],
        [12, 6.000000000e+00, 7.500000000e+00, 1.000000000e+09, -1.339794047e+01, 2.000000080e+00, 9.895649717e+02, -5.983228286e+02, 1.340681576e+02, -1.323046651e+01, 4.863434994e-01, -5.790532602e+02, 2.626052403e+02, -4.463548055e+01, 3.376239891e+00, -9.588786915e-02],
        [13, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.340893585e+01, 2.000000078e+00, 7.335256091e+02, -4.405291562e+02, 9.770954287e+01, -9.519317788e+00, 3.448067237e-01, -5.328832253e+02, 2.398514938e+02, -4.044557740e+01, 3.034597500e+00, -8.547410419e-02],
        [14, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.345593195e+01, 2.000000000e+00, 3.978691889e+02, -2.370975001e+02, 5.158692183e+01, -4.884868277e+00, 1.707270518e-01, -2.340256277e+02, 9.813362251e+01, -1.527892110e+01, 1.051070768e+00, -2.692716945e-02],
        [15, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.349485049e+01, 2.000000083e+00, 2.569833671e+02, -1.513623448e+02, 3.210087153e+01, -2.925756803e+00, 9.724379436e-02, -1.345727293e+01, -6.291081167e+00, 3.235960888e+00, -4.059236666e-01, 1.601245178e-02],
        [16, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.353760159e+01, 1.999999937e+00, 1.015293074e+02, -5.721639224e+01, 1.078607152e+01, -7.890593144e-01, 1.726056327e-02, 1.854818165e+02, -1.000803879e+02, 1.979815884e+01, -1.704221744e+00, 5.413372375e-02],
        [17, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.358502705e+01, 2.000000066e+00, -4.294163461e+01, 2.862162412e+01, -8.285972104e+00, 1.087745268e+00, -5.172153610e-02, 1.676674074e+02, -8.976414784e+01, 1.763329621e+01, -1.507161653e+00, 4.753277254e-02],
        [18, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.361978902e+01, 2.000000042e+00, -3.573422746e+01, 2.403066369e+01, -7.173617800e+00, 9.657608431e-01, -4.662317662e-02, 1.811925229e+02, -9.574636323e+01, 1.861940167e+01, -1.578810247e+00, 4.946799877e-02],
        [19, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.320760816e+01, 1.999999979e+00, 1.263152069e+02, -8.738932892e+01, 2.109042182e+01, -2.166733566e+00, 8.146018979e-02, 9.183312428e+01, -5.232836676e+01, 1.072450810e+01, -9.419512971e-01, 3.023884410e-02],
        [20, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.314266674e+01, 1.999999876e+00, 6.620218058e+02, -4.057504297e+02, 9.180787767e+01, -9.124184449e+00, 3.372518137e-01, 7.034138711e+01, -4.198325416e+01, 8.861351614e+00, -7.930506530e-01, 2.578454342e-02],
        [21, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.317392498e+01, 1.999999966e+00, 6.766093786e+02, -4.129087029e+02, 9.305090790e+01, -9.212128925e+00, 3.392408033e-01, 1.916559096e+01, -1.807294109e+01, 4.677205921e+00, -4.679350245e-01, 1.632115420e-02],
        [22, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.320065945e+01, 1.999999999e+00, 6.969823082e+02, -4.236620289e+02, 9.513714106e+01, -9.388294642e+00, 3.446942719e-01, -6.501317146e+01, 2.138553133e+01, -2.250998891e+00, 7.219326079e-02, 5.467529893e-04],
        [23, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.322914744e+01, 1.999999909e+00, 6.889749928e+02, -4.181421624e+02, 9.373529727e+01, -9.233142268e+00, 3.383772151e-01, -1.382770534e+02, 5.540647456e+01, -8.170017489e+00, 5.295569200e-01, -1.269556386e-02],
        [24, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.333724128e+01, 1.999999854e+00, 4.365566411e+02, -2.672774427e+02, 6.001631369e+01, -5.895458454e+00, 2.149710735e-01, -2.393534124e+02, 1.020845165e+02, -1.624744211e+01, 1.150387566e+00, -3.057723021e-02],
        [25, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.328399669e+01, 2.000000008e+00, 6.461381990e+02, -3.918546518e+02, 8.769548644e+01, -8.618784385e+00, 3.150660827e-01, -2.597409979e+02, 1.113332866e+02, -1.782124571e+01, 1.269519197e+00, -3.396126698e-02],
        [26, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.330103000e+01, 1.999999998e+00, 4.261007401e+02, -2.588846763e+02, 5.764613910e+01, -5.609660122e+00, 2.024165636e-01, -1.982896712e+02, 8.274273985e+01, -1.284074215e+01, 8.845687432e-01, -2.282143299e-02],
        [27, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.332790165e+01, 1.999999922e+00, 4.006816638e+02, -2.439311564e+02, 5.435031497e+01, -5.287693457e+00, 1.906696163e-01, -2.205075564e+02, 9.262919772e+01, -1.448909443e+01, 1.006686819e+00, -2.621294059e-02],
        [28, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.334678710e+01, 1.999999939e+00, 3.967750019e+02, -2.411866801e+02, 5.364872608e+01, -5.210295834e+00, 1.875525119e-01, -2.516823030e+02, 1.065117131e+02, -1.680533335e+01, 1.178363534e+00, -3.098194406e-02],
        [29, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.344369664e+01, 1.999999853e+00, 2.437671888e+02, -1.499592208e+02, 3.332221026e+01, -3.206587185e+00, 1.138639692e-01, -2.874130637e+02, 1.223381969e+02, -1.943178054e+01, 1.371979484e+00, -3.633119448e-02],
        [30, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.338721562e+01, 1.999999911e+00, 3.914867984e+02, -2.378147085e+02, 5.284517777e+01, -5.126420186e+00, 1.843322562e-01, -3.235063319e+02, 1.384252948e+02, -2.211844479e+01, 1.571300198e+00, -4.187323186e-02],
        [31, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.335654643e+01, 1.999999847e+00, 4.325820127e+02, -2.614587597e+02, 5.793273998e+01, -5.611190206e+00, 2.015836827e-01, -3.359152840e+02, 1.437507638e+02, -2.297457475e+01, 1.632470701e+00, -4.351215346e-02],
        [32, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.337675047e+01, 1.999999960e+00, 4.388195965e+02, -2.642662297e+02, 5.834159168e+01, -5.629419790e+00, 2.014339673e-01, -3.430730654e+02, 1.467102631e+02, -2.343160019e+01, 1.663765504e+00, -4.431369286e-02],
        [33, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.339794046e+01, 2.000000074e+00, 3.931399547e+02, -2.363700718e+02, 5.197696913e+01, -4.987097655e+00, 1.772567576e-01, -3.501570134e+02, 1.497141578e+02, -2.390888062e+01, 1.697503580e+00, -4.520887478e-02],
        [34, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.342021680e+01, 2.000000064e+00, 3.772588127e+02, -2.256347960e+02, 4.929790851e+01, -4.694628847e+00, 1.654667382e-01, -3.481053019e+02, 1.486490112e+02, -2.370745096e+01, 1.680991482e+00, -4.471064364e-02],
        [35, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.344369666e+01, 1.999999864e+00, 3.344685842e+02, -1.994816236e+02, 4.332267376e+01, -4.090542180e+00, 1.426839031e-01, -3.227660675e+02, 1.370301996e+02, -2.171543883e+01, 1.529681552e+00, -4.041331983e-02],
        [36, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.345593194e+01, 1.999999999e+00, 3.004054446e+02, -1.781334135e+02, 3.834850324e+01, -3.580074471e+00, 1.232168921e-01, -2.980827664e+02, 1.257508661e+02, -1.978792154e+01, 1.383723149e+00, -3.628014907e-02],
        [37, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.310790583e+01, 2.000000075e+00, -3.687188343e+01, 1.054409719e+01, -8.516586814e-01, 9.339751003e-03, 8.809383936e-04, -2.699384784e+02, 1.129635316e+02, -1.761447452e+01, 1.219971043e+00, -3.166503704e-02],
        [38, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.304095795e+01, 1.999999892e+00, 1.969969064e+02, -1.286503864e+02, 3.008431767e+01, -3.031946980e+00, 1.124456346e-01, -2.331258613e+02, 9.627987243e+01, -1.478515961e+01, 1.007215642e+00, -2.567873120e-02],
        [39, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.306048023e+01, 1.999999916e+00, 2.891710763e+02, -1.819536752e+02, 4.158265841e+01, -4.128940218e+00, 1.515168697e-01, -1.997404800e+02, 8.119476676e+01, -1.223426670e+01, 8.159269666e-01, -2.031079820e-02],
        [40, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.308092198e+01, 2.000000013e+00, 3.393782172e+02, -2.103908454e+02, 4.758278737e+01, -4.688308235e+00, 1.709723418e-01, -1.549247582e+02, 6.091403935e+01, -8.799307373e+00, 5.578963961e-01, -1.305663921e-02],
        [41, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.316749062e+01, 1.999999920e+00, 2.748604341e+02, -1.706429616e+02, 3.843757441e+01, -3.759045290e+00, 1.358263430e-01, -1.163607425e+02, 4.350905533e+01, -5.859305970e+00, 3.376426246e-01, -6.881281652e-03],
        [42, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.318708720e+01, 2.000000093e+00, 3.203285955e+02, -1.966282865e+02, 4.398204769e+01, -4.283031482e+00, 1.543480828e-01, -9.364181222e+01, 3.329814493e+01, -4.141689265e+00, 2.095170962e-01, -3.304665813e-03],
        [43, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.314266674e+01, 1.999999876e+00, 4.184977165e+02, -2.552902161e+02, 5.707764818e+01, -5.576436872e+00, 2.020184726e-01, -8.395646154e+01, 2.898228589e+01, -3.422356654e+00, 1.564059753e-01, -1.838508896e-03],
        [44, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.322914744e+01, 1.999999909e+00, 3.243555305e+02, -1.978255470e+02, 4.397580841e+01, -4.256142657e+00, 1.524431452e-01, -5.506292375e+01, 1.599310639e+01, -1.237152904e+00, -6.611574411e-03, 2.712232383e-03],
        [45, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.325181249e+01, 2.000000089e+00, 3.037823599e+02, -1.856628295e+02, 4.128167884e+01, -3.991656133e+00, 1.427469878e-01, -5.014186072e+01, 1.386962969e+01, -8.950806420e-01, -3.095321225e-02, 3.357984426e-03],
        [46, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.340893584e+01, 2.000000073e+00, 3.529797051e+02, -2.101512262e+02, 4.563946029e+01, -4.315279704e+00, 1.509248358e-01, -4.815922691e+01, 1.301508788e+01, -7.580854951e-01, -4.059091985e-02, 3.608993811e-03],
        [47, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.328399669e+01, 2.000000008e+00, 3.074953924e+02, -1.872462583e+02, 4.149827252e+01, -4.000811852e+00, 1.426973118e-01, -4.897188379e+01, 1.335300002e+01, -8.110051997e-01, -3.684788190e-02, 3.508156457e-03],
        [48, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.322914743e+01, 1.999999904e+00, 4.059717166e+02, -2.462737702e+02, 5.472040126e+01, -5.311320062e+00, 1.911670149e-01, -5.901534554e+01, 1.791385249e+01, -1.587065943e+00, 2.182673278e-02, 1.845559896e-03],
        [49, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.320760815e+01, 1.999999973e+00, 4.369774251e+02, -2.639721687e+02, 5.849617557e+01, -5.667842049e+00, 2.037342202e-01, -7.399698219e+01, 2.469785523e+01, -2.737881327e+00, 1.085351830e-01, -6.022720695e-04],
        [50, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.322184869e+01, 1.999999993e+00, 4.289361021e+02, -2.585593024e+02, 5.714058683e+01, -5.518600115e+00, 1.976499817e-01, -9.269047286e+01, 3.314422349e+01, -4.167341855e+00, 2.159629039e-01, -3.626802503e-03],
        [51, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.323657166e+01, 1.999999946e+00, 3.866985836e+02, -2.328379698e+02, 5.128884878e+01, -4.929614910e+00, 1.755331333e-01, -1.067869310e+02, 3.950715983e+01, -5.243321447e+00, 2.967791238e-01, -5.901223876e-03],
        [52, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.325181248e+01, 2.000000083e+00, 3.947511198e+02, -2.363799049e+02, 5.179393756e+01, -4.951603918e+00, 1.753404387e-01, -1.069681982e+02, 3.995521754e+01, -5.382071424e+00, 3.120248901e-01, -6.467957474e-03],
        [53, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.326760745e+01, 2.000000205e+00, 3.694394448e+02, -2.204699428e+02, 4.806381052e+01, -4.565474883e+00, 1.604614344e-01, -1.180749905e+02, 4.460080701e+01, -6.105217447e+00, 3.616537171e-01, -7.733059623e-03],
        [54, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.328399667e+01, 2.000000001e+00, 3.423943987e+02, -2.041330669e+02, 4.437639784e+01, -4.197363553e+00, 1.467594367e-01, -1.288973984e+02, 4.985324046e+01, -7.056041375e+00, 4.378018318e-01, -1.000965926e-02],
        [55, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.297881025e+01, 1.999999927e+00, -7.663422017e+01, 3.462700567e+01, -6.273553579e+00, 5.487612834e-01, -1.912897528e-02, -1.318428276e+02, 5.081036112e+01, -7.154907590e+00, 4.405355674e-01, -9.955685075e-03],
        [56, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.290657751e+01, 1.999999869e+00, 1.084179205e+02, -7.602229206e+01, 1.843754298e+01, -1.892451591e+00, 7.085434176e-02, -1.346311376e+02, 5.207427468e+01, -7.369834199e+00, 4.568138610e-01, -1.041859875e-02],
        [57, 6.000000000e+00, 7.725500002e+00, 2.824880000e+09, -1.292445241e+01, 1.999999898e+00, 2.995898890e+02, -1.889477671e+02, 4.336642429e+01, -4.330424108e+00, 1.599942758e-01, 5.503972208e+00, -1.227641064e+01, 3.699182312e+00, -3.884476060e-01, 1.375966896e-02],
        [58, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.293554133e+01, 1.999999890e+00, 1.709135500e+02, -1.120124681e+02, 2.615893820e+01, -2.624416758e+00, 9.674223967e-02, -1.375860132e+02, 5.337811974e+01, -7.586786386e+00, 4.730023198e-01, -1.087482303e-02],
        [59, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.293554133e+01, 1.999999890e+00, 1.214691988e+02, -8.336119630e+01, 1.996468944e+01, -2.032283439e+00, 7.562254632e-02, -1.631005912e+02, 6.472051894e+01, -9.476098737e+00, 6.127875286e-01, -1.475060958e-02],
        [60, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.294309494e+01, 1.999999967e+00, 1.302719596e+02, -8.835087414e+01, 2.101971144e+01, -2.131084478e+00, 7.908549730e-02, -1.692901279e+02, 6.742727614e+01, -9.920661139e+00, 6.453186854e-01, -1.564524492e-02],
        [61, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.295078139e+01, 1.999999905e+00, 1.127680235e+02, -7.782238836e+01, 1.865126163e+01, -1.895116816e+00, 7.030502833e-02, -2.059821608e+02, 8.384774285e+01, -1.267344799e+01, 8.502354115e-01, -2.135994609e-02],
        [62, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.295860692e+01, 1.999999936e+00, 1.203145109e+02, -8.212556537e+01, 1.956606386e+01, -1.981212240e+00, 7.333626288e-02, -2.158058793e+02, 8.810144391e+01, -1.336380022e+01, 9.000362964e-01, -2.270715579e-02],
        [63, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.296657573e+01, 1.999999918e+00, 1.212159597e+02, -8.256559477e+01, 1.964122173e+01, -1.986442056e+00, 7.345564343e-02, -2.278531434e+02, 9.336519465e+01, -1.422588608e+01, 9.627883381e-01, -2.441986614e-02],
        [64, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.298296617e+01, 1.999999921e+00, 1.689382403e+02, -1.099987696e+02, 2.551961464e+01, -2.543234152e+00, 9.313568005e-02, -2.282716670e+02, 9.348611199e+01, -1.423588448e+01, 9.628551072e-01, -2.440492772e-02],
        [65, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.299139910e+01, 1.999999880e+00, 1.724155378e+02, -1.120798437e+02, 2.598264738e+01, -2.588807295e+00, 9.481417896e-02, -2.322687147e+02, 9.517466656e+01, -1.450332749e+01, 9.817069914e-01, -2.490386807e-02],
        [66, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.298716240e+01, 1.999999941e+00, 1.286079419e+02, -8.646296410e+01, 2.039801258e+01, -2.050839207e+00, 7.549033493e-02, -2.420048480e+02, 9.935663043e+01, -1.517653800e+01, 1.029875015e+00, -2.619626869e-02],
        [67, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.299567846e+01, 1.999999971e+00, 1.182799697e+02, -8.043389241e+01, 1.908027783e+01, -1.923209794e+00, 7.087268462e-02, -2.464462609e+02, 1.012059056e+02, -1.546468270e+01, 1.049814070e+00, -2.671320158e-02],
        [68, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.300436459e+01, 1.999999966e+00, 1.150510247e+02, -7.859576077e+01, 1.868688175e+01, -1.885844183e+00, 6.954765052e-02, -2.457555063e+02, 1.007538481e+02, -1.536692833e+01, 1.041070997e+00, -2.643279207e-02],
        [69, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.300877391e+01, 2.000000000e+00, 1.266280406e+02, -8.514491730e+01, 2.007089332e+01, -2.015475088e+00, 7.409191965e-02, -2.492442707e+02, 1.021615320e+02, -1.557878384e+01, 1.055183253e+00, -2.678362279e-02],
        [70, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.301772826e+01, 1.999999912e+00, 1.224253568e+02, -8.281395858e+01, 1.958609738e+01, -1.970785167e+00, 7.255458061e-02, -2.488808342e+02, 1.018569466e+02, -1.550601866e+01, 1.048325396e+00, -2.655661748e-02],
        [71, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.303151733e+01, 2.000000051e+00, 1.862181262e+02, -1.199038630e+02, 2.763107534e+01, -2.742586837e+00, 1.001956495e-01, -2.403102476e+02, 9.796272016e+01, -1.484525920e+01, 9.987147871e-01, -2.516533876e-02],
        [72, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.304575796e+01, 2.000000081e+00, 2.297759959e+02, -1.448485621e+02, 3.295877082e+01, -3.245850428e+00, 1.179456377e-01, -2.282155654e+02, 9.249921555e+01, -1.392266984e+01, 9.297052139e-01, -2.323558576e-02],
        [73, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.306048022e+01, 1.999999910e+00, 2.646909006e+02, -1.647716545e+02, 3.719903613e+01, -3.645113853e+00, 1.319890617e-01, -2.165150972e+02, 8.722660467e+01, -1.303415548e+01, 8.633600348e-01, -2.138300143e-02],
        [74, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.308092196e+01, 2.000000008e+00, 2.251239174e+02, -1.414731209e+02, 3.206048507e+01, -3.142433101e+00, 1.135971917e-01, -2.070173544e+02, 8.296725365e+01, -1.231986936e+01, 8.102887128e-01, -1.990853407e-02],
        [75, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.309151488e+01, 1.999999984e+00, 2.627532736e+02, -1.629008146e+02, 3.661592385e+01, -3.571257833e+00, 1.286871297e-01, -1.945762063e+02, 7.740995255e+01, -1.139129234e+01, 7.415172466e-01, -1.800335280e-02],
        [76, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.310790581e+01, 2.000000068e+00, 2.644549626e+02, -1.637369900e+02, 3.675734857e+01, -3.580665992e+00, 1.288721975e-01, -1.725967865e+02, 6.755389456e+01, -9.737633351e+00, 6.184954292e-01, -1.457897448e-02],
        [77, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.311918599e+01, 1.999999933e+00, 2.677629012e+02, -1.650589135e+02, 3.690999414e+01, -3.582378706e+00, 1.284763849e-01, -1.584140848e+02, 6.122430396e+01, -8.680876005e+00, 5.402879020e-01, -1.241386995e-02],
        [78, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.319382006e+01, 2.000000009e+00, 2.420702029e+02, -1.484461630e+02, 3.292288306e+01, -3.162757529e+00, 1.121487556e-01, -1.319886050e+02, 4.940494114e+01, -6.702740089e+00, 3.934770465e-01, -8.336673895e-03],
        [79, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.320760814e+01, 1.999999969e+00, 2.346714957e+02, -1.439356552e+02, 3.189416251e+01, -3.059071523e+00, 1.082595858e-01, -1.130109430e+02, 4.093029258e+01, -5.286747014e+00, 2.885753389e-01, -5.428939868e-03],
        [80, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.316115147e+01, 2.000000093e+00, 2.747370538e+02, -1.689673404e+02, 3.771696324e+01, -3.655841153e+00, 1.309852214e-01, -9.001823908e+01, 3.066094857e+01, -3.570459523e+00, 1.613797666e-01, -1.901561361e-03],
        [81, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.313667715e+01, 2.000000002e+00, 3.142563781e+02, -1.916613838e+02, 4.259167223e+01, -4.119713271e+00, 1.474792530e-01, -7.642731867e+01, 2.462410146e+01, -2.566977318e+00, 8.741068396e-02, 1.388590928e-04],
        [82, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.314266674e+01, 1.999999876e+00, 3.509258060e+02, -2.125470710e+02, 4.702461797e+01, -4.535380912e+00, 1.620138781e-01, -5.173355302e+01, 1.362015056e+01, -7.321282362e-01, -4.826261322e-02, 3.892879264e-03],
        [83, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.315490164e+01, 1.999999944e+00, 3.399729483e+02, -2.056319770e+02, 4.539614689e+01, -4.366195994e+00, 1.554792165e-01, -4.131443229e+01, 8.986236911e+00, 3.924628986e-02, -1.052060828e-01, 5.466043586e-03],
        [84, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.316749062e+01, 1.999999920e+00, 3.640602841e+02, -2.190164327e+02, 4.815603439e+01, -4.616573783e+00, 1.639147626e-01, -3.256862965e+01, 5.115606198e+00, 6.800853161e-01, -1.522315744e-01, 6.756786448e-03],
        [85, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.318045630e+01, 2.000000044e+00, 3.766488275e+02, -2.257321142e+02, 4.947300991e+01, -4.728919006e+00, 1.674240471e-01, -2.300947210e+01, 8.615223509e-01, 1.388425307e+00, -2.045157608e-01, 8.200511055e-03],
        [86, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.319382005e+01, 2.000000006e+00, 3.443622947e+02, -2.064342780e+02, 4.516044966e+01, -4.302253084e+00, 1.516667044e-01, -5.399039282e+00, -7.002814559e+00, 2.702516748e+00, -3.018766003e-01, 1.089953798e-02],
        [87, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.291364147e+01, 2.000000217e+00, -3.706791591e+01, 1.118013187e+01, -1.057728859e+00, 3.312859839e-02, -3.138341244e-06, -3.451314336e+00, -7.779254134e+00, 2.816269849e+00, -3.090776388e-01, 1.106424389e-02],
        [88, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.284163724e+01, 1.999999954e+00, 6.125934670e+01, -4.855548659e+01, 1.248551381e+01, -1.323304763e+00, 5.060744172e-02, -6.021643455e+00, -6.580234329e+00, 2.607440108e+00, -2.929625239e-01, 1.059951856e-02],
        [89, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.285387248e+01, 2.000000090e+00, 1.350863292e+02, -9.126618691e+01, 2.169932948e+01, -2.201947573e+00, 8.186860720e-02, 1.937135880e+01, -1.787129621e+01, 4.485878662e+00, -4.315325969e-01, 1.442445798e-02],
        [90, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.286966604e+01, 1.999999970e+00, 1.784388998e+02, -1.161623817e+02, 2.702376618e+01, -2.704797298e+00, 9.957279361e-02, 2.216057166e+01, -1.904990091e+01, 4.671627339e+00, -4.444534802e-01, 1.475921763e-02],
        [91, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.287289489e+01, 1.999999951e+00, 1.368355213e+02, -9.179790820e+01, 2.169910915e+01, -2.190249857e+00, 8.102241740e-02, 4.516580666e+00, -1.118102949e+01, 3.357662550e+00, -3.470694353e-01, 1.205639951e-02],
        [92, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.287942629e+01, 2.000000032e+00, 1.427130850e+02, -9.499714618e+01, 2.234475916e+01, -2.247599931e+00, 8.291713193e-02, 1.341991149e+01, -1.518503354e+01, 4.030838171e+00, -3.972060658e-01, 1.345248084e-02],
        [93, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.288605524e+01, 1.999999761e+00, 2.341801100e+01, -2.506119713e+01, 7.023029272e+00, -7.610742531e-01, 2.903245750e-02, -3.575331738e+01, 7.276302226e+00, 1.906771859e-01, -1.059475755e-01, 5.184029625e-03],
        [94, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.288272835e+01, 1.999999941e+00, 1.287618322e+02, -8.721780968e+01, 2.073255323e+01, -2.100572716e+00, 7.794295578e-02, -2.307262580e+01, 1.113132278e+00, 1.305250601e+00, -1.948949139e-01, 7.829116438e-03],
        [95, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.288940956e+01, 1.999999880e+00, 1.334821220e+02, -8.985337775e+01, 2.127928526e+01, -2.150628571e+00, 7.965294640e-02, -3.518662723e+01, 6.514543434e+00, 4.030862442e-01, -1.279850170e-01, 5.970168353e-03],
        [96, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.290553004e+01, 2.000000198e+00, 4.545581472e+01, -3.771304300e+01, 9.729129321e+00, -1.017037014e+00, 3.807733199e-02, -4.973805034e+01, 1.342335334e+01, -8.221139917e-01, -3.176841835e-02, 3.146810827e-03],
        [97, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.291150963e+01, 2.000000019e+00, 4.689042092e+01, -3.843347264e+01, 9.859294531e+00, -1.027014690e+00, 3.834833665e-02, -4.657434145e+01, 1.204637835e+01, -5.982449163e-01, -4.786919243e-02, 3.579251285e-03],
        [98, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.290833198e+01, 1.999999824e+00, 1.337584189e+01, -1.907284620e+01, 5.691614909e+00, -6.307838734e-01, 2.430868142e-02, -5.573362773e+01, 1.615667599e+01, -1.288960621e+00, 3.655033732e-03, 2.140047522e-03],
        [99, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.291435263e+01, 1.999999988e+00, 1.376201293e+01, -1.919251815e+01, 5.693799461e+00, -6.287500644e-01, 2.416045199e-02, -4.914211254e+01, 1.314247998e+01, -7.739336035e-01, -3.530513333e-02, 3.241293077e-03],
        [100, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.292045700e+01, 2.000000004e+00, 1.277081775e+01, -1.854047224e+01, 5.534680382e+00, -6.118054153e-01, 2.349768815e-02, -5.074293980e+01, 1.383260974e+01, -8.858904786e-01, -2.718885953e-02, 3.019620454e-03]
    ], dtype=dtype)

    ln10 = torch.log(torch.tensor(10.))
    electron_mass_c2 = 0.510998910*MeV
    wl = h_Planck * c_light
    cm_hep = cm

    def compute_scattering_function(x: torch.Tensor,
                                    Z: torch.Tensor,
                                    scat_func_fit_param: torch.Tensor = scat_func_fit_param,
                                    ln10: float = ln10) -> torch.Tensor:
        result = Z.clone()
        Z_long = Z.long()
        
        x_threshold = torch.gather(scat_func_fit_param[:, 3], 0, Z_long)
        valid_mask = x <= x_threshold
        
        if valid_mask.any():
            lgq = torch.zeros_like(x)
            lgq[valid_mask] = torch.log(x[valid_mask]) / ln10
            
            threshold1 = torch.gather(scat_func_fit_param[:, 1], 0, Z_long)
            threshold2 = torch.gather(scat_func_fit_param[:, 2], 0, Z_long)
            
            mask1 = valid_mask & (lgq < threshold1)
            mask2 = valid_mask & (lgq >= threshold1) & (lgq < threshold2)
            mask3 = valid_mask & (lgq >= threshold2)
            
            if mask1.any():
                Z1 = Z_long[mask1]
                lgq1 = lgq[mask1]
                
                params1 = torch.stack([
                    torch.gather(scat_func_fit_param[:, 4], 0, Z1),
                    torch.gather(scat_func_fit_param[:, 5], 0, Z1)
                ], dim=1)
                
                value = params1[:, 0] + lgq1 * params1[:, 1]
                result[mask1] = torch.exp(value * ln10)
            
            if mask2.any():
                Z2 = Z_long[mask2]
                lgq2 = lgq[mask2]
                
                params2 = torch.stack([
                    torch.gather(scat_func_fit_param[:, 6], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 7], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 8], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 9], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 10], 0, Z2)
                ], dim=1)
                
                value = params2[:, 0] + lgq2 * (params2[:, 1] + lgq2 * (params2[:, 2] + 
                                                                    lgq2 * (params2[:, 3] + 
                                                                            lgq2 * params2[:, 4])))
                result[mask2] = torch.exp(value * ln10)
            
            if mask3.any():
                Z3 = Z_long[mask3]
                lgq3 = lgq[mask3]
                
                params3 = torch.stack([
                    torch.gather(scat_func_fit_param[:, 11], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 12], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 13], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 14], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 15], 0, Z3)
                ], dim=1)
                
                value = params3[:, 0] + lgq3 * (params3[:, 1] + lgq3 * (params3[:, 2] + 
                                                                    lgq3 * (params3[:, 3] + 
                                                                            lgq3 * params3[:, 4])))
                result[mask3] = torch.exp(value * ln10)
        
        return result

    @torch.jit.script
    def theta_generator(energy: torch.Tensor,
                        Z: torch.Tensor,
                        generator: torch.Generator = rng,
                        electron_mass_c2: float = electron_mass_c2,
                        wl: float = wl,
                        cm_hep: float = cm_hep) -> torch.Tensor:
        """
        Generate scattering angles for batched inputs
        """
        original_shape = energy.shape
        energy_flat = energy.reshape(-1)
        Z_flat = Z.reshape(-1)
        batch_size = energy_flat.size(0)
        device = energy.device
        
        e0m = energy_flat / electron_mass_c2
        epsilon0_local = 1.0 / (1.0 + 2.0 * e0m)
        epsilon0_sq = epsilon0_local * epsilon0_local
        alpha1 = -torch.log(epsilon0_local)
        alpha2 = 0.5 * (1.0 - epsilon0_sq)
        wl_photon = wl / energy_flat
        
        theta = torch.zeros_like(energy_flat)
        pending = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        max_iter = 10
        
        for _ in range(max_iter):    
            pending_indices = torch.nonzero(pending).squeeze(-1)
            pending_count = pending_indices.size(0)
            
            rand_vals = torch.rand((pending_count, 3), device=device, generator=generator)
            
            p_e0m = e0m[pending_indices]
            p_epsilon0_sq = epsilon0_sq[pending_indices]
            p_alpha1 = alpha1[pending_indices]
            p_alpha2 = alpha2[pending_indices]
            p_wl_photon = wl_photon[pending_indices]
            p_Z = Z_flat[pending_indices]
            
            epsilon = torch.zeros(pending_count, device=device)
            alpha_ratio = p_alpha1 / (p_alpha1 + p_alpha2)
            method_mask = alpha_ratio > rand_vals[:, 0]
            
            if method_mask.any():
                epsilon[method_mask] = torch.exp(-p_alpha1[method_mask] * rand_vals[method_mask, 1])
            
            not_method_mask = ~method_mask
            if not_method_mask.any():
                epsilon_sq_temp = p_epsilon0_sq[not_method_mask] + (1.0 - p_epsilon0_sq[not_method_mask]) * rand_vals[not_method_mask, 1]
                epsilon[not_method_mask] = torch.sqrt(epsilon_sq_temp)
            
            epsilon_sq = epsilon * epsilon
            one_cos_t = (1.0 - epsilon) / (epsilon * p_e0m)
            sinT2 = one_cos_t * (2.0 - one_cos_t)
            x = torch.sqrt(one_cos_t / 2.0) * cm_hep / p_wl_photon
            
            scatter_func = compute_scattering_function(x, p_Z)
            
            g_reject = (1.0 - epsilon * sinT2 / (1.0 + epsilon_sq)) * scatter_func
            accepted = g_reject > rand_vals[:, 2] * p_Z
            
            if accepted.any():
                accepted_cos_theta = 1.0 - one_cos_t[accepted]
                accepted_indices = pending_indices[accepted]
                
                theta[accepted_indices] = torch.acos(accepted_cos_theta)
                pending[accepted_indices] = False
        
        if pending.any():
            remaining_indices = torch.nonzero(pending).squeeze(-1)
            remaining_count = remaining_indices.size(0)
            
            r_e0m = e0m[remaining_indices]
            r_alpha1 = alpha1[remaining_indices]
            
            rand = torch.rand(remaining_count, device=device, generator=generator)
            
            r_epsilon = torch.exp(-r_alpha1 * rand)
            r_one_cos_t = (1.0 - r_epsilon) / (r_epsilon * r_e0m)
            r_cos_theta = 1.0 - r_one_cos_t
            
            theta[remaining_indices] = torch.acos(r_cos_theta)
        
        return theta.reshape(original_shape)
        
    return torch.compile(
        torch.jit.script(
            theta_generator, 
        ),
        dynamic=True,
        fullgraph=True,
    )

def initialize_cuda_compton(rng, device="cuda"):
    dtype = torch.get_default_dtype()
    if dtype == torch.float16:
        dtype = torch.float32  # fit param is out of range of float16, not recommended.
    print(dtype)
    scat_func_fit_param = torch.tensor([
        [0,              0.,             0.,              0.,              0.,              0.,               0.,              0.,               0.,              0.,               0.,              0.,               0.,              0.,               0.,              0.],
        [1, 6.000000000e+00, 7.087999300e+00, 1.499680000e+08, -1.435559123e+01, 2.000000043e+00, -3.925518125e+02, 2.434944521e+02, -5.784393623e+01, 6.160181204e+00, -2.461326602e-01, -1.649463594e+03, 8.121933215e+02, -1.498313316e+02, 1.227279742e+01, -3.765996345e-01],
        [2, 6.000000000e+00, 7.199000403e+00, 2.500350000e+08, -1.430103027e+01, 2.000000041e+00, 3.574019365e+02, -1.978574937e+02, 3.971327838e+01, -3.443224867e+00, 1.091825227e-01, -4.009960832e+02, 1.575831469e+02, -2.174763446e+01, 1.185163045e+00, -1.814503741e-02],
        [3, 6.000000000e+00, 7.301000136e+00, 3.999450000e+08, -1.357675458e+01, 2.000000074e+00, 7.051635443e+02, -4.223841786e+02, 9.318729225e+01, -9.002642767e+00, 3.220625771e-01, 1.524679907e+03, -7.851479582e+02, 1.509941052e+02, -1.285477984e+01, 4.089348830e-01],
        [4, 6.000000000e+00, 7.349500202e+00, 5.000350000e+08, -1.375202671e+01, 1.999999994e+00, -1.832909604e+02, 1.193997722e+02, -3.034328318e+01, 3.471545044e+00, -1.484222463e-01, 1.397476657e+03, -7.026416933e+02, 1.320720559e+02, -1.099824430e+01, 3.424610532e-01],
        [5, 6.000000000e+00, 7.388999972e+00, 5.997910000e+08, -1.380548571e+01, 2.000000004e+00, -2.334197545e+02, 1.467013466e+02, -3.574851109e+01, 3.925047955e+00, -1.616186492e-01, 6.784713308e+02, -3.419562074e+02, 6.433945831e+01, -5.354244209e+00, 1.663784966e-01],
        [6, 6.000000000e+00, 7.422500001e+00, 6.998420000e+08, -1.388639003e+01, 1.999999863e+00, -2.460254935e+02, 1.516613633e+02, -3.622024219e+01, 3.900099543e+00, -1.576557530e-01, -1.610185428e+02, 7.010907070e+01, -1.142375397e+01, 8.303365180e-01, -2.273786010e-02],
        [7, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.388605429e+01, 1.999999612e+00, -3.054540719e+02, 1.877740247e+02, -4.440273010e+01, 4.718886370e+00, -1.881615004e-01, -2.263864349e+02, 1.017885461e+02, -1.716982752e+01, 1.292954622e+00, -3.668301946e-02],
        [8, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.395860675e+01, 1.999999906e+00, -3.877174895e+02, 2.345831969e+02, -5.431822300e+01, 5.643262324e+00, -2.200840540e-01, -7.949384302e+02, 3.757293602e+02, -6.661741851e+01, 5.256265086e+00, -1.556986777e-01],
        [9, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.400000063e+01, 2.000000106e+00, -2.939854827e+02, 1.784214589e+02, -4.168473845e+01, 4.377669850e+00, -1.724300716e-01, -1.169326170e+03, 5.545642014e+02, -9.863024948e+01, 7.801721240e+00, -2.315522357e-01],
        [10, 6.000000000e+00, 7.451499931e+00, 7.998340000e+08, -1.404575854e+01, 2.000000178e+00, -2.615701853e+02, 1.582596311e+02, -3.698114811e+01, 3.889093901e+00, -1.533613504e-01, -1.275287356e+03, 6.022076554e+02, -1.066410301e+02, 8.398773148e+00, -2.481899800e-01],
        [11, 6.000000000e+00, 7.500000000e+00, 1.000000000e+09, -1.344369665e+01, 1.999999860e+00, 1.112662501e+03, -6.807056448e+02, 1.545837472e+02, -1.548462180e+01, 5.785425068e-01, -1.007702307e+03, 4.699937040e+02, -8.220352105e+01, 6.396099420e+00, -1.867816054e-01],
        [12, 6.000000000e+00, 7.500000000e+00, 1.000000000e+09, -1.339794047e+01, 2.000000080e+00, 9.895649717e+02, -5.983228286e+02, 1.340681576e+02, -1.323046651e+01, 4.863434994e-01, -5.790532602e+02, 2.626052403e+02, -4.463548055e+01, 3.376239891e+00, -9.588786915e-02],
        [13, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.340893585e+01, 2.000000078e+00, 7.335256091e+02, -4.405291562e+02, 9.770954287e+01, -9.519317788e+00, 3.448067237e-01, -5.328832253e+02, 2.398514938e+02, -4.044557740e+01, 3.034597500e+00, -8.547410419e-02],
        [14, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.345593195e+01, 2.000000000e+00, 3.978691889e+02, -2.370975001e+02, 5.158692183e+01, -4.884868277e+00, 1.707270518e-01, -2.340256277e+02, 9.813362251e+01, -1.527892110e+01, 1.051070768e+00, -2.692716945e-02],
        [15, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.349485049e+01, 2.000000083e+00, 2.569833671e+02, -1.513623448e+02, 3.210087153e+01, -2.925756803e+00, 9.724379436e-02, -1.345727293e+01, -6.291081167e+00, 3.235960888e+00, -4.059236666e-01, 1.601245178e-02],
        [16, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.353760159e+01, 1.999999937e+00, 1.015293074e+02, -5.721639224e+01, 1.078607152e+01, -7.890593144e-01, 1.726056327e-02, 1.854818165e+02, -1.000803879e+02, 1.979815884e+01, -1.704221744e+00, 5.413372375e-02],
        [17, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.358502705e+01, 2.000000066e+00, -4.294163461e+01, 2.862162412e+01, -8.285972104e+00, 1.087745268e+00, -5.172153610e-02, 1.676674074e+02, -8.976414784e+01, 1.763329621e+01, -1.507161653e+00, 4.753277254e-02],
        [18, 6.000000000e+00, 7.587999300e+00, 1.499680000e+09, -1.361978902e+01, 2.000000042e+00, -3.573422746e+01, 2.403066369e+01, -7.173617800e+00, 9.657608431e-01, -4.662317662e-02, 1.811925229e+02, -9.574636323e+01, 1.861940167e+01, -1.578810247e+00, 4.946799877e-02],
        [19, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.320760816e+01, 1.999999979e+00, 1.263152069e+02, -8.738932892e+01, 2.109042182e+01, -2.166733566e+00, 8.146018979e-02, 9.183312428e+01, -5.232836676e+01, 1.072450810e+01, -9.419512971e-01, 3.023884410e-02],
        [20, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.314266674e+01, 1.999999876e+00, 6.620218058e+02, -4.057504297e+02, 9.180787767e+01, -9.124184449e+00, 3.372518137e-01, 7.034138711e+01, -4.198325416e+01, 8.861351614e+00, -7.930506530e-01, 2.578454342e-02],
        [21, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.317392498e+01, 1.999999966e+00, 6.766093786e+02, -4.129087029e+02, 9.305090790e+01, -9.212128925e+00, 3.392408033e-01, 1.916559096e+01, -1.807294109e+01, 4.677205921e+00, -4.679350245e-01, 1.632115420e-02],
        [22, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.320065945e+01, 1.999999999e+00, 6.969823082e+02, -4.236620289e+02, 9.513714106e+01, -9.388294642e+00, 3.446942719e-01, -6.501317146e+01, 2.138553133e+01, -2.250998891e+00, 7.219326079e-02, 5.467529893e-04],
        [23, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.322914744e+01, 1.999999909e+00, 6.889749928e+02, -4.181421624e+02, 9.373529727e+01, -9.233142268e+00, 3.383772151e-01, -1.382770534e+02, 5.540647456e+01, -8.170017489e+00, 5.295569200e-01, -1.269556386e-02],
        [24, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.333724128e+01, 1.999999854e+00, 4.365566411e+02, -2.672774427e+02, 6.001631369e+01, -5.895458454e+00, 2.149710735e-01, -2.393534124e+02, 1.020845165e+02, -1.624744211e+01, 1.150387566e+00, -3.057723021e-02],
        [25, 6.000000000e+00, 7.650499797e+00, 1.999860000e+09, -1.328399669e+01, 2.000000008e+00, 6.461381990e+02, -3.918546518e+02, 8.769548644e+01, -8.618784385e+00, 3.150660827e-01, -2.597409979e+02, 1.113332866e+02, -1.782124571e+01, 1.269519197e+00, -3.396126698e-02],
        [26, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.330103000e+01, 1.999999998e+00, 4.261007401e+02, -2.588846763e+02, 5.764613910e+01, -5.609660122e+00, 2.024165636e-01, -1.982896712e+02, 8.274273985e+01, -1.284074215e+01, 8.845687432e-01, -2.282143299e-02],
        [27, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.332790165e+01, 1.999999922e+00, 4.006816638e+02, -2.439311564e+02, 5.435031497e+01, -5.287693457e+00, 1.906696163e-01, -2.205075564e+02, 9.262919772e+01, -1.448909443e+01, 1.006686819e+00, -2.621294059e-02],
        [28, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.334678710e+01, 1.999999939e+00, 3.967750019e+02, -2.411866801e+02, 5.364872608e+01, -5.210295834e+00, 1.875525119e-01, -2.516823030e+02, 1.065117131e+02, -1.680533335e+01, 1.178363534e+00, -3.098194406e-02],
        [29, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.344369664e+01, 1.999999853e+00, 2.437671888e+02, -1.499592208e+02, 3.332221026e+01, -3.206587185e+00, 1.138639692e-01, -2.874130637e+02, 1.223381969e+02, -1.943178054e+01, 1.371979484e+00, -3.633119448e-02],
        [30, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.338721562e+01, 1.999999911e+00, 3.914867984e+02, -2.378147085e+02, 5.284517777e+01, -5.126420186e+00, 1.843322562e-01, -3.235063319e+02, 1.384252948e+02, -2.211844479e+01, 1.571300198e+00, -4.187323186e-02],
        [31, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.335654643e+01, 1.999999847e+00, 4.325820127e+02, -2.614587597e+02, 5.793273998e+01, -5.611190206e+00, 2.015836827e-01, -3.359152840e+02, 1.437507638e+02, -2.297457475e+01, 1.632470701e+00, -4.351215346e-02],
        [32, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.337675047e+01, 1.999999960e+00, 4.388195965e+02, -2.642662297e+02, 5.834159168e+01, -5.629419790e+00, 2.014339673e-01, -3.430730654e+02, 1.467102631e+02, -2.343160019e+01, 1.663765504e+00, -4.431369286e-02],
        [33, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.339794046e+01, 2.000000074e+00, 3.931399547e+02, -2.363700718e+02, 5.197696913e+01, -4.987097655e+00, 1.772567576e-01, -3.501570134e+02, 1.497141578e+02, -2.390888062e+01, 1.697503580e+00, -4.520887478e-02],
        [34, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.342021680e+01, 2.000000064e+00, 3.772588127e+02, -2.256347960e+02, 4.929790851e+01, -4.694628847e+00, 1.654667382e-01, -3.481053019e+02, 1.486490112e+02, -2.370745096e+01, 1.680991482e+00, -4.471064364e-02],
        [35, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.344369666e+01, 1.999999864e+00, 3.344685842e+02, -1.994816236e+02, 4.332267376e+01, -4.090542180e+00, 1.426839031e-01, -3.227660675e+02, 1.370301996e+02, -2.171543883e+01, 1.529681552e+00, -4.041331983e-02],
        [36, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.345593194e+01, 1.999999999e+00, 3.004054446e+02, -1.781334135e+02, 3.834850324e+01, -3.580074471e+00, 1.232168921e-01, -2.980827664e+02, 1.257508661e+02, -1.978792154e+01, 1.383723149e+00, -3.628014907e-02],
        [37, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.310790583e+01, 2.000000075e+00, -3.687188343e+01, 1.054409719e+01, -8.516586814e-01, 9.339751003e-03, 8.809383936e-04, -2.699384784e+02, 1.129635316e+02, -1.761447452e+01, 1.219971043e+00, -3.166503704e-02],
        [38, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.304095795e+01, 1.999999892e+00, 1.969969064e+02, -1.286503864e+02, 3.008431767e+01, -3.031946980e+00, 1.124456346e-01, -2.331258613e+02, 9.627987243e+01, -1.478515961e+01, 1.007215642e+00, -2.567873120e-02],
        [39, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.306048023e+01, 1.999999916e+00, 2.891710763e+02, -1.819536752e+02, 4.158265841e+01, -4.128940218e+00, 1.515168697e-01, -1.997404800e+02, 8.119476676e+01, -1.223426670e+01, 8.159269666e-01, -2.031079820e-02],
        [40, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.308092198e+01, 2.000000013e+00, 3.393782172e+02, -2.103908454e+02, 4.758278737e+01, -4.688308235e+00, 1.709723418e-01, -1.549247582e+02, 6.091403935e+01, -8.799307373e+00, 5.578963961e-01, -1.305663921e-02],
        [41, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.316749062e+01, 1.999999920e+00, 2.748604341e+02, -1.706429616e+02, 3.843757441e+01, -3.759045290e+00, 1.358263430e-01, -1.163607425e+02, 4.350905533e+01, -5.859305970e+00, 3.376426246e-01, -6.881281652e-03],
        [42, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.318708720e+01, 2.000000093e+00, 3.203285955e+02, -1.966282865e+02, 4.398204769e+01, -4.283031482e+00, 1.543480828e-01, -9.364181222e+01, 3.329814493e+01, -4.141689265e+00, 2.095170962e-01, -3.304665813e-03],
        [43, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.314266674e+01, 1.999999876e+00, 4.184977165e+02, -2.552902161e+02, 5.707764818e+01, -5.576436872e+00, 2.020184726e-01, -8.395646154e+01, 2.898228589e+01, -3.422356654e+00, 1.564059753e-01, -1.838508896e-03],
        [44, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.322914744e+01, 1.999999909e+00, 3.243555305e+02, -1.978255470e+02, 4.397580841e+01, -4.256142657e+00, 1.524431452e-01, -5.506292375e+01, 1.599310639e+01, -1.237152904e+00, -6.611574411e-03, 2.712232383e-03],
        [45, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.325181249e+01, 2.000000089e+00, 3.037823599e+02, -1.856628295e+02, 4.128167884e+01, -3.991656133e+00, 1.427469878e-01, -5.014186072e+01, 1.386962969e+01, -8.950806420e-01, -3.095321225e-02, 3.357984426e-03],
        [46, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.340893584e+01, 2.000000073e+00, 3.529797051e+02, -2.101512262e+02, 4.563946029e+01, -4.315279704e+00, 1.509248358e-01, -4.815922691e+01, 1.301508788e+01, -7.580854951e-01, -4.059091985e-02, 3.608993811e-03],
        [47, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.328399669e+01, 2.000000008e+00, 3.074953924e+02, -1.872462583e+02, 4.149827252e+01, -4.000811852e+00, 1.426973118e-01, -4.897188379e+01, 1.335300002e+01, -8.110051997e-01, -3.684788190e-02, 3.508156457e-03],
        [48, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.322914743e+01, 1.999999904e+00, 4.059717166e+02, -2.462737702e+02, 5.472040126e+01, -5.311320062e+00, 1.911670149e-01, -5.901534554e+01, 1.791385249e+01, -1.587065943e+00, 2.182673278e-02, 1.845559896e-03],
        [49, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.320760815e+01, 1.999999973e+00, 4.369774251e+02, -2.639721687e+02, 5.849617557e+01, -5.667842049e+00, 2.037342202e-01, -7.399698219e+01, 2.469785523e+01, -2.737881327e+00, 1.085351830e-01, -6.022720695e-04],
        [50, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.322184869e+01, 1.999999993e+00, 4.289361021e+02, -2.585593024e+02, 5.714058683e+01, -5.518600115e+00, 1.976499817e-01, -9.269047286e+01, 3.314422349e+01, -4.167341855e+00, 2.159629039e-01, -3.626802503e-03],
        [51, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.323657166e+01, 1.999999946e+00, 3.866985836e+02, -2.328379698e+02, 5.128884878e+01, -4.929614910e+00, 1.755331333e-01, -1.067869310e+02, 3.950715983e+01, -5.243321447e+00, 2.967791238e-01, -5.901223876e-03],
        [52, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.325181248e+01, 2.000000083e+00, 3.947511198e+02, -2.363799049e+02, 5.179393756e+01, -4.951603918e+00, 1.753404387e-01, -1.069681982e+02, 3.995521754e+01, -5.382071424e+00, 3.120248901e-01, -6.467957474e-03],
        [53, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.326760745e+01, 2.000000205e+00, 3.694394448e+02, -2.204699428e+02, 4.806381052e+01, -4.565474883e+00, 1.604614344e-01, -1.180749905e+02, 4.460080701e+01, -6.105217447e+00, 3.616537171e-01, -7.733059623e-03],
        [54, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.328399667e+01, 2.000000001e+00, 3.423943987e+02, -2.041330669e+02, 4.437639784e+01, -4.197363553e+00, 1.467594367e-01, -1.288973984e+02, 4.985324046e+01, -7.056041375e+00, 4.378018318e-01, -1.000965926e-02],
        [55, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.297881025e+01, 1.999999927e+00, -7.663422017e+01, 3.462700567e+01, -6.273553579e+00, 5.487612834e-01, -1.912897528e-02, -1.318428276e+02, 5.081036112e+01, -7.154907590e+00, 4.405355674e-01, -9.955685075e-03],
        [56, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.290657751e+01, 1.999999869e+00, 1.084179205e+02, -7.602229206e+01, 1.843754298e+01, -1.892451591e+00, 7.085434176e-02, -1.346311376e+02, 5.207427468e+01, -7.369834199e+00, 4.568138610e-01, -1.041859875e-02],
        [57, 6.000000000e+00, 7.725500002e+00, 2.824880000e+09, -1.292445241e+01, 1.999999898e+00, 2.995898890e+02, -1.889477671e+02, 4.336642429e+01, -4.330424108e+00, 1.599942758e-01, 5.503972208e+00, -1.227641064e+01, 3.699182312e+00, -3.884476060e-01, 1.375966896e-02],
        [58, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.293554133e+01, 1.999999890e+00, 1.709135500e+02, -1.120124681e+02, 2.615893820e+01, -2.624416758e+00, 9.674223967e-02, -1.375860132e+02, 5.337811974e+01, -7.586786386e+00, 4.730023198e-01, -1.087482303e-02],
        [59, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.293554133e+01, 1.999999890e+00, 1.214691988e+02, -8.336119630e+01, 1.996468944e+01, -2.032283439e+00, 7.562254632e-02, -1.631005912e+02, 6.472051894e+01, -9.476098737e+00, 6.127875286e-01, -1.475060958e-02],
        [60, 6.000000000e+00, 7.849500202e+00, 5.000350000e+09, -1.294309494e+01, 1.999999967e+00, 1.302719596e+02, -8.835087414e+01, 2.101971144e+01, -2.131084478e+00, 7.908549730e-02, -1.692901279e+02, 6.742727614e+01, -9.920661139e+00, 6.453186854e-01, -1.564524492e-02],
        [61, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.295078139e+01, 1.999999905e+00, 1.127680235e+02, -7.782238836e+01, 1.865126163e+01, -1.895116816e+00, 7.030502833e-02, -2.059821608e+02, 8.384774285e+01, -1.267344799e+01, 8.502354115e-01, -2.135994609e-02],
        [62, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.295860692e+01, 1.999999936e+00, 1.203145109e+02, -8.212556537e+01, 1.956606386e+01, -1.981212240e+00, 7.333626288e-02, -2.158058793e+02, 8.810144391e+01, -1.336380022e+01, 9.000362964e-01, -2.270715579e-02],
        [63, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.296657573e+01, 1.999999918e+00, 1.212159597e+02, -8.256559477e+01, 1.964122173e+01, -1.986442056e+00, 7.345564343e-02, -2.278531434e+02, 9.336519465e+01, -1.422588608e+01, 9.627883381e-01, -2.441986614e-02],
        [64, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.298296617e+01, 1.999999921e+00, 1.689382403e+02, -1.099987696e+02, 2.551961464e+01, -2.543234152e+00, 9.313568005e-02, -2.282716670e+02, 9.348611199e+01, -1.423588448e+01, 9.628551072e-01, -2.440492772e-02],
        [65, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.299139910e+01, 1.999999880e+00, 1.724155378e+02, -1.120798437e+02, 2.598264738e+01, -2.588807295e+00, 9.481417896e-02, -2.322687147e+02, 9.517466656e+01, -1.450332749e+01, 9.817069914e-01, -2.490386807e-02],
        [66, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.298716240e+01, 1.999999941e+00, 1.286079419e+02, -8.646296410e+01, 2.039801258e+01, -2.050839207e+00, 7.549033493e-02, -2.420048480e+02, 9.935663043e+01, -1.517653800e+01, 1.029875015e+00, -2.619626869e-02],
        [67, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.299567846e+01, 1.999999971e+00, 1.182799697e+02, -8.043389241e+01, 1.908027783e+01, -1.923209794e+00, 7.087268462e-02, -2.464462609e+02, 1.012059056e+02, -1.546468270e+01, 1.049814070e+00, -2.671320158e-02],
        [68, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.300436459e+01, 1.999999966e+00, 1.150510247e+02, -7.859576077e+01, 1.868688175e+01, -1.885844183e+00, 6.954765052e-02, -2.457555063e+02, 1.007538481e+02, -1.536692833e+01, 1.041070997e+00, -2.643279207e-02],
        [69, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.300877391e+01, 2.000000000e+00, 1.266280406e+02, -8.514491730e+01, 2.007089332e+01, -2.015475088e+00, 7.409191965e-02, -2.492442707e+02, 1.021615320e+02, -1.557878384e+01, 1.055183253e+00, -2.678362279e-02],
        [70, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.301772826e+01, 1.999999912e+00, 1.224253568e+02, -8.281395858e+01, 1.958609738e+01, -1.970785167e+00, 7.255458061e-02, -2.488808342e+02, 1.018569466e+02, -1.550601866e+01, 1.048325396e+00, -2.655661748e-02],
        [71, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.303151733e+01, 2.000000051e+00, 1.862181262e+02, -1.199038630e+02, 2.763107534e+01, -2.742586837e+00, 1.001956495e-01, -2.403102476e+02, 9.796272016e+01, -1.484525920e+01, 9.987147871e-01, -2.516533876e-02],
        [72, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.304575796e+01, 2.000000081e+00, 2.297759959e+02, -1.448485621e+02, 3.295877082e+01, -3.245850428e+00, 1.179456377e-01, -2.282155654e+02, 9.249921555e+01, -1.392266984e+01, 9.297052139e-01, -2.323558576e-02],
        [73, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.306048022e+01, 1.999999910e+00, 2.646909006e+02, -1.647716545e+02, 3.719903613e+01, -3.645113853e+00, 1.319890617e-01, -2.165150972e+02, 8.722660467e+01, -1.303415548e+01, 8.633600348e-01, -2.138300143e-02],
        [74, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.308092196e+01, 2.000000008e+00, 2.251239174e+02, -1.414731209e+02, 3.206048507e+01, -3.142433101e+00, 1.135971917e-01, -2.070173544e+02, 8.296725365e+01, -1.231986936e+01, 8.102887128e-01, -1.990853407e-02],
        [75, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.309151488e+01, 1.999999984e+00, 2.627532736e+02, -1.629008146e+02, 3.661592385e+01, -3.571257833e+00, 1.286871297e-01, -1.945762063e+02, 7.740995255e+01, -1.139129234e+01, 7.415172466e-01, -1.800335280e-02],
        [76, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.310790581e+01, 2.000000068e+00, 2.644549626e+02, -1.637369900e+02, 3.675734857e+01, -3.580665992e+00, 1.288721975e-01, -1.725967865e+02, 6.755389456e+01, -9.737633351e+00, 6.184954292e-01, -1.457897448e-02],
        [77, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.311918599e+01, 1.999999933e+00, 2.677629012e+02, -1.650589135e+02, 3.690999414e+01, -3.582378706e+00, 1.284763849e-01, -1.584140848e+02, 6.122430396e+01, -8.680876005e+00, 5.402879020e-01, -1.241386995e-02],
        [78, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.319382006e+01, 2.000000009e+00, 2.420702029e+02, -1.484461630e+02, 3.292288306e+01, -3.162757529e+00, 1.121487556e-01, -1.319886050e+02, 4.940494114e+01, -6.702740089e+00, 3.934770465e-01, -8.336673895e-03],
        [79, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.320760814e+01, 1.999999969e+00, 2.346714957e+02, -1.439356552e+02, 3.189416251e+01, -3.059071523e+00, 1.082595858e-01, -1.130109430e+02, 4.093029258e+01, -5.286747014e+00, 2.885753389e-01, -5.428939868e-03],
        [80, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.316115147e+01, 2.000000093e+00, 2.747370538e+02, -1.689673404e+02, 3.771696324e+01, -3.655841153e+00, 1.309852214e-01, -9.001823908e+01, 3.066094857e+01, -3.570459523e+00, 1.613797666e-01, -1.901561361e-03],
        [81, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.313667715e+01, 2.000000002e+00, 3.142563781e+02, -1.916613838e+02, 4.259167223e+01, -4.119713271e+00, 1.474792530e-01, -7.642731867e+01, 2.462410146e+01, -2.566977318e+00, 8.741068396e-02, 1.388590928e-04],
        [82, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.314266674e+01, 1.999999876e+00, 3.509258060e+02, -2.125470710e+02, 4.702461797e+01, -4.535380912e+00, 1.620138781e-01, -5.173355302e+01, 1.362015056e+01, -7.321282362e-01, -4.826261322e-02, 3.892879264e-03],
        [83, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.315490164e+01, 1.999999944e+00, 3.399729483e+02, -2.056319770e+02, 4.539614689e+01, -4.366195994e+00, 1.554792165e-01, -4.131443229e+01, 8.986236911e+00, 3.924628986e-02, -1.052060828e-01, 5.466043586e-03],
        [84, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.316749062e+01, 1.999999920e+00, 3.640602841e+02, -2.190164327e+02, 4.815603439e+01, -4.616573783e+00, 1.639147626e-01, -3.256862965e+01, 5.115606198e+00, 6.800853161e-01, -1.522315744e-01, 6.756786448e-03],
        [85, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.318045630e+01, 2.000000044e+00, 3.766488275e+02, -2.257321142e+02, 4.947300991e+01, -4.728919006e+00, 1.674240471e-01, -2.300947210e+01, 8.615223509e-01, 1.388425307e+00, -2.045157608e-01, 8.200511055e-03],
        [86, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.319382005e+01, 2.000000006e+00, 3.443622947e+02, -2.064342780e+02, 4.516044966e+01, -4.302253084e+00, 1.516667044e-01, -5.399039282e+00, -7.002814559e+00, 2.702516748e+00, -3.018766003e-01, 1.089953798e-02],
        [87, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.291364147e+01, 2.000000217e+00, -3.706791591e+01, 1.118013187e+01, -1.057728859e+00, 3.312859839e-02, -3.138341244e-06, -3.451314336e+00, -7.779254134e+00, 2.816269849e+00, -3.090776388e-01, 1.106424389e-02],
        [88, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.284163724e+01, 1.999999954e+00, 6.125934670e+01, -4.855548659e+01, 1.248551381e+01, -1.323304763e+00, 5.060744172e-02, -6.021643455e+00, -6.580234329e+00, 2.607440108e+00, -2.929625239e-01, 1.059951856e-02],
        [89, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.285387248e+01, 2.000000090e+00, 1.350863292e+02, -9.126618691e+01, 2.169932948e+01, -2.201947573e+00, 8.186860720e-02, 1.937135880e+01, -1.787129621e+01, 4.485878662e+00, -4.315325969e-01, 1.442445798e-02],
        [90, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.286966604e+01, 1.999999970e+00, 1.784388998e+02, -1.161623817e+02, 2.702376618e+01, -2.704797298e+00, 9.957279361e-02, 2.216057166e+01, -1.904990091e+01, 4.671627339e+00, -4.444534802e-01, 1.475921763e-02],
        [91, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.287289489e+01, 1.999999951e+00, 1.368355213e+02, -9.179790820e+01, 2.169910915e+01, -2.190249857e+00, 8.102241740e-02, 4.516580666e+00, -1.118102949e+01, 3.357662550e+00, -3.470694353e-01, 1.205639951e-02],
        [92, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.287942629e+01, 2.000000032e+00, 1.427130850e+02, -9.499714618e+01, 2.234475916e+01, -2.247599931e+00, 8.291713193e-02, 1.341991149e+01, -1.518503354e+01, 4.030838171e+00, -3.972060658e-01, 1.345248084e-02],
        [93, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.288605524e+01, 1.999999761e+00, 2.341801100e+01, -2.506119713e+01, 7.023029272e+00, -7.610742531e-01, 2.903245750e-02, -3.575331738e+01, 7.276302226e+00, 1.906771859e-01, -1.059475755e-01, 5.184029625e-03],
        [94, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.288272835e+01, 1.999999941e+00, 1.287618322e+02, -8.721780968e+01, 2.073255323e+01, -2.100572716e+00, 7.794295578e-02, -2.307262580e+01, 1.113132278e+00, 1.305250601e+00, -1.948949139e-01, 7.829116438e-03],
        [95, 6.000000000e+00, 7.951499931e+00, 7.998340000e+09, -1.288940956e+01, 1.999999880e+00, 1.334821220e+02, -8.985337775e+01, 2.127928526e+01, -2.150628571e+00, 7.965294640e-02, -3.518662723e+01, 6.514543434e+00, 4.030862442e-01, -1.279850170e-01, 5.970168353e-03],
        [96, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.290553004e+01, 2.000000198e+00, 4.545581472e+01, -3.771304300e+01, 9.729129321e+00, -1.017037014e+00, 3.807733199e-02, -4.973805034e+01, 1.342335334e+01, -8.221139917e-01, -3.176841835e-02, 3.146810827e-03],
        [97, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.291150963e+01, 2.000000019e+00, 4.689042092e+01, -3.843347264e+01, 9.859294531e+00, -1.027014690e+00, 3.834833665e-02, -4.657434145e+01, 1.204637835e+01, -5.982449163e-01, -4.786919243e-02, 3.579251285e-03],
        [98, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.290833198e+01, 1.999999824e+00, 1.337584189e+01, -1.907284620e+01, 5.691614909e+00, -6.307838734e-01, 2.430868142e-02, -5.573362773e+01, 1.615667599e+01, -1.288960621e+00, 3.655033732e-03, 2.140047522e-03],
        [99, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.291435263e+01, 1.999999988e+00, 1.376201293e+01, -1.919251815e+01, 5.693799461e+00, -6.287500644e-01, 2.416045199e-02, -4.914211254e+01, 1.314247998e+01, -7.739336035e-01, -3.530513333e-02, 3.241293077e-03],
        [100, 6.000000000e+00, 8.000000000e+00, 1.000000000e+10, -1.292045700e+01, 2.000000004e+00, 1.277081775e+01, -1.854047224e+01, 5.534680382e+00, -6.118054153e-01, 2.349768815e-02, -5.074293980e+01, 1.383260974e+01, -8.858904786e-01, -2.718885953e-02, 3.019620454e-03]
    ], dtype=dtype, device=device)

    ln10 = torch.log(torch.tensor(10., device=device))
    electron_mass_c2 = 0.510998910*MeV
    wl = h_Planck * c_light
    cm_hep = cm

    def compute_scattering_function(x: torch.Tensor,
                                    Z: torch.Tensor,
                                    scat_func_fit_param: torch.Tensor = scat_func_fit_param,
                                    ln10: float = ln10) -> torch.Tensor:
        result = Z.clone()
        Z_long = Z.long()
        
        x_threshold = torch.gather(scat_func_fit_param[:, 3], 0, Z_long)
        valid_mask = x <= x_threshold
        
        if valid_mask.any():
            lgq = torch.zeros_like(x)
            lgq[valid_mask] = torch.log(x[valid_mask]) / ln10
            
            threshold1 = torch.gather(scat_func_fit_param[:, 1], 0, Z_long)
            threshold2 = torch.gather(scat_func_fit_param[:, 2], 0, Z_long)
            
            mask1 = valid_mask & (lgq < threshold1)
            mask2 = valid_mask & (lgq >= threshold1) & (lgq < threshold2)
            mask3 = valid_mask & (lgq >= threshold2)
            
            if mask1.any():
                Z1 = Z_long[mask1]
                lgq1 = lgq[mask1]
                
                params1 = torch.stack([
                    torch.gather(scat_func_fit_param[:, 4], 0, Z1),
                    torch.gather(scat_func_fit_param[:, 5], 0, Z1)
                ], dim=1)
                
                value = params1[:, 0] + lgq1 * params1[:, 1]
                result[mask1] = torch.exp(value * ln10)
            
            if mask2.any():
                Z2 = Z_long[mask2]
                lgq2 = lgq[mask2]
                
                params2 = torch.stack([
                    torch.gather(scat_func_fit_param[:, 6], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 7], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 8], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 9], 0, Z2),
                    torch.gather(scat_func_fit_param[:, 10], 0, Z2)
                ], dim=1)
                
                value = params2[:, 0] + lgq2 * (params2[:, 1] + lgq2 * (params2[:, 2] + 
                                                                    lgq2 * (params2[:, 3] + 
                                                                            lgq2 * params2[:, 4])))
                result[mask2] = torch.exp(value * ln10)
            
            if mask3.any():
                Z3 = Z_long[mask3]
                lgq3 = lgq[mask3]
                
                params3 = torch.stack([
                    torch.gather(scat_func_fit_param[:, 11], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 12], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 13], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 14], 0, Z3),
                    torch.gather(scat_func_fit_param[:, 15], 0, Z3)
                ], dim=1)
                
                value = params3[:, 0] + lgq3 * (params3[:, 1] + lgq3 * (params3[:, 2] + 
                                                                    lgq3 * (params3[:, 3] + 
                                                                            lgq3 * params3[:, 4])))
                result[mask3] = torch.exp(value * ln10)
        
        return result

    def theta_generator(energy: torch.Tensor,
                        Z: torch.Tensor,
                        generator: torch.Generator = rng,
                        electron_mass_c2: float = electron_mass_c2,
                        wl: float = wl,
                        cm_hep: float = cm_hep) -> torch.Tensor:
        """
        Generate scattering angles for batched inputs
        """

        
        energy_flat = energy.reshape(-1)
        Z_flat = Z.reshape(-1)
        batch_size = energy_flat.size(0)
        device = energy.device
        
        e0m = energy_flat / electron_mass_c2
        epsilon0_local = 1.0 / (1.0 + 2.0 * e0m)
        epsilon0_sq = epsilon0_local * epsilon0_local
        alpha1 = -torch.log(epsilon0_local)
        alpha2 = 0.5 * (1.0 - epsilon0_sq)
        wl_photon = wl / energy_flat
        
        theta = torch.zeros_like(energy_flat)
        pending = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        max_iter = 10
        
        for _ in range(max_iter):    
            pending_indices = torch.nonzero(pending).squeeze(-1)
            pending_count = pending_indices.size(0)
            
            rand_vals = torch.rand((pending_count, 3), device=device, generator=generator)
            
            p_e0m = e0m[pending_indices]
            p_epsilon0_sq = epsilon0_sq[pending_indices]
            p_alpha1 = alpha1[pending_indices]
            p_alpha2 = alpha2[pending_indices]
            p_wl_photon = wl_photon[pending_indices]
            p_Z = Z_flat[pending_indices]
            
            epsilon = torch.zeros(pending_count, device=device)
            alpha_ratio = p_alpha1 / (p_alpha1 + p_alpha2)
            method_mask = alpha_ratio > rand_vals[:, 0]
            
            if method_mask.any():
                epsilon[method_mask] = torch.exp(-p_alpha1[method_mask] * rand_vals[method_mask, 1])
            
            not_method_mask = ~method_mask
            if not_method_mask.any():
                epsilon_sq_temp = p_epsilon0_sq[not_method_mask] + (1.0 - p_epsilon0_sq[not_method_mask]) * rand_vals[not_method_mask, 1]
                epsilon[not_method_mask] = torch.sqrt(epsilon_sq_temp)
            
            epsilon_sq = epsilon * epsilon
            one_cos_t = (1.0 - epsilon) / (epsilon * p_e0m)
            sinT2 = one_cos_t * (2.0 - one_cos_t)
            x = torch.sqrt(one_cos_t / 2.0) * cm_hep / p_wl_photon
            
            scatter_func = compute_scattering_function(x, p_Z)
            
            g_reject = (1.0 - epsilon * sinT2 / (1.0 + epsilon_sq)) * scatter_func
            
            accepted = g_reject > rand_vals[:, 2] * p_Z
            
            if accepted.any():
                accepted_cos_theta = 1.0 - one_cos_t[accepted]
                accepted_indices = pending_indices[accepted]
                
                theta[accepted_indices] = torch.acos(accepted_cos_theta)
                
                pending[accepted_indices] = False
        
        if pending.any():
            remaining_indices = torch.nonzero(pending).squeeze(-1)
            remaining_count = remaining_indices.size(0)
            
            r_e0m = e0m[remaining_indices]
            r_alpha1 = alpha1[remaining_indices]
            
            rand = torch.rand(remaining_count, device=device, generator=generator)
            
            r_epsilon = torch.exp(-r_alpha1 * rand)
            r_one_cos_t = (1.0 - r_epsilon) / (r_epsilon * r_e0m)
            r_cos_theta = 1.0 - r_one_cos_t
            
            theta[remaining_indices] = torch.acos(r_cos_theta)
        
        return theta
    return theta_generator

class Process(ABC):
    """ Класс процесса """

    def __init__(self, attenuation_database=None, device=None, rng=None, dtype=torch.float32):
        """ Конструктор процесса """
        attenuation_database = attenuation_database if attenuation_database is None else attenuation_database
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.rng = torch.default_generator if rng is None else rng
        self._energy_range = torch.tensor([1*keV, 1*MeV], device=self.device)
        self._construct_attenuation_function(attenuation_database)
        self.dtype = dtype

    def _construct_attenuation_function(self, attenuation_database):
        self.attenuation_function = AttenuationFunction(self, attenuation_database, device=self.device)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def energy_range(self):
        return self._energy_range

    @energy_range.setter
    def energy_range(self, value):
        self._energy_range = value
        self._construct_attenuation_function()

    def get_LAC(self, particle, material):
        energy = particle.energy
        LAC = self.attenuation_function(material, energy).to(self.dtype)
        return LAC

    def __call__(self, particle, material, batch_indices):
        """ Применить процесс """
        return ParticleInteractionData(particle,
                                       process_name=self.name,
                                       energy_deposit=torch.zeros_like(particle.energy, device=self.device),
                                       scattering_angles=torch.zeros((1, 2, particle.energy.shape[1]), device=self.device),
                                       batch_indices=batch_indices)


class PhotoelectricEffect(Process):
    """ Класс фотоэффекта """
    def __init__(self, rng=None, attenuation_database=None, device="cpu", dtype=torch.float32):
        super().__init__(attenuation_database, device, rng, dtype)

    def __call__(self, particle, material, batch_indices):
        """ Применить фотоэффект """
        interaction_data = super().__call__(particle, material, batch_indices)
        energy_deposit = particle.energy.clone()
        particle.change_energy(energy_deposit)
        interaction_data.energy_deposit = energy_deposit
        return interaction_data
    

class CoherentScattering(Process):
    """ Класс когерентного рассеяния """
    
    def __init__(self, rng=None, attenuation_database=None, device="cpu", dtype=torch.float32):
        Process.__init__(self, attenuation_database, device, rng, dtype)
        if device == "cpu":
            self.theta_generator = initialize_cpu_coherent(self.rng)
        else:
            self.theta_generator = initialize_cuda_coherent(self.rng, device=device)

    def generate_theta(self, particle, material):
        """ Сгенерировать угол рассеяния - theta """
        energy = particle.energy
        Z = material.Zeff
        theta = self.theta_generator(energy, Z)
        return theta

    def generate_phi(self, size):
        """ Сгенерировать угол рассеяния - phi """
        phi = torch.pi * (torch.rand(size, device=self.device) * 2 - 1)
        return phi

    def __call__(self, particle, material, batch_indices):
        """ Применить когерентное рассеяние """
        size = particle.size
        theta = self.generate_theta(particle, material)
        phi = self.generate_phi(size)
        theta = theta.expand_as(phi)
        phi = phi.expand_as(theta)
        particle.rotate(theta, phi)
        interaction_data = super().__call__(particle, material, batch_indices)
        interaction_data.scattering_angles = torch.stack((theta, phi), dim=1)
        return interaction_data


class ComptonScattering(CoherentScattering):
    """ Класс эффекта Комптона """

    def __init__(self, rng=None, attenuation_database=None, device="cpu", dtype=torch.float32):
        Process.__init__(self, attenuation_database, device, rng, dtype)
        if device == "cpu":
            self.theta_generator = initialize_cpu_compton(self.rng)
        else:
            self.theta_generator = initialize_cuda_compton(self.rng, device=device)

    def calculate_energy_deposit(self, theta, particle_energy):
        """ Вычислить изменения энергий """
        k = particle_energy / (0.510998910 * MeV)
        k1_cos = k * (1 - torch.cos(theta))
        energy_deposit = particle_energy * k1_cos / (1 + k1_cos)
        return energy_deposit

    def __call__(self, particle, material, batch_indices):
        """ Применить эффект Комптона """
        interaction_data = super().__call__(particle, material, batch_indices)
        theta = interaction_data.scattering_angles[:, 0, :]
        energy_deposit = self.calculate_energy_deposit(theta, particle.energy)
        particle.change_energy(energy_deposit)
        interaction_data.energy_deposit = energy_deposit
        return interaction_data


class PairProduction(Process):
    """ Класс эффекта образования электрон-позитронных пар """
    pass

processes_list = [
    PhotoelectricEffect,
    ComptonScattering,
    CoherentScattering,
    # PairProduction
]

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
        self.distance_epsilon = 0.5  # change that if you have some problems with code
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
            distance, inside = numba_ray_casting(position.to(torch.float32).numpy(), direction.to(torch.float32).cpu().numpy(), np.array(self.half_size, dtype=np.float32), self.distance_epsilon)
            distance = torch.from_numpy(distance).to(self.dtype)
            inside = torch.from_numpy(inside).to(torch.bool)
            return distance, inside
        else:
            return torch_ray_casting(position, direction, self.half_size, self.distance_epsilon)

class ElementaryVolume:
    """ Базовый класс элементарного объёма """
    _counter = count(1)

    def __init__(self, geometry, material, name=None, input_shape=None, dtype=torch.float32):
        """ Конструктор объёма """
        self.geometry = geometry
        self.material = material
        self.dtype = dtype
        self.name = f'{self.__class__.__name__}{next(self._counter)}' if name is None else name
        self._dublicate_counter = count(1)
        self.device = self.geometry.device

    def __init_subclass__(cls):
        cls._counter = count(1)

    def __repr__(self):
        return f'{self.name}'

    @property
    def size(self):
        return self.geometry.size

    @size.setter
    def size(self, value):
        self.geometry.size = value

    def dublicate(self):
        result = deepcopy(self)
        result.name = f'{self.name}.{next(self._dublicate_counter)}'
        return result

    def check_inside(self, position):
        """ Проверка на попадание в объём """
        return self.geometry.check_inside(position)

    def check_outside(self, position):
        """ Проверка на непопадание в объём """
        return self.geometry.check_outside(position)

    def cast_path(self, position, direction, batch_indices=None):
        """ Определение объекта местонахождения и длины пути частицы """
        current_volume = VolumeArray(position.shape[0], device=position.device)
        distance, inside = self.geometry.cast_path(position, direction)
        current_volume[inside] = self
        return distance, current_volume

    def get_material_by_position(self, position, batch_indices=None):
        """ Получить материал по координаты """
        material = MaterialArray(position.shape[0], device=position.device)
        inside = self.geometry.check_inside(position)  # bug maybe, returns a dimension [1, N]
        particle_idx = torch.where(inside)
        material[particle_idx] = self.material
        return material


class VolumeWithChilds(ElementaryVolume):
    """ Базовый класс объёма с детьми """    

    def __init__(self, geometry, material, name=None, input_shape=None, dtype=torch.float32):
        super().__init__(geometry, material, name, dtype)
        self.childs = []
        self.input_shape = input_shape

    def dublicate(self):
        result = super().dublicate()
        childs = result.childs
        result.childs = []
        for child in childs:
            child.dublicate()
        return result
    
    def cast_path(self, position, direction, batch_indices=None, entry_point=False):
        if entry_point:
            distance, current_volume = super().cast_path(position.flatten(0, 1), direction.flatten(0, 1), batch_indices)
        else:
            distance, current_volume = super().cast_path(position, direction, batch_indices)

        if len(self.childs) > 0:
            if entry_point:
                current_volume = current_volume.reshape(self.input_shape)
                distance = distance.reshape(self.input_shape)
                inside = current_volume.data != 0
                particle_batch_indices, particle_indices = torch.where(inside)
                batch_indices = particle_batch_indices.clone()
            else:
                inside = current_volume.data != 0
                particle_batch_indices = True
                particle_indices = torch.where(inside)[0]

                batch_indices = batch_indices[particle_indices]

            position_inside = position[particle_batch_indices, particle_indices]
            direction_inside = direction[particle_batch_indices, particle_indices]
            distance_inside = distance[particle_batch_indices, particle_indices]
            current_volume_inside = current_volume[particle_batch_indices, particle_indices]
            if len(batch_indices) != 0:
                distance_to_child = torch.full_like(distance_inside, float('inf'))
                for child in self.childs:
                    child_dist, child_vol = child.cast_path(position_inside, direction_inside, batch_indices)
                    vol_mask = child_vol.data != 0
                    current_volume_inside[vol_mask] = child_vol[vol_mask] # there was torch.where but VolumeArray not compatible
                    distance_to_child = torch.minimum(distance_to_child, child_dist)

                combined_dist = torch.minimum(distance_inside, distance_to_child)
                distance[particle_batch_indices, particle_indices] = combined_dist
                current_volume[particle_batch_indices, particle_indices] = current_volume_inside

        return distance, current_volume

    def get_material_by_position(self, position, batch_indices):
        if len(position.shape) == 3:
            material = super().get_material_by_position(position.squeeze(0), batch_indices)  # bug solved, fmod on remainder fixed
        else:
            material = super().get_material_by_position(position, batch_indices)
        if len(self.childs) > 0:
            inside = material.data != 0
            particle_indices = torch.where(inside)
            if len(position.shape) == 3:
                position_inside = position.squeeze(0)[particle_indices]
            else:
                position_inside = position[particle_indices]
            material_inside = material[particle_indices]
            batch_indices_inside = batch_indices[particle_indices]
            for child in self.childs:
                child_material = child.get_material_by_position(position_inside, batch_indices_inside)
                inside_child = child_material.data != 0 
                material_inside[inside_child] = child_material[inside_child]
            
            material[particle_indices] = material_inside   
        return material

    def add_child(self, child):
        """ Добавить дочерний объём """
        assert isinstance(child, TransformableVolume), 'Только трансформируемый объём может быть дочерним'
        if child.parent is None:
            self.childs.append(child)
        elif child in self.childs:
            print('Добавляемый объём уже является дочерним данному объёму')
        else:
            print('Внимение! Добавляемый объём уже является дочерним. Новый родитель установлен')
            child.parent.childs.remove(child)
        child.parent = self


class TransformableVolume(ElementaryVolume):
    """ Базовый класс трансформируемого объёма """

    def __init__(self, geometry, material, batch_size, name=None, input_shape=None, dtype=torch.float32):
        super().__init__(geometry, material, name, input_shape, dtype)
        self.batch_size = batch_size
        self.transformation_matrix = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        self.parent = None

    def dublicate(self):
        result = super().dublicate()
        result.parent = None
        if self.parent is not None:
            result.set_parent(self.parent)
        return result

    @property
    def total_transformation_matrix(self):
        if isinstance(self.parent, TransformableVolume):
            return torch.bmm(self.transformation_matrix, self.parent.total_transformation_matrix)
        return self.transformation_matrix

    def convert_to_local_position(self, position, batch_indices, as_parent=True):
        """ Преобразовать в локальные координаты """
        if len(batch_indices) == 0:
            return position
        total_transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        total_transformation_matrix = torch.transpose(total_transformation_matrix, 1, 2)
        position_ones = torch.concat([position, torch.ones(position.shape[0], 1, device=self.device)], dim=-1)
        local_positions = torch.matmul(position_ones.unsqueeze(1), total_transformation_matrix[batch_indices]).squeeze(1)
        position = local_positions[:, :3]
        return position

    def convert_to_local_direction(self, direction, batch_indices, as_parent=True):
        """ Преобразовать в локальное направление """
        if len(batch_indices) == 0:
            return direction
        total_transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        total_transformation_matrix = torch.transpose(total_transformation_matrix[:, :3, :3], 1, 2)
        direction = torch.matmul(direction.unsqueeze(1), total_transformation_matrix[batch_indices]).squeeze(1)
        return direction

    def check_inside(self, position, batch_indices, local=False, as_parent=False):
        if not local:
            position = self.convert_to_local_position(position, batch_indices, as_parent)
        return super().check_inside(position)

    def check_outside(self, position, batch_indices, local=False, as_parent=False):
        if not local:
            position = self.convert_to_local_position(position, batch_indices, as_parent)
        return super().check_outside(position)

    def translate(self, x=0., y=0., z=0., inLocal=False):
        """ Переместить объём """
        if isinstance(x, float):
            x = torch.tensor(x).expand((self.batch_size, 1)).to(self.device)
        if isinstance(y, float):
            y = torch.tensor(y).expand((self.batch_size, 1)).to(self.device)
        if isinstance(z, float):
            z = torch.tensor(z).expand((self.batch_size, 1)).to(self.device)
        translation = torch.stack([x, y, z], dim=1)
        translation_matrix = compute_translation_matrix(-translation)
        if inLocal:
            self.transformation_matrix = torch.bmm(translation_matrix, self.transformation_matrix)
        else:
            self.transformation_matrix = torch.bmm(self.transformation_matrix, translation_matrix)


    def rotate(self, alpha=0., beta=0., gamma=0., rotation_center=[0., 0., 0.], inLocal=False):
        """ Повернуть объём вокруг координатных осей """
        if isinstance(alpha, float):
            alpha = torch.tensor(alpha).expand((self.batch_size, 1)).to(self.device)
        if isinstance(beta, float):
            beta = torch.tensor(beta).expand((self.batch_size, 1)).to(self.device)
        if isinstance(gamma, float):
            gamma = torch.tensor(gamma).expand((self.batch_size, 1)).to(self.device)
        if isinstance(rotation_center[0], float):
            rotation_center = torch.tensor(rotation_center).expand(self.batch_size, 3).unsqueeze(-1).to(self.device)  # bug, not tidy code...
        rotation_angles = torch.stack([alpha, beta, gamma], dim=1)
        rotation_matrix = compute_translation_matrix(-rotation_center)
        rotation_matrix = torch.bmm(rotation_matrix, compute_rotation_matrix(-rotation_angles))
        rotation_matrix = torch.bmm(rotation_matrix, compute_translation_matrix(rotation_center))
        if inLocal:
            self.transformation_matrix = torch.bmm(rotation_matrix, self.transformation_matrix)
        else:
            self.transformation_matrix = torch.bmm(self.transformation_matrix, rotation_matrix)

    def cast_path(self, position, direction, batch_indices, local=False, as_parent=True):
        if not local:
            position = self.convert_to_local_position(position, batch_indices, as_parent)
            direction = self.convert_to_local_direction(direction, batch_indices, as_parent)
        return super().cast_path(position, direction, batch_indices)

    def get_material_by_position(self, position, batch_indices, local=False, as_parent=True):
        if not local:
            position = self.convert_to_local_position(position, batch_indices, as_parent)
        material = super().get_material_by_position(position, batch_indices)
        return material

    def set_parent(self, parent):
        assert isinstance(parent, VolumeWithChilds), 'Этот объём не может быть родителем'
        parent.add_child(self)


class TransformableVolumeWithChild(TransformableVolume, VolumeWithChilds):
    """ Базовый класс трансформируемого объёма с детьми """  


class VolumeArray(NonuniqueArray):
    """ Класс списка объёмов """

    @property
    def material(self):
        """ Список материалов """
        material = MaterialArray(self.shape, device=self.device)
        for volume, indices in self.inverse_indices.items():
            if volume is None:
                continue
            material[indices] = volume.material
        return material

class WoodcockVolume(TransformableVolume):
    """
    Базовый класс Woodcock объёма
    """


class WoodcockParameticVolume(WoodcockVolume):
    """
    Класс параметричекого Woodcock объёма
    """

    def _parametric_function(self, position):
        return [], None

    def get_material_by_position(self, position, batch_indices, local=False, as_parent=True):
        if not local:
            position = self.convert_to_local_position(position, batch_indices)
        material = super().get_material_by_position(position, batch_indices, True)
        inside_indices = torch.where(material.data != 0)[0]
        indices, new_material = self._parametric_function(position[inside_indices])
        material[inside_indices[indices]] = new_material
        return material

class WoodcockVoxelVolume(WoodcockParameticVolume):
    """
    Класс воксельного Woodcock объёма
    
    [coordinates = (x, y, z)] = cm
    [material] = uint[:,:,:]
    [voxel_size] = cm
    """

    def __init__(self, voxel_size, material_distribution, batch_size=1, name=None, device=None, dtype=torch.float32):
        size = torch.tensor(material_distribution.shape, dtype=torch.int64, device=device) * voxel_size
        super().__init__(
            geometry=Box(size, device=device, dtype=dtype),
            material=None,
            name=name,
            batch_size=batch_size,
            dtype=dtype
        )
        self.material_distribution = material_distribution
        self._voxel_size_ratio = voxel_size / self.size

    @property
    def voxel_size(self):
        return self.size * self._voxel_size_ratio

    @voxel_size.setter
    def voxel_size(self, value):
        self._voxel_size_ratio = value / self.size

    @property
    @cache
    def material(self):
        material_list = self.material_distribution.material_list
        return max(material_list)

    @material.setter
    def material(self, value):
        pass

    def _parametric_function(self, position):
        indices = ((position + (self.size/2 - self.voxel_size/2)) / self.voxel_size).to(torch.int32)
        material = self.material_distribution.get_materials_at_indices(indices)
        return torch.ones_like(material.data, dtype=torch.bool), material

class ParametricParallelCollimator(WoodcockParameticVolume):
    """
    Torch-optimized class for parametric collimator with parallel channels
    """

    def __init__(self, size, hole_diameter, septa, material=None, name=None, batch_size=1, dtype=torch.float32, device="cpu"):
        material = material_database['Pb'] if material is None else material
        super().__init__(
            geometry=Box(torch.tensor(size, device=device), device=device, dtype=dtype),
            material=material,
            name=name,
            batch_size=batch_size,
            dtype=dtype
        )
        self._hole_diameter = torch.tensor(hole_diameter, device=device)
        self._septa = torch.tensor(septa, device=device)
        self._vacuum = material_database['Vacuum']
        self.device = device
        self._compute_constants()
    
    def _compute_constants(self):
        x_period = self._hole_diameter + self._septa
        y_period = torch.sqrt(torch.tensor(3.0, device=self.device)) * x_period
        self._period = torch.stack((x_period, y_period))
        self._a = torch.sqrt(torch.tensor(3.0, device=self.device)) / 4
        d = self._hole_diameter * 2 / torch.sqrt(torch.tensor(3.0, device=self.device))
        self._corner = self._period / 2
        self._ad = self._a * d
        self._ad_2 = self._ad / 2

    def _parametric_function(self, position):    
        position_parametric = position[:, :2]
        position_parametric = torch.remainder(position_parametric, self._period)
        position_parametric = torch.abs(position_parametric - self._corner)
        cond1 = position_parametric[:, 0] <= self._ad
        cond2 = self._a * position_parametric[:, 1] + position_parametric[:, 0] / 4 <= self._ad_2
        collimated = cond1 & cond2
        position_parametric = torch.abs(position_parametric[~collimated] - self._corner)
        cond1_2 = position_parametric[:, 0] <= self._ad
        cond2_2 = self._a * position_parametric[:, 1] + position_parametric[:, 0] / 4 <= self._ad_2
        collimated_2 = cond1_2 & cond2_2
        
        collimated[~collimated] = collimated_2
        return collimated, self._vacuum


class GammaCamera(TransformableVolumeWithChild):

    def __init__(self, collimator, detector, shielding_thickness=2*cm, glass_backend_thickness=5*cm, name=None, batch_size=1, dtype=torch.float32, device="cpu"):
        detector_box_size = torch.where(collimator.size > detector.size, collimator.size, detector.size)
        detector_box_size[2] = collimator.size[2] + detector.size[2] + glass_backend_thickness
        detector_box = TransformableVolumeWithChild(
            geometry=Box(detector_box_size, device=device, dtype=dtype),
            material=material_database['Air, Dry (near sea level)'],
            name='Detector_box',
            batch_size=batch_size,
            dtype=dtype
        )
        self.device = device
        glass_backend_size = detector_box_size.clone()
        glass_backend_size[2] = glass_backend_thickness
        glass_backend = TransformableVolume(
            geometry=Box(glass_backend_size, device=device, dtype=dtype),
            material=material_database['Glass, Borosilicate (Pyrex)'],
            name='Glass_backend',
            batch_size=batch_size,
            dtype=dtype
        )
        
        super().__init__(
            geometry=Box(torch.tensor([detector_box_size[0] + 2*shielding_thickness, 
                        detector_box_size[1] + 2*shielding_thickness, 
                        detector_box_size[2] + shielding_thickness], device=device), device=device, dtype=dtype),
            material=material_database['Pb'],
            name=name,
            batch_size=batch_size,
            dtype=dtype
        )
        shielding_translation = torch.full((batch_size, 1), shielding_thickness, device=self.device)
        detector_box.translate(z=shielding_translation)
        detector_box.set_parent(self)
        
        collimator_translation = torch.full((batch_size, 1), (detector_box_size[2]/2 - collimator.size[2]/2), device=self.device)
        collimator.translate(z=collimator_translation)
        detector_box.add_child(collimator)
        
        detector_translation = torch.full((batch_size, 1), (detector_box_size[2]/2 - collimator.size[2] - detector.size[2]/2), device=self.device)
        detector.translate(z=detector_translation)
        detector_box.add_child(detector)
        
        glass_translation = torch.full((batch_size, 1),
                                       (detector_box_size[2]/2 - collimator.size[2] - detector.size[2] - glass_backend.size[2]/2),
                                        device=self.device)
        glass_backend.translate(z=glass_translation)
        detector_box.add_child(glass_backend)

    @property
    def detector_box(self):
        return self.childs[0]

    @property
    def collimator(self):
        return self.detector_box.childs[0]

    @property
    def detector(self):
        return self.detector_box.childs[1]

vlen_str = string_dtype(encoding='utf-8')

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class SimulationDataManager:
    """ 
    Manager of data for all interacted particles.
    """

    def __init__(self, filename: str, sensitive_volumes: list[TransformableVolume], angles: Tensor, lock=None, device=None, **kwds):
        self.filename = Path(f'output data/{filename}')
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.sensitive_volumes = sensitive_volumes
        self.lock = lock
        self.count = {0: 0, 1: 0}
        self.angles = angles.to(device) if device is not None else angles
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.angles = self.angles.to(self.device)
        self.save_emission_distribution = True
        self.save_dose_distribution = True
        self.distribution_voxel_size = 4. * mm
        self.clear_interaction_data()
        self.iteraction_buffer_size = int(10**3)
        self._buffered_interaction_number = 0
        self.args = [
            'save_emission_distribution',
            'save_dose_distribution',
            'distribution_voxel_size',
            'iteraction_buffer_size'
        ]
        self.valid_filters = []
        self.min_energy = 1*keV

        for arg in self.args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

    def check_valid(self, particles, volume, batch_indices):
        """
            Mark valid particles for simulation
            Needed for multiple batches procession to not count interactions multiple times
        """
        result = torch.logical_and(particles.valid[0], volume.check_inside(particles.position[0], batch_indices))
        for filter in self.valid_filters:
            result *= filter(particles)
        return result

    def check_progress_in_file(self):
        try:
            file = File(self.filename, 'r')
        except Exception:
            last_time = None
            state = None
        else:
            try:
                last_time = file['Source timer']
                state = None
                last_time = float(np.array(last_time))
            except Exception:
                print(f'\tFailed to restore progress')
                last_time = None
                state = None
            finally:
                print(f'\tProgress restored')
                file.close()
        finally:
            print(f'\tSource timer: {last_time}')
            return last_time, state
        
    def add_interaction_data(self, interaction_data: ParticleInteractionData):
        batch_indices = interaction_data.batch_indices
        for volume in self.sensitive_volumes:
            in_volume_mask = self.check_valid(interaction_data.particle_batch, volume, batch_indices)
            if not in_volume_mask.any():
                continue

            unique_batches = torch.unique(batch_indices)
            self._buffered_interaction_number += in_volume_mask.sum().item()
            for batch_idx in unique_batches:
                batch_idx = batch_idx.item()

                batch_mask = (batch_indices == batch_idx)
                batch_mask = torch.logical_and(batch_mask, in_volume_mask)
                self.count[batch_idx] += batch_mask.sum()
                # print('\n', 'CHECKING TOTAL BATCH ELEMENTS', batch_idx, self.count[batch_idx])
                if not batch_mask.any():
                    continue

                angle_name = str(self.angles[batch_idx].cpu().item())
                volume_name = volume.name
                masked_data = self._process_interaction_data(
                    interaction_data, 
                    batch_mask,
                    volume
                )
                
                self.interaction_data[(volume_name, angle_name)].append(masked_data)

        if self._buffered_interaction_number >= self.iteraction_buffer_size:
            self.save_interaction_data()
            self._buffered_interaction_number = 0

    def flush(self):
        """Force save any remaining buffered data."""
        if self._buffered_interaction_number > 0:
            self.save_interaction_data()


    def _process_interaction_data(self, interaction_data: ParticleInteractionData, mask: Tensor, volume: TransformableVolume):
        particle_batch = interaction_data.particle_batch
        batch_indices = interaction_data.batch_indices[mask].cpu().numpy()

        particle_ID = particle_batch.ID[:, mask][0].cpu().numpy()
        emission_time = particle_batch.emission_time[:, mask][0].cpu().numpy()
        particle_type = particle_batch._type[:, mask][0].cpu().numpy()
        energy_deposit = interaction_data.energy_deposit[:, mask][0].cpu().numpy()
        global_pos  = particle_batch.position[:, mask][0].cpu().numpy()
        global_dir  = particle_batch.direction[:, mask][0].cpu().numpy()
        local_pos   = volume.convert_to_local_position(particle_batch.position[:, mask][0], batch_indices, as_parent=False).cpu().numpy()                    # (N, 3)
        local_dir   = volume.convert_to_local_direction(particle_batch.direction[:, mask][0], batch_indices, as_parent=False).cpu().numpy()                    # (N, 3)
        emission_position = particle_batch.emission_position[:, mask][0].cpu().numpy()
        scattering_angles = interaction_data.scattering_angles[:, :, mask][0].T.cpu().numpy()
        process_name = np.array(interaction_data.process_name)
        distance_traveled = particle_batch.distance_traveled[:, mask][0].cpu().numpy()

        return {
            'global_position': global_pos,
            'global_direction': global_dir,
            'local_position': local_pos,
            'local_direction': local_dir,
            'process_name': process_name,
            'particle_type': particle_type,
            'particle_ID': particle_ID,
            'energy_deposit': energy_deposit,
            'scattering_angles': scattering_angles,
            'emission_time': emission_time,
            'emission_position': emission_position,
            'distance_traveled': distance_traveled,
        }

    def save_interaction_data(self):
        """Thread-safe save operation"""
        if self.lock:
            with self.lock:
                self._save()
        else:
            self._save()
        self.clear_interaction_data()
        self._buffered_interaction_number = 0

    def clear_interaction_data(self):
        self.interaction_data = defaultdict(list)

    def _save(self):
        with File(self.filename, 'a') as f:
            group = f.require_group('Interaction Data')
            for (vol_name, angle), entries in self.interaction_data.items():
                vol_group = group.require_group(vol_name)
                angle_group = vol_group.require_group(str(angle))
                for field, data in self._concatenate_entries(entries).items():
                    if field not in angle_group:
                        maxshape = (None,) + data.shape[1:]
                        dtype = vlen_str if field == 'process_name' else data.dtype
                        angle_group.create_dataset(
                            field,
                            data=data.astype(dtype),
                            maxshape=maxshape,
                            chunks=True,
                            compression="gzip"
                        )
                    else:
                        current = angle_group[field]
                        current.resize(current.shape[0] + data.shape[0], axis=0)
                        current[-data.shape[0]:] = data
        # _logger.error(f"Save failed: {str(e)}")

    def _concatenate_entries(self, entries: list) -> dict:
        concatenated = defaultdict(list)
        for entry in entries:
            for k, v in entry.items():
                concatenated[k].append(v)
        return {k: np.concatenate(v) for k, v in concatenated.items()}

class PropagationWithInteraction:
    """ Class for particle propagation with interaction """
    def __init__(self, processes_list=None, attenuation_database=None, rng=None, device=None, dtype=torch.float32):
        self.processes_list = processes_list
        self.attenuation_database = attenuation_database if attenuation_database is None else attenuation_database
        self.rng = torch.default_generator if rng is None else rng
        self.device = device
        self.processes = [process(rng=rng, attenuation_database=self.attenuation_database, device=device, dtype=dtype) for process in processes_list]

    def __call__(self, particles, volume):
        distance, current_volume = volume.cast_path(particles.position, particles.direction, entry_point=True)
        materials = current_volume.material
        processes_LAC = torch.transpose(self.get_processes_LAC(particles, materials), 1, 2)
        total_LAC = processes_LAC.sum(dim=1)
        u = torch.rand(total_LAC.shape, generator=self.rng, device=self.device)
        free_path = -torch.log(u + 1e-8) / total_LAC
        interacted_mask = free_path < distance
        interacted_batch, interacted_indices = torch.where(interacted_mask)
        distance = torch.where(interacted_mask, free_path, distance)
        particles.move(distance)
        interaction_data = []
        if interacted_batch.numel() > 0:
            total_LAC = total_LAC[interacted_batch, interacted_indices]

            woodcoock_mask = current_volume.type_matching(WoodcockVolume)
            woodcoock_interacted_mask = interacted_mask & woodcoock_mask
            woodcock_batch, woodcock_indices = torch.where(woodcoock_interacted_mask)

            if woodcock_batch.numel() > 0:
                woodcock_particles = particles[woodcock_batch, woodcock_indices]  # returns [1, N, 3]
                woodcock_mat = volume.get_material_by_position(woodcock_particles.position, woodcock_batch)
                materials[woodcock_batch, woodcock_indices] = woodcock_mat
                processes_LAC[woodcock_batch, :, woodcock_indices] = self.get_processes_LAC(woodcock_particles, woodcock_mat).squeeze(0)

            processes_LAC = torch.transpose(processes_LAC[interacted_batch, :, interacted_indices], 0, 1)  # shape batch * indices, 3
            for process, mask in self.choose_process(processes_LAC, total_LAC):
                process_indices = interacted_indices[mask]
                process_batch = interacted_batch[mask]
                processing_particles = particles[process_batch, process_indices]
                processing_materials = materials[process_batch, process_indices]
                
                data = process(processing_particles, processing_materials, process_batch)
                interaction_data.append(data)
                particles.replace_with_new([process_batch, process_indices], processing_particles)

            return ParticleInteractionData.cat(interaction_data)
        return None
    
    def get_processes_LAC(self, particles, materials):
        return torch.stack([process.get_LAC(particles, materials) for process in self.processes], dim=-1)

    def get_total_LAC(self, particles, materials):
        return torch.sum(torch.stack([process.get_LAC(particles, materials) for process in self.processes]), dim=-1)

    def choose_process(self, processes_LAC, total_LAC):
        probabilities = processes_LAC / total_LAC.unsqueeze(0)
        cumulative_probs = torch.cumsum(probabilities, dim=0)
        rnd = torch.rand_like(total_LAC, device=total_LAC.device)
        rnd_expanded = rnd.unsqueeze(0).expand_as(cumulative_probs)

        masks = torch.zeros_like(cumulative_probs, dtype=torch.bool)
        masks[0, :] = rnd_expanded[0, :] <= cumulative_probs[0, :]

        masks[1:, :] = ((rnd_expanded[1:, :] <= cumulative_probs[1:, :]) & (rnd_expanded[1:, :] > cumulative_probs[:-1, :]))
        return [(self.processes[i], masks[i, :]) for i in range(len(self.processes))]

Queue = queue.Queue
Thread = mt.Thread

class SimulationManager(Thread):
    """ Класс менеджера симуляции """

    def __init__(self, source, simulation_volume, propagation_manager=None, stop_time=1*s, particles_number=10**3, queue=None):
        super().__init__()
        self.source = source
        self.simulation_volume = simulation_volume
        self.propagation_manager = PropagationWithInteraction() if propagation_manager is None else propagation_manager
        self.stop_time = stop_time
        self.particles_number = int(particles_number)
        self.valid_filters = []
        self.min_energy = 1*keV
        self.queue = Queue(maxsize=1) if queue is None else queue
        self.step = 1
        self.profile = False
        self.daemon = True
        self.pbar = None
        signal(SIGINT, self.sigint_handler)

    def check_valid(self, particles):
        result = particles.energy > self.min_energy
        result *= self.simulation_volume.check_inside(particles.position)
        for filter in self.valid_filters:
            result *= filter(particles)
        return result

    def sigint_handler(self, signal, frame):
        _logger.error(f'{self.name} interrupted at {datetime_from_seconds(self.source.timer/second)}')
        self.stop_time = 0

    def send_data(self, data):
        self.queue.put(data)

    def next_step(self):
        propagation_data = self.propagation_manager(self.particles, self.simulation_volume)
        invalid_particles = ~self.check_valid(self.particles)

        if propagation_data is not None:
            _logger.debug(f'{self.name} generated {propagation_data.size} events')
            self.send_data(propagation_data)
        self.particles.valid[invalid_particles] = False  # to not count them multiple times in propagation data

        active_batches = self.source.timer <= self.stop_time
        invalid_particles = invalid_particles * active_batches.unsqueeze(1)
        invalid_counts = torch.sum(invalid_particles, dim=1, dtype=torch.int64)
        if torch.any(invalid_counts > 0):
            new_particles = self.source.generate_particles(invalid_counts)
            invalid_batch, invalid_indices = torch.where(invalid_particles)
            self.particles.replace_with_new([invalid_batch, invalid_indices], new_particles)     
        self.step += 1

    def has_active_particles(self):
        """Check if there are any active particles across all batches"""
        valid_particles = self.check_valid(self.particles)
        active_mask = torch.any(valid_particles, dim=1) & (self.source.timer <= self.stop_time)
        return torch.any(active_mask)

    def run(self):
        if self.profile:
            self.run_profile()
        else:
            self._run()

    def run_profile(self):
        runctx('self._run()', globals(), locals(), f'stats/{self.name}.txt')

    def _run(self):
        """ Реализация работы потока частиц """
        _logger.warning(f'{self.name} started from {datetime_from_seconds(self.source.timer.min().item()/second)} to {datetime_from_seconds(self.stop_time/second)}')
        start_timepoint = datetime.now()
        
        self.particles = self.source.generate_particles(self.particles_number)
        total_time_to_simulate = (self.stop_time - self.source.timer.min().item()) / second
        
        self.pbar = tqdm(
            total=total_time_to_simulate,
            initial=0.0,
            desc=f"Source exhaustion",
            unit="s",
            bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]'
        )
        
        prev_min_timer = self.source.timer.min().item()
        timer_reached_stop = False

        while self.has_active_particles():
            self.next_step()
            if not timer_reached_stop:
                progress_change = (self.source.timer.min().item() - prev_min_timer) / second
                if progress_change > 0:
                    current = self.pbar.n + progress_change
                    if current > total_time_to_simulate:
                        progress_change = total_time_to_simulate - self.pbar.n
                    if progress_change > 0:
                        self.pbar.update(progress_change)
                        self.pbar.refresh()
                    prev_min_timer = self.source.timer.min().item()
                if self.source.timer.min().item() >= self.stop_time:
                    timer_reached_stop = True
                    self.pbar.close()

            _logger.debug(f'Source timer of {self.name} at {datetime_from_seconds(self.source.timer.min().item()/second)}')
        
        self.queue.put('stop')
        stop_timepoint = datetime.now()
        _logger.warning(f'{self.name} finished at {datetime_from_seconds(self.source.timer.min().item()/second)}')
        _logger.info(f'The simulation of {self.name} took {stop_timepoint - start_timepoint}')


def prepare_config(configs):
    """Convert list of individual configs to batched configuration"""
    batched_config = {
        'filename': configs[0]['filename'],
        'time_interval': configs[0]['time_interval'],
        'angles': torch.tensor([c['angle'] for c in configs]),
        'radius': configs[0]['radius'],
        'delta_angles': torch.tensor([c['delta_angle'] for c in configs]),
        'gamma_cameras': configs[0]['gamma_cameras'],
        'seed': configs[0]['seed'],
        'particles_number': configs[0]['particles_number']
    }
    return batched_config

def modeling(config, device="cpu", dtype=torch.float32):
    # Set up logging
    log_path = Path(f'logs/{config["filename"]}/batch.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )
    logger = logging.getLogger()

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Extract batched parameters
    batch_size = len(config['angles'])
    particles_number = config['particles_number']
    angles = config['angles'].unsqueeze(1).to(device)
    radii = config['radius']
    delta_angles = config['delta_angles'].to(device)
    start_time, stop_time = config['time_interval']
    rng = torch.Generator(device=device).manual_seed(config['seed'])

    # Setup simulation volume
    simulation_volume = VolumeWithChilds(
        geometry=Box(torch.tensor([120*cm, 120*cm, 80*cm]), device=device, dtype=dtype),
        material=material_database['Air, Dry (near sea level)'],
        name='Simulation_volume',
        dtype=dtype,
        input_shape=[batch_size, particles_number]
    )

    # Load phantom material distribution
    material_ID_distribution = torch.from_numpy(np.load('phantoms/material_map.npy')).to(device)
    material_distribution = MaterialArray(material_ID_distribution.shape, device=device)
    material_distribution[material_ID_distribution == 0] = material_database['Air, Dry (near sea level)']
    material_distribution[material_ID_distribution == 1] = material_database['Lung']
    material_distribution[material_ID_distribution == 2] = material_database['Adipose Tissue (ICRU-44)']
    material_distribution[material_ID_distribution == 3] = material_database['Tissue, Soft (ICRU-44)']
    material_distribution[material_ID_distribution == 4] = material_database['Bone, Cortical (ICRU-44)']

    phantom = WoodcockVoxelVolume(
        voxel_size=4*mm,
        material_distribution=material_distribution,
        name='Phantom',
        batch_size=batch_size,
        dtype=dtype,
        device=device
    )
    phantom.set_parent(simulation_volume)


    detector_list = []

    for i in range(config['gamma_cameras']):
        detector = TransformableVolume(
            geometry=Box(torch.tensor([54.*cm, 40*cm, 0.95*cm]), device=device, dtype=dtype),
            material=material_database['Sodium Iodide'],
            batch_size=batch_size,
            dtype=dtype,
            name=f'Detector at angles starting at {round((angles[0] + delta_angle*i).item()/degree, 1)} deg'
        )

        collimator = ParametricParallelCollimator(
            size=[detector.size[0], detector.size[1], 3.5*cm],
            hole_diameter=1.5*mm,
            batch_size=batch_size,
            device=device,
            septa=0.2*mm,
            material=material_database['Pb'],
            name=f'Collimator at angles starting at {round((angles[0] + delta_angle*i).item()/degree, 1)} deg',
            dtype=dtype
        )
        print(collimator.name)

        spect_head = GammaCamera(
            collimator=collimator,
            detector=detector,
            shielding_thickness=2*cm,
            glass_backend_thickness=7.6*cm,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name=f'Gamma_camera at angles starting at {round((angles[0] + delta_angle*i).item()/degree, 1)} deg'
        )
        spect_head.rotate(gamma=torch.full((batch_size, 1), torch.pi/2, device=device))
        spect_translation = torch.full((batch_size, 1), radius + spect_head.size[2]/2, device=device)
        spect_head.translate(y=spect_translation)
        spect_head.rotate(alpha=angles + delta_angle*i)
    
        simulation_volume.add_child(spect_head)
        detector_list.append(detector)

    distribution = torch.from_numpy(np.load('phantoms/source_function.npy')).to(device)
    distribution[distribution == 40] = 10
    distribution[distribution == 30] = 20
    distribution[distribution == 70] = 40
    distribution[distribution == 80] = 40
    distribution[distribution == 89] = 50
    distribution[distribution == 140] = 40
    distribution[distribution == 1200] = 1000
    distribution[distribution == 700] = 550
    distribution[distribution == 10000] = 7000

    source = Тс99m_MIBI(
        distribution=distribution,
        activity=300*MBq,
        voxel_size=4*mm,
        batch_size=batch_size,
        device=device
    )
    source.rng = rng
    source.set_state(start_time)

    propagation_manager = PropagationWithInteraction(
        attenuation_database=attenuation_database,
        rng=source.rng,
        device=device,
        dtype=dtype,
        processes_list=processes_list
    )

    simulation_manager = SimulationManager(
        source=source,
        simulation_volume=simulation_volume,
        propagation_manager=propagation_manager,
        particles_number=particles_number,
        stop_time=stop_time
    )
    simulation_manager.name = f'{round(angles[0].item()/degree, 1)} deg'
    simulation_manager.start()

    simulation_data_manager = SimulationDataManager(
        filename=f'{config["filename"]}/batch_results.hdf',
        sensitive_volumes=detector_list,
        angles=angles.squeeze(1),
        iteraction_buffer_size=10**7
    )

    while True:
        data = simulation_manager.queue.get()
        if isinstance(data, ParticleInteractionData):
            simulation_data_manager.add_interaction_data(data)
        elif data == 'stop':
            simulation_manager.join()
            simulation_data_manager.flush()
            break
        else:
            raise ValueError("Invalid data received from simulation")

if __name__ == '__main__':
    filename = 'heart_32_main'
    radius = 233 * mm
    views = 2
    gamma_cameras_per_view = 1
    endpoint=True
    angle_start = 0.0
    angle_stop = np.pi
    # change the distance_epsilon to match your precision or else no progress
    # programm forced to use float32 at some places so it would be a mixed precision
    dtype = torch.float16
    num_gantry_stops = views // gamma_cameras_per_view
    angles = np.linspace(angle_start, angle_stop, num_gantry_stops, endpoint=endpoint)
    # angle_step = angles[1] - angles[0]
    delta_angle = pi
    time_interval = (0 * millisecond, 100 * millisecond)
    seed = 42
    particles_number = 1000000

    configs = []
    for i in range(views // gamma_cameras_per_view):
        configs.append({
            'filename': filename,
            'time_interval': time_interval,
            'angle': angles[i],
            'delta_angle': delta_angle,
            'gamma_cameras': gamma_cameras_per_view,
            'seed': seed,
            'radius': radius,
            'particles_number': particles_number
        })
    print(f"Number of configurations (gantry stops): {len(configs)}")
    print(f"Total number of views/projections: {len(configs) * gamma_cameras_per_view}")
    print(f"Number of gamma cameras per stop: {gamma_cameras_per_view}")
    print(f"Delta angle between cameras at a stop: {delta_angle / np.pi:.2f} * pi ({np.degrees(delta_angle):.1f} degrees)")
    print(f"First few base angles for stops (degrees): {[np.degrees(a) for a in angles[:min(5, len(angles))]]}")
    batched_config = prepare_config(configs)
    torch.set_default_dtype(dtype)

    modeling(batched_config, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)