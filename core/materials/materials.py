import torch
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from core.other.nonunique_array import NonuniqueArray
from core.materials.atomic_properties import atomic_number
import numpy as np
from hepunits import*


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
