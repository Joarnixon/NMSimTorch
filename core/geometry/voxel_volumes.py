from functools import cache
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
from core.geometry.geometries import Box
import torch

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