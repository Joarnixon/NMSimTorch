from core.geometry.volumes import TransformableVolume
import torch
import time

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