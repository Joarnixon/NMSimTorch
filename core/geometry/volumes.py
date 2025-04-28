import torch
from itertools import count
from copy import deepcopy
from core.materials.materials import MaterialArray
from core.other.nonunique_array import NonuniqueArray
import core.other.utils as utils
import time


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

    def _convert_to_local_position(position: torch.Tensor, batch_indices: torch.Tensor, total_transformation_matrix: torch.Tensor) -> torch.Tensor:
        if len(batch_indices) == 0:
            return position
        total_transformation_matrix = torch.transpose(total_transformation_matrix, 1, 2)
        position_ones = torch.cat([position, torch.ones(position.shape[0], 1, device=position.device)], dim=-1)
        local_positions = torch.matmul(position_ones.unsqueeze(1), total_transformation_matrix[batch_indices]).squeeze(1)
        return local_positions[:, :3]

    def _convert_to_local_direction(direction: torch.Tensor, batch_indices: torch.Tensor, total_transformation_matrix: torch.Tensor) -> torch.Tensor:
        if len(batch_indices) == 0:
            return direction
        total_transformation_matrix = torch.transpose(total_transformation_matrix[:, :3, :3], 1, 2)
        return torch.matmul(direction.unsqueeze(1), total_transformation_matrix[batch_indices]).squeeze(1)

    def convert_to_local_position(self, position, batch_indices, as_parent=True):
        total_transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        return TransformableVolume._convert_to_local_position(position, batch_indices, total_transformation_matrix)

    def convert_to_local_direction(self, direction, batch_indices, as_parent=True):
        total_transformation_matrix = self.transformation_matrix if as_parent else self.total_transformation_matrix
        return TransformableVolume._convert_to_local_direction(direction, batch_indices, total_transformation_matrix)
    

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
        translation_matrix = utils.compute_translation_matrix(-translation)
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
        rotation_matrix = utils.compute_translation_matrix(-rotation_center)
        rotation_matrix = torch.bmm(rotation_matrix, utils.compute_rotation_matrix(-rotation_angles))
        rotation_matrix = torch.bmm(rotation_matrix, utils.compute_translation_matrix(rotation_center))
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

