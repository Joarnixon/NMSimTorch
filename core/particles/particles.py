import torch
from torch import Tensor
from abc import ABC

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
    
    