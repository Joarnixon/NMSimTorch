import torch
import torch.nn.functional as F
from core.particles.particles import ParticleBatch
from hepunits import *
import core.other.utils as utils
from typing import Union


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
        
        self.distribution = torch.as_tensor(distribution, device=self.device)
        self.distribution /= torch.sum(self.distribution)
        self.initial_activity = torch.sum(distribution) if activity is None else torch.as_tensor(activity, device=self.device)
        self.initial_activity = self.initial_activity.to(torch.float64)
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
        translation_matrix = utils.compute_translation_matrix(translation)
        if in_local:
            self.transformation_matrix = self.transformation_matrix @ translation_matrix
        else:
            self.transformation_matrix = translation_matrix @ self.transformation_matrix

    def rotate(self, alpha=0., beta=0., gamma=0., rotation_center=[0., 0., 0.], in_local=False):
        """Rotate source. Not tested"""
        rotation_angles = torch.tensor([alpha, beta, gamma], device=self.device)
        rotation_center = torch.tensor(rotation_center, device=self.device)
        rotation_matrix = utils.compute_translation_matrix(rotation_center)
        rotation_matrix = rotation_matrix @ utils.compute_rotation_matrix(rotation_angles)
        rotation_matrix = rotation_matrix @ utils.compute_translation_matrix(-rotation_center)
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
        dt = torch.zeros_like(self.timer)
        
        log_2 = torch.log(torch.tensor(2.0, device=self.device, dtype=torch.float64))
        dt = torch.log((self.nuclei_number + n_per_batch) / self.nuclei_number) * self.half_life / log_2
        a = 2**(-self.timer/self.half_life)
        b = 2**(-(self.timer + dt)/self.half_life)
        
        total = n_per_batch.sum().item()

        a_expanded = torch.repeat_interleave(a, n_per_batch)
        b_expanded = torch.repeat_interleave(b, n_per_batch)
        
        alpha = torch.rand(total, device=self.device, generator=self.rng) * (a_expanded - b_expanded) + b_expanded
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
            n = torch.tensor([n], device=self.device).expand(batch_size)
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