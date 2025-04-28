import torch
import core.physics.g4compton as g4compton
import core.physics.g4coherent as g4coherent
from core.materials.attenuation_functions import AttenuationFunction
from core.particles.particles import ParticleInteractionData
import settings.database_setting as settings
from abc import ABC
from typing import TypeVar
from hepunits import *
import time

class Process(ABC):
    """ Класс процесса """

    def __init__(self, attenuation_database=None, device=None, rng=None, dtype=torch.float32):
        """ Конструктор процесса """
        attenuation_database = settings.attenuation_database if attenuation_database is None else attenuation_database
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
            self.theta_generator = g4coherent.initialize_cpu(self.rng)
        else:
            self.theta_generator = g4coherent.initialize_cuda(self.rng, device=device)

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
            self.theta_generator = g4compton.initialize_cpu(self.rng)
        else:
            self.theta_generator = g4compton.initialize_cuda(self.rng, device=device)

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
