import settings.database_setting as database_setting
import settings.processes_settings as processes_settings
from core.geometry.woodcoock_volumes import WoodcockVolume
from core.particles.particles import ParticleInteractionData
from hepunits import*
import torch
from torch import inf
import time

class PropagationWithInteraction:
    """ Class for particle propagation with interaction """
    def __init__(self, processes_list=None, attenuation_database=None, rng=None, device=None, dtype=torch.float32):
        processes_list = processes_settings.processes_list if processes_list is None else processes_list
        self.attenuation_database = database_setting.attenuation_database if attenuation_database is None else attenuation_database
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
        