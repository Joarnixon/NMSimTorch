import logging
from pathlib import Path
import torch
from torch import Tensor
from h5py import File, string_dtype
import numpy as np
from core.particles.particles import ParticleInteractionData
from core.geometry.volumes import TransformableVolume
from collections import defaultdict
from hepunits import *
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
        batch_indices = interaction_data.batch_indices[mask].cpu()

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