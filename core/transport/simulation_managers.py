from cProfile import runctx
from datetime import datetime
import logging
from signal import SIGINT, signal
from hepunits import*
import torch
from core.other.utils import datetime_from_seconds
from core.transport.propagation_managers import PropagationWithInteraction
import threading as mt
import queue
import time
from tqdm import tqdm

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

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
        invalid_counts = torch.sum(invalid_particles, dim=1)
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