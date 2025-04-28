import os
import torch
import numpy as np
from hepunits import *
import logging
from pathlib import Path
from core.materials.materials import MaterialArray
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.geometries import Box
from core.geometry.parametric_collimators import ParametricParallelCollimator
from core.geometry.volumes import TransformableVolume, VolumeWithChilds
from core.transport.simulation_managers import SimulationManager
from core.data.data_manager import SimulationDataManager
from core.geometry.voxel_volumes import WoodcockVoxelVolume
from core.transport.propagation_managers import PropagationWithInteraction
from core.particles.particles import ParticleInteractionData
from core.source.sources import Тс99m_MIBI
from settings.database_setting import material_database, attenuation_database


def prepare_config(configs):
    """Convert list of individual configs to batched configuration"""
    batched_config = {
        'filename': configs[0]['filename'],
        'time_interval': configs[0]['time_interval'],
        'angles': torch.tensor([c['angle'] for c in configs]),
        'radius': configs[0]['radius'],
        'delta_angles': torch.tensor([c['delta_angle'] for c in configs]),
        'gamma_cameras': configs[0]['gamma_cameras'],
        'seed': configs[0]['seed'],
        'particles_number': configs[0]['particles_number']
    }
    return batched_config

def modeling(config, device="cpu", dtype=torch.float32):
    # Set up logging
    log_path = Path(f'logs/{config["filename"]}/batch.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='[%(asctime)s: %(levelname)s] %(message)s'
    )
    logger = logging.getLogger()

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Extract batched parameters
    batch_size = len(config['angles'])
    particles_number = config['particles_number']
    angles = config['angles'].unsqueeze(1).to(device)
    radii = config['radius']
    delta_angles = config['delta_angles'].to(device)
    start_time, stop_time = config['time_interval']
    rng = torch.Generator(device=device).manual_seed(config['seed'])

    # Setup simulation volume
    simulation_volume = VolumeWithChilds(
        geometry=Box(torch.tensor([120*cm, 120*cm, 80*cm]), device=device, dtype=dtype),
        material=material_database['Air, Dry (near sea level)'],
        name='Simulation_volume',
        dtype=dtype,
        input_shape=[batch_size, particles_number]
    )

    # Load phantom material distribution
    material_ID_distribution = torch.from_numpy(np.load('phantoms/material_map.npy')).to(device)
    material_distribution = MaterialArray(material_ID_distribution.shape, device=device)
    material_distribution[material_ID_distribution == 0] = material_database['Air, Dry (near sea level)']
    material_distribution[material_ID_distribution == 1] = material_database['Lung']
    material_distribution[material_ID_distribution == 2] = material_database['Adipose Tissue (ICRU-44)']
    material_distribution[material_ID_distribution == 3] = material_database['Tissue, Soft (ICRU-44)']
    material_distribution[material_ID_distribution == 4] = material_database['Bone, Cortical (ICRU-44)']

    phantom = WoodcockVoxelVolume(
        voxel_size=4*mm,
        material_distribution=material_distribution,
        name='Phantom',
        batch_size=batch_size,
        dtype=dtype,
        device=device
    )
    phantom.set_parent(simulation_volume)


    detector_list = []

    for i in range(config['gamma_cameras']):
        detector = TransformableVolume(
            geometry=Box(torch.tensor([54.*cm, 40*cm, 0.95*cm]), device=device, dtype=dtype),
            material=material_database['Sodium Iodide'],
            batch_size=batch_size,
            dtype=dtype,
            name=f'Detector at angles starting at {round((angles[0] + delta_angle*i).item()/degree, 1)} deg'
        )

        collimator = ParametricParallelCollimator(
            size=[detector.size[0], detector.size[1], 3.5*cm],
            hole_diameter=1.5*mm,
            batch_size=batch_size,
            device=device,
            septa=0.2*mm,
            material=material_database['Pb'],
            name=f'Collimator at angles starting at {round((angles[0] + delta_angle*i).item()/degree, 1)} deg',
            dtype=dtype
        )
        print(collimator.name)

        spect_head = GammaCamera(
            collimator=collimator,
            detector=detector,
            shielding_thickness=2*cm,
            glass_backend_thickness=7.6*cm,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            name=f'Gamma_camera at angles starting at {round((angles[0] + delta_angle*i).item()/degree, 1)} deg'
        )
        spect_head.rotate(gamma=torch.full((batch_size, 1), torch.pi/2, device=device))
        spect_translation = torch.full((batch_size, 1), radius + spect_head.size[2]/2, device=device)
        spect_head.translate(y=spect_translation)
        spect_head.rotate(alpha=angles + delta_angle*i)
    
        simulation_volume.add_child(spect_head)
        detector_list.append(detector)

    distribution = torch.from_numpy(np.load('phantoms/source_function.npy')).to(device)
    distribution[distribution == 40] = 10
    distribution[distribution == 30] = 20
    distribution[distribution == 70] = 40
    distribution[distribution == 80] = 40
    distribution[distribution == 89] = 50
    distribution[distribution == 140] = 40
    distribution[distribution == 1200] = 1000
    distribution[distribution == 700] = 550
    distribution[distribution == 10000] = 7000

    source = Тс99m_MIBI(
        distribution=distribution,
        activity=300*MBq,
        voxel_size=4*mm,
        batch_size=batch_size,
        device=device
    )
    source.rng = rng
    source.set_state(start_time)

    propagation_manager = PropagationWithInteraction(
        attenuation_database=attenuation_database,
        rng=source.rng,
        device=device,
        dtype=dtype
    )

    simulation_manager = SimulationManager(
        source=source,
        simulation_volume=simulation_volume,
        propagation_manager=propagation_manager,
        particles_number=particles_number,
        stop_time=stop_time
    )
    simulation_manager.name = f'{round(angles[0].item()/degree, 1)} deg'
    simulation_manager.start()

    simulation_data_manager = SimulationDataManager(
        filename=f'{config["filename"]}/batch_results.hdf',
        sensitive_volumes=detector_list,
        angles=angles.squeeze(1),
        iteraction_buffer_size=10**7
    )

    while True:
        data = simulation_manager.queue.get()
        if isinstance(data, ParticleInteractionData):
            simulation_data_manager.add_interaction_data(data)
        elif data == 'stop':
            simulation_manager.join()
            simulation_data_manager.flush()
            break
        else:
            raise ValueError("Invalid data received from simulation")

if __name__ == '__main__':
    filename = 'heart_32_main'
    radius = 233 * mm
    views = 2
    gamma_cameras_per_view = 1
    endpoint=True
    angle_start = 0.0
    angle_stop = np.pi
    dtype = torch.float32  # change Geometry.distance_epsilon if seeing bugs
    num_gantry_stops = views // gamma_cameras_per_view
    angles = np.linspace(angle_start, angle_stop, num_gantry_stops, endpoint=endpoint)
    delta_angle = pi
    time_interval = (0 * millisecond, 10 * millisecond)
    seed = 42
    particles_number = 100000

    configs = []
    for i in range(views // gamma_cameras_per_view):
        configs.append({
            'filename': filename,
            'time_interval': time_interval,
            'angle': angles[i],
            'delta_angle': delta_angle,
            'gamma_cameras': gamma_cameras_per_view,
            'seed': seed,
            'radius': radius,
            'particles_number': particles_number
        })
    print(f"Number of configurations (gantry stops): {len(configs)}")
    print(f"Total number of views/projections: {len(configs) * gamma_cameras_per_view}")
    print(f"Number of gamma cameras per stop: {gamma_cameras_per_view}")
    print(f"First few base angles (degrees): {[np.degrees(a) for a in angles[:min(5, len(angles))]]}")
    batched_config = prepare_config(configs)
    torch.set_default_dtype(dtype)


    modeling(batched_config, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)