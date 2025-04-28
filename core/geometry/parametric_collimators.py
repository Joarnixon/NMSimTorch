import torch
from core.geometry.woodcoock_volumes import WoodcockParameticVolume
import settings.database_setting as settings
from core.geometry.geometries import Box


class ParametricParallelCollimator(WoodcockParameticVolume):
    """
    Torch-optimized class for parametric collimator with parallel channels
    """

    def __init__(self, size, hole_diameter, septa, material=None, name=None, batch_size=1, dtype=torch.float32, device="cpu"):
        material = settings.material_database['Pb'] if material is None else material
        super().__init__(
            geometry=Box(torch.tensor(size, device=device), device=device, dtype=dtype),
            material=material,
            name=name,
            batch_size=batch_size,
            dtype=dtype
        )
        self._hole_diameter = torch.tensor(hole_diameter, device=device)
        self._septa = torch.tensor(septa, device=device)
        self._vacuum = settings.material_database['Vacuum']
        self.device = device
        self._compute_constants()
    
    def _compute_constants(self):
        x_period = self._hole_diameter + self._septa
        y_period = torch.sqrt(torch.tensor(3.0, device=self.device)) * x_period
        self._period = torch.stack((x_period, y_period))
        self._a = torch.sqrt(torch.tensor(3.0, device=self.device)) / 4
        d = self._hole_diameter * 2 / torch.sqrt(torch.tensor(3.0, device=self.device))
        self._corner = self._period / 2
        self._ad = self._a * d
        self._ad_2 = self._ad / 2

    def _parametric_function(self, position):    
        position_parametric = position[:, :2]
        position_parametric = torch.remainder(position_parametric, self._period)
        position_parametric = torch.abs(position_parametric - self._corner)
        cond1 = position_parametric[:, 0] <= self._ad
        cond2 = self._a * position_parametric[:, 1] + position_parametric[:, 0] / 4 <= self._ad_2
        collimated = cond1 & cond2
        position_parametric = torch.abs(position_parametric[~collimated] - self._corner)
        cond1_2 = position_parametric[:, 0] <= self._ad
        cond2_2 = self._a * position_parametric[:, 1] + position_parametric[:, 0] / 4 <= self._ad_2
        collimated_2 = cond1_2 & cond2_2
        
        collimated[~collimated] = collimated_2
        return collimated, self._vacuum