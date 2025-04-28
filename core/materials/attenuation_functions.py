import torch
import numpy as np
from functools import partial
import contextlib
import time

class Interp1d():
    @staticmethod
    def apply(x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            v[name] = vec
            is_flat[name] = v[name].shape[0] == 1

        device = x.device

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        # the squeeze is because torch.searchsorted does accept either a nd with
        # matching shapes for x and xnew or a 1d vector for x. Here we would
        # have (1,len) for x sometimes 
        torch.searchsorted(v['x'].contiguous().squeeze(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        is_flat['slopes'] = is_flat['x']
        
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)
        return ynew


class AttenuationFunction(dict):
    """Attenuation function class"""

    def __init__(self, process, attenuation_database, device):
        self.__class__.__name__ = self.__class__.__name__ + 'Of' + process.name
        self.__class__.__qualname__ = self.__class__.__qualname__ + 'Of' + process.name
        dtype = torch.get_default_dtype()
        if dtype == torch.float16:
            dtype = torch.float32  # density is out of range of float16, not recommended.
        self.dtype = dtype

        for material, attenuation_data in attenuation_database.items():
            energy = torch.from_numpy(np.copy(attenuation_data['Energy'])).to(dtype).to(device)
            attenuation_coefficient = torch.tensor(
                attenuation_data['Coefficient'][process.name] * material.density, 
                dtype=dtype,
                device=device
            )

            lower_limit = torch.searchsorted(energy, process.energy_range[0].clone(), side='left')
            upper_limit = torch.searchsorted(energy, process.energy_range[1].clone(), side='right')
            
            
            energy = energy[lower_limit:upper_limit]
            attenuation_coefficient = attenuation_coefficient[lower_limit:upper_limit]
            attenuation_function = partial(
                Interp1d.apply,
                energy.unsqueeze(0),
                attenuation_coefficient.unsqueeze(0)
            )
            
            self.update({material.name: attenuation_function})
    
    def __call__(self, material, energy):
        """Get linear attenuation coefficient"""
        mass_coefficient = torch.zeros_like(energy, dtype=self.dtype)        
        for material_type, mask in material.inverse_indices.items():
            if torch.any(mask):
                mass_coefficient = mass_coefficient.masked_scatter(mask, self[material_type.name](torch.masked_select(energy, mask)))
        return mass_coefficient