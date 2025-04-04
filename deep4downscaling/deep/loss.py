"""
This module contains loss functions for training deep learning
downscaling models.

Author: Jose González-Abad
"""

import os
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import xarray as xr
import scipy.stats
from typing import Union

class MaeLoss(nn.Module):

    """
    Standard Mean Absolute Error (MAE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output)
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MaeLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean(torch.abs(target - output))
        return loss

class MseLoss(nn.Module):

    """
    Standard Mean Square Error (MSE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output)
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MseLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean((target - output) ** 2)
        return loss

class NLLGaussianLoss(nn.Module):

    """
    Negative Log-Likelihood of a Gaussian distribution. It is possible to compute
    this metric over a target dataset with nans.

    Notes
    -----
    This loss function needs as input two values, corresponding to the mean and
    the logarithm of the variance. THese must be provided concatenated as an
    unique vector.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted mean and logarithm of the
        variance.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(NLLGaussianLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        dim_target = target.shape[1]

        mean = output[:, :dim_target]
        log_var = output[:, dim_target:]
        precision = torch.exp(-log_var)

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            mean = mean[~nans_idx]
            log_var = log_var[~nans_idx]
            precision = precision[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean(0.5 * precision * (target-mean)**2 + 0.5 * log_var)
        return loss

class NLLBerGammaLoss(nn.Module):

    """
    Negative Log-Likelihood of a Bernoulli-gamma distributions. It is possible to compute
    this metric over a target dataset with nans.

    Notes
    -----
    This loss function needs as input three values, corresponding to the p, shape
    and scale parameters. THese must be provided concatenated as an unique vector.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted p, shape and scale.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(NLLBerGammaLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        dim_target = target.shape[1]

        p = output[:, :dim_target]
        shape = torch.exp(output[:, dim_target:(dim_target*2)])
        scale = torch.exp(output[:, (dim_target*2):])

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            p = p[~nans_idx]
            shape = shape[~nans_idx]
            scale = scale[~nans_idx]
            target = target[~nans_idx]

        bool_rain = torch.greater(target, 0).type(torch.float32)
        epsilon = 0.000001

        noRainCase = (1 - bool_rain) * torch.log(1 - p + epsilon)
        rainCase = bool_rain * (torch.log(p + epsilon) +
                            (shape - 1) * torch.log(target + epsilon) -
                            shape * torch.log(scale + epsilon) -
                            torch.lgamma(shape + epsilon) -
                            target / (scale + epsilon))
        
        loss = -torch.mean(noRainCase + rainCase)
        return loss

class Asym(nn.Module):

    """
    Generalization of the asymmetric loss function tailored for daily precipitation developed in
    Doury et al. 2024. It is possible to compute this metric over a target dataset
    with nans.

    Doury, A., Somot, S. & Gadat, S. On the suitability of a convolutional neural
    network based RCM-emulator for fine spatio-temporal precipitation. Clim Dyn (2024).
    https://doi.org/10.1007/s00382-024-07350-8

    Notes
    -----
    This loss function relies on gamma distribution fitted for each gridpoint in the
    spatial domain. This class provides all the methods require to fit these 
    distributions to the data.
    The level of asymmetry can be adjusted by the pairs (asym_weight/cdf_pow) or (weight_list/perc_list), but not both.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    asym_weight : positive float, optional
        Weight for the asymmetric term at the loss function relative to the MAE term.
        Default value: 1 (as in Doury et al., 2024)

    cdf_pow : float, optional
        Pow for the CDF at the asymmetric term of the loss function.
        Default value: 2 (as in Doury et al., 2024)
        Higher values make a bigger differentiation between the weight for high/low percentiles

    weight_list : list of positive floats, optional
        Weights for each range of percentiles at the asymmetric term of the loss function.
        By default, the asymmetric term is defined by asym_weight and cdf_pow. In order to use weight_list and
        perc_list instead, define asym_weight and cdf_pow as None.

    perc_list : list of floats between 0-100, optional
        List of percentiles to apply the weight_list at the asymmetric term of the loss function.
        By default, the asymmetric term is defined by asym_weight and cdf_pow. In order to use weight_list and
        perc_list instead, define asym_weight and cdf_pow as None.
        From 0 to the firs percentile defined, the weight will be 0. From the first percentile to the next one, the
        weight will be the first element of weight_list, and so on.

    asym_path : str
        Path to the folder to save the fitted distributions.

    appendix : str, optional
        String to add to the files generated/loaded for this loss function.
        (e.g., appendix=test1 -> scale_test1.npy). If not provided no appendix
        will be added.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted mean and logarithm of the
        variance.
    """

    def __init__(self, ignore_nans: bool, asym_path: str,
                 asym_weight: float = 1.0, cdf_pow: float = 2.0,
                 weight_list: list = None, perc_list: list = None,
                 appendix: str = None) -> None:
        super(Asym, self).__init__()

        # Check if asym_weight and cdf_pow are both None or both provided, and the same for weight_list/perc_list,
        # and also that at least one pair is provided
        if (asym_weight is not None and cdf_pow is None) or (asym_weight is None and cdf_pow is not None):
            raise ValueError("Both 'asym_weight' and 'cdf_pow' must be either provided together or omitted together.")
        if (weight_list is not None and perc_list is None) or (weight_list is None and perc_list is not None):
            raise ValueError("Both 'weight_list' and 'perc_list' must be either provided together or omitted together.")
        if not (asym_weight is not None and cdf_pow is not None) or not (
                weight_list is not None and perc_list is not None):
            raise ValueError("At least on pair 'asym_weight'/'cdf_pow' or 'weight_list'/'perc_list'  must be provided")

        # If asym_weight or cdf_pow are provided, ensure that weight_list and perc_list are None and asym_weight is
        # positive
        if asym_weight is not None or cdf_pow is not None:
            if weight_list is not None or perc_list is not None:
                raise ValueError(
                    "If 'perc_list' or 'weight_list' are provided, 'cdf_pow' and 'asym_weight' must be None.")
            if asym_weight < 0:
                raise ValueError("'asym_weight' must be positive.")

        # If weight_list or perc_list are provided, ensure that asym_weight and cdf_pow are None, perc_list and
        # weight_list are of the same length, all weights in weight_list are positive and all percentiles in perc_list
        # are between 0 and 100
        if weight_list is not None or perc_list is not None:
            if asym_weight is not None or cdf_pow is not None:
                raise ValueError(
                    "If 'weight_list' or 'perc_list' are provided, 'asym_weight' and 'cdf_pow' must be None.")
            if len(weight_list) != len(perc_list):
                raise ValueError("'weight_list' and 'perc_list' must have the same length.")
            if any(w < 0 for w in weight_list):
                raise ValueError("All elements in 'weight_list' must be positive.")
            if any(p < 0 or p > 100 for p in perc_list):
                raise ValueError("All elements in 'perc_list' must be between 0 and 100.")

        self.ignore_nans = ignore_nans
        self.asym_path = asym_path
        self.asym_weight = asym_weight
        self.cdf_pow = cdf_pow
        self.weight_list = weight_list
        self.perc_list = perc_list
        self.appendix = appendix

    def parameters_exist(self):

        """
        Check for the existence of the gamma distributions
        """

        if self.appendix:
            shape_file_name = f'shape_{self.appendix}.npy'
            scale_file_name = f'scale_{self.appendix}.npy'
            loc_file_name = f'loc_{self.appendix}.npy'
        else:
            shape_file_name = 'shape.npy'
            scale_file_name = 'scale.npy'
            loc_file_name = 'loc.npy'

        shape_exist = os.path.exists(f'{self.asym_path}/{shape_file_name}')
        scale_exist = os.path.exists(f'{self.asym_path}/{scale_file_name}')
        loc_exist = os.path.exists(f'{self.asym_path}/{loc_file_name}')

        return (shape_exist and scale_exist and loc_exist)

    def load_parameters(self):

        """
        Load the gamma distributions from asym_path.
        """

        if self.appendix:
            shape_file_name = f'shape_{self.appendix}.npy'
            scale_file_name = f'scale_{self.appendix}.npy'
            loc_file_name = f'loc_{self.appendix}.npy'
        else:
            shape_file_name = 'shape.npy'
            scale_file_name = 'scale.npy'
            loc_file_name = 'loc.npy'

        self.shape = np.load(f'{self.asym_path}/{shape_file_name}')
        self.scale = np.load(f'{self.asym_path}/{scale_file_name}')
        self.loc = np.load(f'{self.asym_path}/{loc_file_name}')

    def _compute_gamma_parameters(self, x: np.ndarray) -> tuple:

        """
        Fit a gamma distribution to the wet days of the provided
        1D np.ndarray.

        Parameters
        ----------      
        x : np.ndarray
            1D np.ndarray containing the precipitation values across time
            for a specific gridpoint.

        Returns
        -------
        tuple
        The shape, loc and scale parameters of the fitted gamma
        distribution.
        """

        # If nan return nan
        if np.sum(np.isnan(x)) == len(x):
            return np.nan, np.nan, np.nan
        else:
            x = x[~np.isnan(x)] # Remove nans
            x = x[x >= 1] # Filter wet days
            try: # Compute dist.
                fit_shape, fit_loc, fit_scale = scipy.stats.gamma.fit(x)
            except: # If its not possible return nan
                fit_shape, fit_loc, fit_scale = np.nan, np.nan, np.nan 
            return fit_shape, fit_loc, fit_scale

    def compute_parameters(self, data: xr.Dataset, var_target: str):

        """
        Iterate over the xr.Dataset and compute for each spatial gridpoint
        the parameters of a fitted gamma distribution for the wet days.

        Parameters
        ----------      
        data : xr.Dataset
            Dataset containing the variable used as target in the model. It is
            important to provide it in the same way as it will be provided
            as target to the forward() method (e.g., nan-filtered).

        var_target : str
            Target variable.
        """

        # Get years
        gamma_params = []
        group_years = data.groupby('time.year')

        # Iterate over years
        for year, group in group_years:
            print(f'Year: {year}')
            y_year = group[var_target].values
            params_year = np.apply_along_axis(self._compute_gamma_parameters,
                                              axis=0, arr=y_year) # shape, loc, scale
            gamma_params.append(params_year)

        # Compute yearly mean
        gamma_params = np.nanmean(np.stack(gamma_params), axis=0)
        
        self.shape = gamma_params[0, :]
        self.scale = gamma_params[2, :]
        self.loc = gamma_params[1, :]

        # Save the parameters in the asym_path
        if self.appendix:
            shape_file_name = f'shape_{self.appendix}.npy'
            scale_file_name = f'scale_{self.appendix}.npy'
            loc_file_name = f'loc_{self.appendix}.npy'
        else:
            shape_file_name = 'shape.npy'
            scale_file_name = 'scale.npy'
            loc_file_name = 'loc.npy'

        np.save(file=f'{self.asym_path}/{shape_file_name}',
                arr=self.shape)
        np.save(file=f'{self.asym_path}/{scale_file_name}',
                arr=self.scale)
        np.save(file=f'{self.asym_path}/{loc_file_name}',
                arr=self.loc)

    def mask_parameters(self, mask: xr.Dataset):

        """
        Mask the shape, scale and loc parameters. This is required for
        models composed of a final fully-connected layer.

        Parameters
        ----------
        mask : xr.Dataset
            Mask without time dimension containing 1s for non-nan and
            0 for nans.
        """

        mask_var = list(mask.keys())[0]
        mask_dims = list(mask.dims.keys())

        # If gridpoint dimension does not exist, create it
        if 'gridpoint' not in mask_dims:
            mask = mask.stack(gridpoint=('lat', 'lon'))

        # We assume that 1 -> non-nan and 0 -> nan
        mask_values = mask[mask_var].values
        mask_ones = np.where(mask_values == 1)

        self.shape = self.shape[mask_ones]
        self.scale = self.scale[mask_ones]
        self.loc = self.loc[mask_ones]

    def prepare_parameters(self, device: str):

        """
        Move the gamma parameters to device and remove nans. The latter is key,
        as if the gamma has not been fitted to the data (nans), it means that
        there are not (few) rainy days. In this case, we set a weight of one
        for any underestimation (which will be uncommon).

        Parameters
        ----------
        device : str
            Device used to run the training (cuda or cpu)
        """

        self.shape = torch.tensor(self.shape).to(device)
        self.scale = torch.tensor(self.scale).to(device)
        self.loc = torch.tensor(self.loc).to(device)

        epsilon = 0.0000001
        if torch.isnan(self.shape).any():
            self.shape[torch.isnan(self.shape)] = epsilon
        if torch.isnan(self.scale).any():
            self.scale[torch.isnan(self.scale)] = epsilon
        if torch.isnan(self.loc).any():
            self.loc[torch.isnan(self.loc)] = 0

    def compute_cdf(self, data: torch.Tensor) -> torch.Tensor:
    
        """
        Compute the value of the cumulative distribution function (CDF) for
        the data.

        Parameters
        ----------      
        data : torch.Tensor
            Data (from the target dataset) to compute the CDF for.
        """

        # Compute cdfs for Torch
        if isinstance(data, torch.Tensor):
            data = data - self.loc # For scipy, loc corresponds to the mean
            data[data < 0] = 0 # Remove the negative values, which are automatically handled by scipy
            m = td.Gamma(concentration=self.shape,
                         rate=1/self.scale,
                         validate_args=False) # Deactivates the validation of the paremeters (e.g., support)
                                              # In this way the cdf method handles nans
            cdfs = m.cdf(data)

        # Compute cdfs for Numpy
        elif isinstance(data, np.ndarray):
            cdfs = np.empty_like(data)
            cdfs = scipy.stats.gamma.cdf(data,
                                         a=self.shape, scale=self.scale, loc=self.loc)

        else:
            raise ValueError('Unsupported type for the data argument.')

        return cdfs

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        """
        Compute the loss function for the target and output data
        """

        cdfs = self.compute_cdf(data=target)
        cdfs = torch.nan_to_num(cdfs, nan=0.0)

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]
            cdfs = cdfs[~nans_idx]

        # Using asym_weight and cdf_pow
        if self.asym_weight is not None:
            loss_mae = torch.mean(torch.abs(target - output))
            loss_asym = torch.mean((cdfs ** self.cdf_pow) * torch.max(torch.tensor(0.0), target - output))
            loss = loss_mae + self.asym_weight * loss_asym

        # Using weight_list and perc_list
        else:
            loss = []
            for i in range(len(self.weight_list)):
                w_i, p_min, p_max = self.weight_list[i], self.perc_list[i] / 100, 1
                if len(self.weight_list) > i + 1:
                    p_max = self.perc_list[i + 1] / 100
                cdf_i = torch.clone(cdfs)
                cdf_i[cdf_i < p_min] = 0
                cdf_i[cdf_i > p_max] = 0
                cdf_i[cdf_i > 0] = 1
                loss_i = torch.mean(
                    torch.abs(target - output) + (cdf_i * w_i) * torch.max(torch.tensor(0.0), target - output))
                loss.append(loss_i)
            loss = torch.mean(torch.stack(loss))

        return loss
