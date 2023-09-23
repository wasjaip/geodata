# Copyright 2023 Michael Davidson (UCSD), Xiqiang Liu (UCSD)

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from typing import Optional
from scipy.interpolate import CubicSpline

import numpy as np
import pandas as pd
import xarray as xr

from ...logging import logger
from ._base import HEIGHTS, WindBaseModel

try:
    from numba import njit, prange
except ImportError:
    logger.warning("Numba not installed. Using pure Python implementation.")
    prange = range

    from ...utils import dummy_njit as njit


@njit(parallel=True)
def _compute_wind_speed_int(
    heights: np.ndarray, speeds: np.ndarray, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wind speed from heights and speeds.

    Args:
        heights (np.ndarray): Array of heights.
        speeds (np.ndarray): Array of speeds.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of coefficients (alpha, beta) and residuals.
    """
    # pylint: disable=not-an-iterable
    # n_coeffs = 4
    # coeffs = np.empty(shape=heights.shape[:-1] + (heights.shape[-1] - 1, n_coeffs))

    # for lon in prange(heights.shape[0]):
    #     for lat in prange(heights.shape[1]):
    #         for time in prange(heights.shape[2]):
    #             h = heights[time, lat, lon]
    #             v = speeds[time, lat, lon]
    #             cs = CubicSpline(h, v)
    #             coeffs[time, lat, lon] = cs.c.T
                
    # return coeffs
    cs = CubicSpline(heights, speeds, axis=1)
    interpolated_wind_speed = cs(height)
    
    return interpolated_wind_speed


class WindInterpolationModel(WindBaseModel):
    """Wind speed estimation based on the an extrapolation model.

    Model Details:
        The wind speed is estimated using a linear regression of the logarithm of the wind speed
        against the logarithm of the height. The coefficients of the regression are stored in the
        dataset as a 2D array of shape (2,). The first coefficient is the slope of the regression
        line, and the second coefficient is the intercept. The wind speed is then estimated as
        :math:`\\hat{v} = \\alpha \\cdot \\log(z) + \\beta`, where :math:`\\alpha` and :math:`\\beta`
        are the coefficients, and :math:`z` is the height.

    Residuals:
        The residuals of the regression are stored in the dataset as a 2D array of shape (n,).
        The residuals are the difference between the estimated wind speed and the actual wind speed.
        The residuals are stored in the same order as the heights used in the regression.

    Example:
        >>> from geodata import Dataset
        >>> from geodata.model.wind import WindExtrapolationModel
        >>> dataset = Dataset(module="merra2", weather_data_config="slv_flux_hourly", years=slice(2010, 2010), months=slice(1,2))
        >>> model = WindExtrapolationModel(dataset)
        >>> model.prepare()
        >>> model.estimate(height=12, xs=slice(1, 2), ys=slice(1, 2), years=slice(2010, 2010), months=slice(1, 2))
    """

    def _prepare_fn(
        self,
        ds: xr.Dataset,
        half_precision: bool = True,
    ) -> xr.Dataset:
        """Compute wind speed from for MERRA2 dataset.

        Args:
            ds (xr.Dataset): MERRA2 dataset.
            compute_lml (bool, optional): Include speed at LML in calculation. Defaults to True.
            half_precision (bool, optional): Use float32 precision to store coefficients and residuals. Defaults to True.
            compute_residuals (bool, optional): Compute and store residuals against existing data points. Defaults to True.

        Returns:
            xr.Dataset: Dataset with wind speed.
        """
        return ds
        variables = [f for f in HEIGHTS if f in ds and f.replace("u", "v") in ds]
        heights = np.array([HEIGHTS[f] for f in variables])

        logger.debug("Selected variables: %s", variables)
        logger.debug("Shape of heights: %s", heights.shape)
        # Basic sanity check
        if 0 in heights.shape:
            raise ValueError(
                "Dataset does not contain any other useable heights other than lml"
            )

        speeds = []
        for var in variables:
            speeds.append(
                ((ds[var] ** 2 + ds[var.replace("u", "v")] ** 2) ** 0.5).values
            )
        speeds = np.stack(speeds, axis=-1).astype(heights.dtype)

        coeffs = _compute_wind_speed_int(heights, speeds)

        if half_precision:
            coeffs = coeffs.astype("float32")
        ds = ds.assign_coords(coeff=["coeffs"])
        ds["coeffs"] = (("time", "lat", "lon", "coeff"), coeffs)

        return ds[["coeffs"]]

    # pylint: disable=arguments-differ
    def _estimate_dataset(
        self,
        height: int,
        years: slice,
        months: Optional[slice] = None,
        xs: Optional[slice] = None,
        ys: Optional[slice] = None,
        use_real_data: Optional[bool] = False,
    ) -> xr.Dataset:
        assert height > 0, "Height must be greater than 0."

        if months is None:
            months = slice(1, 12)

        start_time = pd.Timestamp(year=years.start, month=months.start, day=1)
        end_time = pd.Timestamp(
            year=years.stop, month=months.stop, day=31, hour=23, minute=59, second=59
        )
        print(self.files)
        ds = xr.open_mfdataset(self.files)
        print(1)
        if xs is None:
            xs = ds.coords["lon"]
        if ys is None:
            ys = ds.coords["lat"]
        return

        ds = ds.sel(lon=xs, lat=ys, time=slice(start_time, end_time))

        if height in HEIGHTS.values() and use_real_data:
            logger.info("Using real data for estimation at height %d", height)
            return (ds[f"u{height}m"] ** 2 + ds[f"v{height}m"] ** 2) ** 0.5

        u_var = [f for f in HEIGHTS if f in ds and f.replace("u", "v") in ds]
        v_var = [u.replace("u", "v") for u in u_var]
        print(u_var, v_var)
        return
        variables = [f for f in HEIGHTS if f in ds and f.replace("u", "v") in ds]
        heights = np.array([HEIGHTS[f] for f in variables])
        coeffs = ds["coeffs"]
        result = np.empty_like(ds['time'])
        for time in prange(result.shape[0]):
            for lat in prange(result.shape[1]):
                for lon in prange(result.shape[2]):
                    cs = CubicSpline(heights[time, lat, lon], coeffs[time, lat, lon])
                    result[time, lat, lon] = cs(height)
                    
        return result
        # return result.drop_vars("coeff")  # remove unnecessary coordinate

    def _estimate_cutout(
        self,
        height: int,
        years: slice,
        months: Optional[slice] = None,
        xs: Optional[slice] = None,
        ys: Optional[slice] = None,
        use_real_data: Optional[bool] = False,
    ) -> xr.Dataset:
        assert height > 0, "Height must be greater than 0."

        if months is None:
            months = slice(1, 12)

        start_time = pd.Timestamp(year=years.start, month=months.start, day=1)
        end_time = pd.Timestamp(
            year=years.stop, month=months.stop, day=31, hour=23, minute=59, second=59
        )

        ds = xr.open_mfdataset(self.files)

        if xs is None:
            xs = ds.coords["lon"]
        if ys is None:
            ys = ds.coords["lat"]

        ds = ds.sel(x=xs, y=ys, time=slice(start_time, end_time))

        if height in HEIGHTS.values() and use_real_data:
            logger.info("Using real data for estimation at height %d", height)
            return (ds[f"u{height}m"] ** 2 + ds[f"v{height}m"] ** 2) ** 0.5

        alpha = ds["coeffs"][..., 0]
        beta = ds["coeffs"][..., 1]

        exp = np.exp(-beta / alpha)
        disph = ds["disph"]
        mask1 = exp > 0
        mask2 = disph > 0

        h = np.where(mask1, 50, height) # decreasing wind speed over heights
        h = np.where(mask2, 50, height) # displacement height is greater than 0
        
        result = alpha * np.log((h - disph) / exp)

        return result.drop_vars("coeff")  # remove unnecessary coordinate


    # pylint: enable=arguments-differ