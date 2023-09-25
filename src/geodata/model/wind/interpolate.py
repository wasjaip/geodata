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

import numpy as np
import pandas as pd
import xarray as xr

from ...logging import logger
from ._base import HEIGHTS, LEVEL_TO_HEIGHT, WindBaseModel

try:
    from numba import njit, prange
except ImportError:
    logger.warning("Numba not installed. Using pure Python implementation.")
    prange = range

    from ...utils import dummy_njit as njit


@njit(parallel=True)
def _compute_wind_speed_int(
    heights: np.ndarray, speeds: np.ndarray
):
    """Compute wind speed from heights and speeds.

    Args:
        heights (np.ndarray): Array of heights.
        speeds (np.ndarray): Array of speeds.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of coefficients (alpha, beta) and residuals.
    """
    # pylint: disable=not-an-iterable
    # time: level: latitude: longitude
    # speed.values[0, :, 20, 20] time lat lon
    n = len(heights)
    coef = np.zeros(speeds.shape)

    for time in prange(speeds.shape[0]):
        for lat in prange(speeds.shape[2]):
            for lon in prange(speeds.shape[3]):
                for i in range(n):
                    term = speeds[time, i, lat, lon]
                    for j in range(n):
                        if j != i:
                            term = term / (heights[i] - heights[j])
                    coef[time, i, lat, lon] = term
    return coef

@njit(parallel=True)
def _estimate_wind_speed_int(
    coeffs: np.ndarray, height: float, shape: tuple, n: int, level_heights: np.ndarray
):
    # pylint: disable=not-an-iterable
    result = np.zeros((shape[0], shape[2], shape[3]), dtype=np.float32)

    for time in prange(shape[0]):
        for lat in prange(shape[2]):
            for lon in prange(shape[3]):
                for i in range(n):
                    term = coeffs[time, i, lat, lon]
                    for j in range(n):
                        if j != i:
                            term = term * (height - level_heights[j])
                    result[time, lat, lon] += term[0]

    return result

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
        variables = [var for var in ds if var.replace("u", "v") in ds and var == 'u']
        heights = np.array(list(LEVEL_TO_HEIGHT.values()))

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

        ds = ds.assign_coords(coefficient=["coefficient"])
        ds["coeffs"] = (("time", "level", "latitude", "longitude", "coefficient"), coeffs)

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

        ds = xr.open_mfdataset(self.files)

        if xs is None:
            xs = ds.coords["longitude"]
        if ys is None:
            ys = ds.coords["latitude"]

        ds = ds.sel(longitude=xs, latitude=ys, time=slice(start_time, end_time))

        if height in HEIGHTS.values() and use_real_data:
            logger.info("Using real data for estimation at height %d", height)
            return (ds[f"u{height}m"] ** 2 + ds[f"v{height}m"] ** 2) ** 0.5

        coef = ds["coeffs"].values.astype(np.float32)
        shape = ds["u"].shape
        n = len(LEVEL_TO_HEIGHT)
        height = np.float32(height)
        level_heights = np.array(list(LEVEL_TO_HEIGHT.values())).astype(np.float32)

        result = _estimate_wind_speed_int(coef, height, shape, n=n, level_heights=level_heights)
        time = ds.coords["time"].values
        lat = ds.coords["latitude"].values
        lon = ds.coords["longitude"].values
        result = xr.DataArray(
            result,
            dims = ["time", "latitude", "longitude"],
            coords=dict(
                longitude=lon,
                latitude=lat,
                time=time
            )
        )

        return result

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
            xs = ds.coords["longitude"]
        if ys is None:
            ys = ds.coords["latitude"]

        ds = ds.sel(longitude=xs, latitude=ys, time=slice(start_time, end_time))

        if height in HEIGHTS.values() and use_real_data:
            logger.info("Using real data for estimation at height %d", height)
            return (ds[f"u{height}m"] ** 2 + ds[f"v{height}m"] ** 2) ** 0.5

        coef = ds["coeffs"].values.astype(np.float32)
        shape = ds["u"].shape
        n = len(LEVEL_TO_HEIGHT)
        height = np.float32(height)
        level_heights = np.array(list(LEVEL_TO_HEIGHT.values())).astype(np.float32)

        result = _estimate_wind_speed_int(coef, height, shape, n=n, level_heights=level_heights)
        time = ds.coords["time"].values
        lat = ds.coords["latitude"].values
        lon = ds.coords["longitude"].values
        result = xr.DataArray(
            result,
            dims = ["time", "latitude", "longitude"],
            coords=dict(
                longitude=lon,
                latitude=lat,
                time=time
            )
        )

        return result

    # pylint: enable=arguments-differ
