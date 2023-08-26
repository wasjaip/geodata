## Copyright 2022 Michael Davidson (UCSD), Xiqiang Liu (UCSD)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
DATASET_ROOT_PATH = Path(os.environ.get("GETDATA_ROOT", Path.home() / ".local" / "geodata")).resolve()


gebco_path = DATASET_ROOT_PATH / "gebco"
cutout_dir = DATASET_ROOT_PATH / "cutouts"
era5_dir = DATASET_ROOT_PATH / "era5"
merra2_dir = DATASET_ROOT_PATH / "merra2"
MASK_DIR = DATASET_ROOT_PATH / "masks"

# Check if these paths exists
for path in [gebco_path, cutout_dir, era5_dir, merra2_dir, MASK_DIR]:
    if not path.is_dir():
        path.mkdir(exist_ok=True, parents=True)
        logger.info("Dataset storage location %s does not exists, creating.", path)

gebco_path = str(gebco_path)
cutout_dir = str(cutout_dir)
era5_dir = str(era5_dir)
merra2_dir = str(merra2_dir)
MASK_DIR = str(MASK_DIR)

# weather_dataset = {'module': 'cordex', 'model': 'MPI-M-MPI-ESM-LR'}
weather_dataset = {"module": "era5"}
untrimmable_datasets = {"era5"}
