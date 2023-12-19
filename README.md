# Geodata

[![DOI](https://zenodo.org/badge/218690319.svg)](https://zenodo.org/badge/latestdoi/218690319)

**Geodata** - это библиотека на языке Python для сбора и предварительного анализа геопространственных данных. Данная библиотека предоставляет инструменты для работы с геопространственными и растровыми наборами данных физических переменных. Эти данные широко распространены и имеют все более высокое разрешение. Геоданные позволяют упростить сбор и использование геопространственных наборов данных путем создания общих скриптов для "готовых к анализу" физических переменных.

## Особенности

- **Простота использования:** Легко идентифицировать, загружать и работать с новыми источниками геопространственных данных.
- **Совместимость:** Базируется на библиотеке **[atlite](https://github.com/PyPSA/atlite)** для преобразования метеоданных в данные систем энергосистем.
- **Графическое представление:** Иллюстрация процесса работы библиотеки:

  ![График рабочего процесса Geodata](images/geodata_workflow_chart.png)




## Installation

**Geodata** has been tested to run with python3 (>= 3.9). Read the [package setup instructions](doc/general/packagesetup.md) to configure and install the package.
Installation will also install the following dependencies:
* `numpy`
* `scipy`
* `pandas`
* `bottleneck`
* `numexpr`
* `xarray`
* `netcdf4`
* `dask`
* `boto3`
* `toolz`
* `pyproj`
* `requests`
* `matplotlib`
* `rasterio`
* `rioxarray`
* `shapely`
* `progressbar2`

## Documentation

Read the [Introduction to Geodata](doc/general/Introduction.md) documentation to get started. 

Read the [Table of Contents](doc/general/tableofcontents.md) to navigate through the documentation. 

You may also jump directly to [Example Notebooks](example_notebooks).



## Contributing

We welcome suggestions for feature enhancements and the identification of bugs. Please make an issue or contact the [authors](https://mdavidson.org/about/) of geodata.


## License

Geodata is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 (2007). This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](/LICENSE.txt) for more details.

## Support

The Geodata team would like to thank the Center for Global Transformation at UC San Diego for providing financial support to the project.




