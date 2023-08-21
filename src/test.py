from geodata import Dataset
from geodata.model.wind import WindExtrapolationModel
dataset = Dataset(
    module="merra2",
    weather_data_config="slv_flux_hourly",
    years=slice(2010, 2010),
    months=slice(1,1)
)
dataset.trim_variables()
model = WindExtrapolationModel(dataset)
model.prepare()
model.estimate(xs=slice(1, 1), ys=slice(1, 1), years=slice(2010, 2010), months=slice(1, 1), height=1)