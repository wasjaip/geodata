from geodata import Dataset
from geodata.model.wind import WindExtrapolationModel
dataset = Dataset(
    module="merra2",
    weather_data_config="slv_flux_hourly",
    years=slice(2010, 2010),
    months=slice(1,1)
)
model = WindExtrapolationModel(dataset)
model.prepare()
result = model.estimate(
    height=12,
    xs=slice(1, 1),
    ys=slice(1, 1),
    years=slice(2010, 2010),
    months=slice(1, 1)
)
print(result)