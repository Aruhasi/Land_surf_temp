import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.gridspec as gridspec

ERA5 = xr.open_dataset('/work/mh0033/m301036/LSAT/Data/ERA5/mslp.mon.mean.1.0.nc')
print(ERA5)

ds = ERA5.sel(time=slice('1979-01-01', '2019-12-01'))
ds_clima = ds.groupby('time.month').mean('time')

#calculate anomalies
ERA5_ano = ERA5.groupby('month') - ERA5.groupby('month').mean('time')
print(ERA5_ano)

