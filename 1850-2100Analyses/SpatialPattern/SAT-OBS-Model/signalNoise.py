# signal-to-nosie ratio for SAT trend pattern
# 1. calculate the spatial pattern of SAT
# 2. calculate the spatial pattern of SAT-OBS
# 3. calculate the spatial pattern of SAT-Model
# 4. calculate the signal-to-noise ratio for SAT-Model

# Signal to noise defined as the ensemble mean of the trend pattern divided 
#           by the ensemble standard deviation of the trend pattern

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# define function


# import data
data_hist_ssp245_MPI_ESM  = '/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/MergeDataOut/tas_Amon_1850-2022_*.nc'
ds = xr.open_mfdataset(data_hist_ssp245_MPI_ESM, combine = 'nested', concat_dim = 'run')
print(ds)

