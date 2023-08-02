#!/usr/bin/env python
# coding: utf-8

# In[1]:
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import linregress

def calc_trend(data):
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(data)), data)  
    return slope, p_value

# ## MPI-ESM-Trend Calculating
# In[8]:
data_hist_ssp245_MPI_ESM  = '/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/MergeDataOut/tas_Amon_1850-2100_*.nc'
ds = xr.open_mfdataset(data_hist_ssp245_MPI_ESM, combine = 'nested', concat_dim = 'run')
ds

# In[9]:
#Perfrom land sea mask
land_sea_mask=xr.open_dataset('/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/GR15_lsm_regrid.nc')

# In[10]:
mask_data = land_sea_mask['var1']
mask_data
# Align the time coordinates between the mask dataset and the original dataset
time_values = pd.to_datetime(mask_data['time'].values, format='mixed', dayfirst=True)
mask_data['time'] = time_values

# Align the time coordinates between the mask dataset and the original dataset
mask_data = mask_data.reindex(time=ds['time'], method='nearest')
# Apply the land-sea mask to the original dataset
masked_tas = ds.where(mask_data == 1, drop=False)
masked_tas
# In[11]
tas = masked_tas['tas'].loc[:,'2023-01-01':'2100-12-31',0:90,:]
tas = tas - 273.15
print(tas.min().values)
tas_climatology = tas.groupby('time.month').mean(dim='time')
tas_climatology
tas_ano = tas.groupby('time.month') - tas_climatology
tas_ano_ds = tas_ano.to_dataset()
tas_ano_ds

# In[12]:
seasons = ['JJA', 'DJF', 'MAM', 'SON']
season_means = {}

for season in seasons:
    if season == 'JJA':
        months = [6,7,8]
    elif season == 'DJF':
        months =[12,1,2]
    elif season == 'MAM':
        months = [3,4,5]
    elif season == 'SON':
        months = [9,10,11]

    season_months = tas_ano_ds.sel(time=tas.time.dt.month.isin(months))
    
    # Calculate the seasonal mean SAT anomalies
    season_mean_anomalies = season_months.groupby('time.year').mean('time')
    
    # Store the seasonal mean in the dictionary
    season_means[season] = season_mean_anomalies['tas']

# Access the multiyear JJA mean SAT anomalies
    
tas_ano_ds['JJA'] = season_means['JJA']
tas_ano_ds['DJF'] = season_means['DJF']
tas_ano_ds['MAM'] = season_means['MAM']
tas_ano_ds['SON'] = season_means['SON']

tas_ano_ds
# In[13]:
import scipy.stats as stats
from scipy.stats import linregress

def calc_trend(data):
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(data)), data)  
    return slope, p_value
# In[14]:
tas_ano_ds['slope_JJA'], tas_ano_ds['p_value_JJA'] = xr.apply_ufunc(calc_trend, tas_ano_ds['JJA'].chunk(dict(run=-1)), input_core_dims=[['year']], output_core_dims=[[],[]], vectorize=True, dask='parallelized', output_dtypes=[float,float], dask_gufunc_kwargs={'allow_rechunk': True})
tas_ano_ds['slope_JJA'].attrs['units'] = '°C/65yrs'
tas_ano_ds['p_value_JJA'].attrs['units'] = 'p_value'

# In[15]:
tas_ano_ds['slope_DJF'], tas_ano_ds['p_value_DJF'] = xr.apply_ufunc(calc_trend, tas_ano_ds['DJF'].chunk(dict(run=-1, year=-1)), input_core_dims=[['year']], output_core_dims=[[],[]], vectorize=True, dask='parallelized', output_dtypes=[float,float], dask_gufunc_kwargs={'allow_rechunk': True})
tas_ano_ds['slope_DJF'].attrs['units'] = '°C/65yrs'
tas_ano_ds['p_value_DJF'].attrs['units'] = 'p_value'

# In[16]:
tas_ano_ds['slope_MAM'], tas_ano_ds['p_value_MAM'] = xr.apply_ufunc(calc_trend, tas_ano_ds['MAM'].chunk(dict(run=-1, year=-1)), input_core_dims=[['year']], output_core_dims=[[],[]], vectorize=True, dask='parallelized', output_dtypes=[float,float], dask_gufunc_kwargs={'allow_rechunk': True})
tas_ano_ds['slope_MAM'].attrs['units'] = '°C/65yrs'
tas_ano_ds['p_value_MAM'].attrs['units'] = 'p_value'

# In[17]:
tas_ano_ds['slope_SON'], tas_ano_ds['p_value_SON'] = xr.apply_ufunc(calc_trend, tas_ano_ds['SON'].chunk(dict(run=-1, year=-1)), input_core_dims=[['year']], output_core_dims=[[],[]], vectorize=True, dask='parallelized', output_dtypes=[float,float], dask_gufunc_kwargs={'allow_rechunk': True})
tas_ano_ds['slope_SON'].attrs['units'] = '°C/65yrs'
tas_ano_ds['p_value_SON'].attrs['units'] = 'p_value'

# In[18]:
tas_ano_ds = tas_ano_ds.compute()

# In[19]:
tas_ano_ds

# In[20]:
model_65yr_MAM = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/MPI-ESM-LR_NH_SATAs_2023_65yr_MAM_trend.txt',delimiter='\t', skip_header=1)
model_65yr_JJA = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/MPI-ESM-LR_NH_SATAs_2023_65yr_JJA_trend.txt',delimiter='\t', skip_header=1)
model_65yr_SON = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/MPI-ESM-LR_NH_SATAs_2023_65yr_SON_trend.txt',delimiter='\t', skip_header=1)
model_65yr_DJF = np.genfromtxt(fname='/home/m/m301036/josie/LSAT/1850-2100Analyses/MPI-ESM-LR_NH_SATAs_2023_65yr_DJF_trend.txt',delimiter='\t', skip_header=1)

DJF_65yr = xr.DataArray(model_65yr_DJF[:,1], dims=['run'], coords={'run': np.arange(1, 31, 1)})
MAM_65yr = xr.DataArray(model_65yr_MAM[:,1], dims=['run'], coords={'run': np.arange(1, 31, 1)})
SON_65yr = xr.DataArray(model_65yr_SON[:,1], dims=['run'], coords={'run': np.arange(1, 31, 1)})
JJA_65yr = xr.DataArray(model_65yr_JJA[:,1], dims=['run'], coords={'run': np.arange(1, 31, 1)})

# assume you have an xarray called 'xr_data'
# get the most minimum five values and their indices
min_indices = np.argpartition(DJF_65yr.values.flatten(), 5)[:5]
min_values = DJF_65yr.values.flatten()[min_indices]
min_coords = np.unravel_index(min_indices, DJF_65yr.shape)

# create a new xarray to store the most minimum values
min_xr = xr.DataArray(min_values, dims=['value'], coords={'value': np.arange(5)})

# print the new xarray
print(min_xr)
print(min_coords)

# In[21]:
max_indices = np.argpartition(DJF_65yr.values.flatten(), -5)[-5:]
max_values = DJF_65yr.values.flatten()[max_indices]
max_coords = np.unravel_index(max_indices, DJF_65yr.shape)

max_xr = xr.DataArray(max_values, dims=['value'], coords={'value': np.arange(5)})
print(max_xr)
print(max_coords)
type(max_coords)
max_coords


# In[22]:
max_JJA_indices = np.argpartition(JJA_65yr.values.flatten(), -5)[-5:]
max_JJA_values = JJA_65yr.values.flatten()[max_JJA_indices]
max_JJA_coords = np.unravel_index(max_JJA_indices, JJA_65yr.shape)

max_JJA_xr = xr.DataArray(max_JJA_values, dims=['value'], coords={'value': np.arange(5)})
print(max_JJA_xr)
print(max_JJA_coords)
type(max_JJA_coords)
max_JJA_coords

# In[23]:
min_JJA_indices = np.argpartition(JJA_65yr.values.flatten(), 5)[:5]
min_JJA_values = JJA_65yr.values.flatten()[min_JJA_indices]
min_JJA_coords = np.unravel_index(min_JJA_indices, JJA_65yr.shape)

min_JJA_xr = xr.DataArray(min_JJA_values, dims=['value'], coords={'value': np.arange(5)})
print(min_JJA_xr)
print(min_JJA_coords)
type(min_JJA_coords)
min_JJA_coords

# In[24]:
max_MAM_indices = np.argpartition(MAM_65yr.values.flatten(), -5)[-5:]
max_MAM_values = MAM_65yr.values.flatten()[max_MAM_indices]
max_MAM_coords = np.unravel_index(max_MAM_indices, MAM_65yr.shape)

max_MAM_xr = xr.DataArray(max_MAM_values, dims=['value'], coords={'value': np.arange(5)})
print(max_MAM_xr)
print(max_MAM_coords)
type(max_MAM_coords)
max_MAM_coords
# In[25]:
min_MAM_indices = np.argpartition(MAM_65yr.values.flatten(), 5)[:5]
min_MAM_values = MAM_65yr.values.flatten()[min_MAM_indices]
min_MAM_coords = np.unravel_index(min_MAM_indices, MAM_65yr.shape)

min_MAM_xr = xr.DataArray(min_MAM_values, dims=['value'], coords={'value': np.arange(5)})
print(min_MAM_xr)
print(min_MAM_coords)
type(min_MAM_coords)
min_MAM_coords
# In[26]:
max_SON_indices = np.argpartition(SON_65yr.values.flatten(), -5)[-5:]
max_SON_values = SON_65yr.values.flatten()[max_SON_indices]
max_SON_coords = np.unravel_index(max_SON_indices, SON_65yr.shape)

max_SON_xr = xr.DataArray(max_SON_values, dims=['value'], coords={'value': np.arange(5)})
print(max_SON_xr)
print(max_SON_coords)
type(max_SON_coords)
max_SON_coords

# In[27]:
min_SON_indices = np.argpartition(SON_65yr.values.flatten(), 5)[:5]
min_SON_values = SON_65yr.values.flatten()[min_SON_indices]
min_SON_coords = np.unravel_index(min_SON_indices, SON_65yr.shape)

min_SON_xr = xr.DataArray(min_SON_values, dims=['value'], coords={'value': np.arange(5)})
print(min_SON_xr)
print(min_SON_coords)
type(min_SON_coords)
min_SON_coords
# ## Plot the trend pattern of the DJF 65-yr Spatial pattern

# In[28]:
slope_JJA = tas_ano_ds['slope_JJA'] 
slope_DJF = tas_ano_ds['slope_DJF']
slope_MAM = tas_ano_ds['slope_MAM']
slope_SON = tas_ano_ds['slope_SON']

p_value_JJA = tas_ano_ds['p_value_JJA']
p_value_DJF = tas_ano_ds['p_value_DJF']
p_value_MAM = tas_ano_ds['p_value_MAM']
p_value_SON = tas_ano_ds['p_value_SON']

# In[29]:
slope_JJA_data = slope_JJA*65.0
slope_DJF_data = slope_DJF*65.0
slope_MAM_data = slope_MAM*65.0
slope_SON_data = slope_SON*65.0

# In[30]:
slope_DJF_data

# In[31]:

slope_JJA_MME = slope_JJA_data.mean(dim='run')
slope_DJF_MME = slope_DJF_data.mean(dim='run')
slope_MAM_MME = slope_MAM_data.mean(dim='run')
slope_SON_MME = slope_SON_data.mean(dim='run')

# In[32]:

#using the for loop to pick up the min5 and max5 data of DJF 

# Extract trend spatial data for minimum five runs in DJF
DJF_min5_trend = []
for i in range(5):
    run_index = min_coords[0][i]
    DJF_min5_trend.append(slope_DJF_data[run_index,:,:])
DJF_min5_trend = xr.concat(DJF_min5_trend, dim='run')

# Extract trend spatial data for maximum five runs in DJF
DJF_max5_trend = []
for i in range(5):
    run_index = max_coords[0][i]
    DJF_max5_trend.append(slope_DJF_data[run_index,:,:])
DJF_max5_trend = xr.concat(DJF_max5_trend, dim='run')

# Extract trend spatial data for minimum five runs in JJA
JJA_min5_trend = []
for i in range(5):
    run_index = min_JJA_coords[0][i]
    JJA_min5_trend.append(slope_JJA_data[run_index,:,:])
JJA_min5_trend = xr.concat(JJA_min5_trend, dim='run')

# Extract trend spatial data for maximum five runs in JJA
JJA_max5_trend = []
for i in range(5):
    run_index = max_JJA_coords[0][i]
    JJA_max5_trend.append(slope_JJA_data[run_index,:,:])
JJA_max5_trend = xr.concat(JJA_max5_trend, dim='run')

# Extract trend spatial data for minimum five runs in MAM
MAM_min5_trend = []
for i in range(5):
    run_index = min_MAM_coords[0][i]
    MAM_min5_trend.append(slope_MAM_data[run_index,:,:])
MAM_min5_trend = xr.concat(MAM_min5_trend, dim='run')

# Extract trend spatial data for maximum five runs in MAM
MAM_max5_trend = []
for i in range(5):
    run_index = max_MAM_coords[0][i]
    MAM_max5_trend.append(slope_MAM_data[run_index,:,:])
MAM_max5_trend = xr.concat(MAM_max5_trend, dim='run')

# Extract trend spatial data for minimum five runs in SON
SON_min5_trend = []
for i in range(5):
    run_index = min_SON_coords[0][i]
    SON_min5_trend.append(slope_SON_data[run_index,:,:])
SON_min5_trend = xr.concat(SON_min5_trend, dim='run')

# Extract trend spatial data for maximum five runs in SON
SON_max5_trend = []
for i in range(5):
    run_index = max_SON_coords[0][i]
    SON_max5_trend.append(slope_SON_data[run_index,:,:])
SON_max5_trend = xr.concat(SON_max5_trend, dim='run')

# In[33]:
JJA_max5_trend

# In[34]:
# Calculate the DJF min 5 trend mean
DJF_min5_trend_mean = DJF_min5_trend.mean(dim='run')
DJF_max5_trend_mean = DJF_max5_trend.mean(dim='run')

# Calculate the JJA min 5 trend mean
JJA_min5_trend_mean = JJA_min5_trend.mean(dim='run')
JJA_max5_trend_mean = JJA_max5_trend.mean(dim='run')

# Calculate the MAM min 5 trend mean
MAM_min5_trend_mean = MAM_min5_trend.mean(dim='run')
MAM_max5_trend_mean = MAM_max5_trend.mean(dim='run')

# Calculate the SON min 5 trend mean
SON_min5_trend_mean = SON_min5_trend.mean(dim='run')
SON_max5_trend_mean = SON_max5_trend.mean(dim='run')

# In[35]:
# put the trend data into a dataset for plotting
trend_data = {
        'DJF':{
                'MME': slope_DJF_MME[::-1,:],
                'min5': DJF_min5_trend_mean[::-1,:],
                'max5': DJF_max5_trend_mean[::-1,:]
        },
        'JJA':{
                'MME': slope_JJA_MME[::-1,:],
                'min5': JJA_min5_trend_mean[::-1,:],
                'max5': JJA_max5_trend_mean[::-1,:]
        },
        'MAM':{
                'MME': slope_MAM_MME[::-1,:],
                'min5': MAM_min5_trend_mean[::-1,:],
                'max5': MAM_max5_trend_mean[::-1,:]
        },
        'SON':{
                'MME': slope_SON_MME[::-1,:],
                'min5': SON_min5_trend_mean[::-1,:],
                'max5': SON_max5_trend_mean[::-1,:]
        },

}

# In[36]:
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap

# Update the plot_trend_data function to take the axis as an argument
def plot_trend_data(trend_data, ax, title):
    cmap = mpl.cm.RdBu_r
    bounds = [-4.0, -3.5, -3.0, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0]
    # bounds = [-3.0, -2.75, -2.5, -2.25, -2, -1.75, -1.5,-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    ax.coastlines()
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())  # Set the Northern Hemisphere extent
    ax.set_xticks([])
    ax.set_yticks([])
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    im = ax.imshow(trend_data, extent=[-180, 180, 0, 90], transform=ccrs.PlateCarree(central_longitude=180), cmap=cmap, norm=norm)
    ax.set_title(title,fontname='Times New Roman',fontsize=8)
    # Adding the title as text in the bottom-right corner
    return im

#Creat a function to plot the trend data
fig, axs = plt.subplots(4, 3, figsize=(9.5, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# loop over the subplots and plot the data
for i, season in enumerate(['DJF', 'MAM', 'JJA', 'SON']):
    for j, member in enumerate(['MME', 'min5', 'max5']):
        im = plot_trend_data(trend_data[season][member], axs[i, j], f'{season} {member.upper()}')

# add a common colorbar for all subplots
cax = fig.add_axes([0.2, 0.25, 0.7, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('SAT Trend (°C/65yrs)', fontname='Times New Roman',fontsize=10)

# plt.subplots_adjust(hspace=0.1)  # Adjust the spacing between subplots
plt.tight_layout(rect=[0.1, 0.25, 1, 0.95])  # Adjust the layout to accommodate the suptitle and colorbar
plt.savefig('2023-2087-MPI-ESM-SAT_Trend_Patterns.png', bbox_inches='tight', dpi=300)

plt.show()
# In[107]:
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_regression_map(data, ax, title):
    # Define the levels and colors for the colormap
    # Create the colormap and norm
    cmap = mpl.cm.RdBu_r
    # bounds = [-4.0, -3.5, -3.0, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4.0]
    bounds = [-3.0, -2.75, -2.5, -2.25, -2, -1.75, -1.5,-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    ax.coastlines()
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())  # Set the Northern Hemisphere extent
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(data, extent=[-180, 180, 0, 90], transform=ccrs.PlateCarree(central_longitude=180), cmap=cmap, norm=norm)
    # ax.set_title(title, fontsize=10)
    # Adding the title as text in the bottom-right corner
    ax.text(0.98, 0.10, title, transform=ax.transAxes, ha='right', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=1.0), fontdict={'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8})
    return im


# In[108]:


SAT_trend['DJF'].shape
model_members = slope_DJF_data['run']
model_members


# In[111]:


#Plot the 30run SAT trend pattern, MME and the OBS spatial pattern

# Assuming 'model_members' is a list of 30 2D arrays for the model members, 'ensemble_mean' is the 2D array for the ensemble mean,
# 'observational_data' is the 2D array for the observational data.
# Titles for individual subplots (change these titles as per your specific model members)
titles_model = ['M{}'.format(i+1) for i in range(len(model_members))]

# Title for the MME plot
title_mme = 'MME'

# Title for the observational plot
title_observation = 'OBS'

fig, axs = plt.subplots(8, 4, figsize=(10, 6.5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

# Add a title for the entire figure
# fig.suptitle('Northern Hemisphere SAT Regression Patterns', fontsize=14, fontweight='bold', y=0.95, x=0.5)

for i, ax in enumerate(axs.flat):
    if i < len(model_members):
        im = plot_regression_map(slope_MAM_data[i,::-1,:], ax, titles_model[i])
    elif i == len(model_members):
        im = plot_regression_map(slope_MAM_MME[::-1,:], ax, title_mme)
    else:
        im = plot_regression_map(SAT_trend['MAM'][::-1,:], ax, title_observation)

# Add a common colorbar for all subplots
cax = fig.add_axes([0.20, 0.01, 0.7, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('°C/65yrs', fontsize=12)

plt.tight_layout(rect=[0.1, 0.05, 1, 0.95])  # Adjust the layout to accommodate the suptitle and colorbar
plt.savefig('2023-MPI-ESM-LR_MAM_65yr_trend_subplot-MME-OBS.png', bbox_inches='tight', dpi=300)

plt.show()

# In[65]:




