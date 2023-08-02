import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

def norm(a):
    
    a_norm = (a-a.mean())/a.std()
    return a_norm


#definition of the functions to drawing map with the figure extent, space of longitude and latitude 
def plot_with_map(axes, leftlon, rightlon, lowerlat, upperlat, spec1, spec2, title):
    lon_formatter=cticker.LongitudeFormatter()
    lat_formatter=cticker.LatitudeFormatter()    
    axes.set_extent(img_extent, crs=ccrs.PlateCarree())
    axes.add_feature(cfeature.COASRLINE.with_scale('50m'))
    axes.set_xticks(np.arange(leftlon, rightlon+spec1, spec1), crs=ccrs.PlateCarree())
    axes.set_yticks(np.arange(lowerlat, upperlat+spec2, spec2), crs=ccrs.PlateCarree())
    lon_formatter=cticker.LongitudeFormatter()
    lat_formatter=cticker.LatitudeFormatter()    
    axes.xaxis.set_major_formatter(lon_formatter)
    axes.yaxis.set_major_formatter(lat_formatter)    
    axes.set_title(title, fontsize=14, loc='left')