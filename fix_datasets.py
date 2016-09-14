import numpy as np
import netCDF4 as nc
import spiceypy as sp

def fix_heliosares():
    fdir = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/Run1/'
    fname = 'Hsw_18_06_14_t00600.nc'

    ds = nc.Dataset(fdir+fname, mode='r')
    x, y, z = ds.variables['x'][:], ds.variables['y'][:], ds.variables['z'][:]

    ux = ds.variables['Ux'][:]
    lat, lon, alt = np.zeros_like(ux), np.zeros_like(ux), np.zeros_like(ux)

    print lat.shape
    for i in range(258):
        for j in range(418):
            for k in range(410):
                p_rec = [x[i], y[j], z[k]]
                p_lat = sp.spiceypy.reclat(p_rec)
                alt[k, j, i], lon[k, j, i], lat[k, j, i] = p_lat
        
    lat = lat*180/np.pi
    lon = lon*180/np.pi
    alt = alt - 3390 

    ds_new = nc.Dataset(fdir+'geo_18_06_14_t00600.nc', mode='w')

    ds_new.createDimension('size_x', 258)
    ds_new.createDimension('size_y', 418)
    ds_new.createDimension('size_z', 410)
    latvar = ds_new.createVariable('latitude', lat.dtype, ds['Density'].dimensions)
    lonvar = ds_new.createVariable('longitude', lon.dtype, ds['Density'].dimensions)
    altvar = ds_new.createVariable('altitude', alt.dtype, ds['Density'].dimensions)
    latvar[:] = lat
    lonvar[:] = lon
    altvar[:] = alt

    ds.close()
    ds_new.close()

fix_heliosares()

fdir = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/Run1/'
fname = 'geo_18_06_14_t00600.nc'
ds = nc.Dataset(fdir+fname, mode='r')
print ds.variables['latitude'][:]
ds.close()



