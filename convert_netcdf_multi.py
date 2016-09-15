import numpy as np
import netCDF4 as nc
import h5py
import spiceypy as sp
import glob

fdir = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/Run2/'
h5name = 'run2.h5' 
fnames = glob.glob(fdir+'*.nc')

dims = (410, 418, 258)

with h5py.File(fdir+h5name, 'w') as f:
    for fname in fnames:
        print fname
        with nc.Dataset(fname, mode='r') as ds:

            species = fname.split('/')[-1].split('_')[0]

            for k,v in ds.variables.items():
                if np.all(v.shape == dims):
                    print k
                    f.create_dataset(species+'_'+k, data=v[:])

with nc.Dataset(fnames[0], mode='r') as ds:
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    z = ds.variables['z'][:]

    
print 'creating mesh'
ymesh, zmesh, xmesh = np.meshgrid(y, z, x)
x = xmesh.flatten()
y = ymesh.flatten()
z = zmesh.flatten()

N = z.shape[0]

lat, lon, alt = np.zeros(N), np.zeros(N), np.zeros(N)

print 'calculating lat/lon/alt'
for i in range(N):
    p_rec = [x[i], y[i], z[i]]
    p_lat = sp.spiceypy.reclat(p_rec)
    alt[i], lon[i], lat[i] = p_lat
    
lat = lat*180/np.pi
lon = lon*180/np.pi
alt = alt - 3390 


print 'saving file'
with h5py.File(fdir+h5name, 'r+') as f:
    f.create_dataset('latitude', data=lat)
    f.create_dataset('altitude', data=alt)
    f.create_dataset('longitude', data=lon)


