"""
Converts a collection of netcdf files into a single
hdf5 file. Assumes standard HELIOSARES output.

All *.nc files should be contained in a single directory.
The new hdf5 file will be created in the parent directory.

For consistency with other hdf5 files, variables "x", "y",
and "z" will be the corresponding 1D arrays, and the 3D
arrays will be contained in variables "xmesh", etc.

There is some inconsistency in the directionality of
the magnetic field vector in the files saved on the MAVEN
SDC website, so there is a check in place that checks
for how I saved the file, this won't work for/affect anyone
else.

Because I am lazy, you need to manually input what dimensions
the data is that you want to grab in the variable dims.

Assumes the files are labeled xxx_yyy.nc where xxx gives the
species name.

"""
import numpy as np
import netCDF4 as nc
import h5py
import spiceypy as sp
import glob


fdir = '/Volumes/triton/Data/ModelChallenge/R2349/Heliosares/'
h5name = '../helio_r2349.h5' 
fnames = glob.glob(fdir+'*.nc')
f_var_rename = 'Misc/netcdf_names.txt'
dims = (386, 386, 202)
#dims = (410, 418, 258)



with nc.Dataset(fnames[0], mode='r') as ds:
    # x = ds.variables['x'][:]
    # y = ds.variables['y'][:]
    # z = ds.variables['z'][:]
    x = ds.variables['X_axis'][:]
    y = ds.variables['Y_axis'][:]
    z = ds.variables['Z_axis'][:]

xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
xmesh, ymesh, zmesh = xmesh, ymesh, zmesh

# Reversing the axis because things aren't in ij indexing
if fdir[-2] == '2': 
    print 'reversing'
    zmesh == zmesh*-1

with h5py.File(fdir+h5name, 'w') as f:
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('z', data=z)
    f.create_dataset('xmesh', data=xmesh)
    f.create_dataset('ymesh', data=ymesh)
    f.create_dataset('zmesh', data=zmesh)

# Set up name conversion dictionary
name_conversion = {}
for pair in file(f_var_rename):
    k,v = pair.split(',')
    name_conversion[k] = v

with h5py.File(fdir+h5name, 'r+') as f:
    for fname in fnames:
        print "Processing: ", fname
        with nc.Dataset(fname, mode='r') as ds:

            species = fname.split('/')[-1].split('_')[0]

            for k,v in ds.variables.items():
                if np.all(v.shape == dims):
                    if species+'_'+k in name_conversion:
                        key = name_conversion[species+'_'+k]
                    elif k in name_conversion:
                        key = name_conversion[k]
                    else:
                        key = species+"_"+k

                    if species == 'Magw' and fdir[-2] == '2' and k!='Bz':
                        print 'mag'
                        f.create_dataset(key, data=-1*v[:].T)
                    else:
                        f.create_dataset(key, data=v[:].T)




