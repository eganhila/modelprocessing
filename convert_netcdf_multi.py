"""
Converts a collection of netcdf files into a single
hdf5 file. Assumes standard HELIOSARES output.

All *.nc files should be contained in a single directory.

For consistency with other hdf5 files, variables "x", "y",
and "z" will be the corresponding 1D arrays, and the 3D
arrays will be contained in variables "xmesh", etc.

There is some inconsistency in the directionality of
the magnetic field vector in the files saved on the MAVEN
SDC website, so there is a check in place that checks
for how I saved the file, this won't work for/affect anyone
else.

Assumes the files are labeled xxx_yyy.nc where xxx gives the
species name.

Inputs:

    --outname (-o): filename to save h5 output to. Default heliosares.h5
    --dir (-d): Directory to find files in. Defaults to current dir
    --test (-t): Doesn't do anything right now, will eventually test

"""
import numpy as np
import netCDF4 as nc
import h5py
import spiceypy as sp
import glob
import sys
import getopt

f_var_rename = 'Misc/netcdf_names.txt'
mars_r = 3390
axis_labels = ['X_axis', 'Y_axis', 'Z_axis']
# axis_labels = ['x','y','z']

def convert_dataset(fnames, h5_name):

    # Get grid
    with nc.Dataset(fnames[0], mode='r') as ds:
        # x = ds.variables['x'][:]
        # y = ds.variables['y'][:]
        # z = ds.variables['z'][:]
        x = ds.variables[axis_labels[0]][:]
        y = ds.variables[axis_labels[1]][:]
        z = ds.variables[axis_labels[2]][:]

    xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
    xmesh, ymesh, zmesh = xmesh, ymesh, zmesh

    # Reversing the axis because things aren't in ij indexing
    if fdir[-2] == '2': 
        print 'reversing'
        zmesh == zmesh*-1

    # Set dims
    dims = (x.shape[0], y.shape[0], z.shape[0])

    # Going to make the lat/lon/alt fields
    lat, lon, alt = convert_coords_cart_sphere(np.array([xmesh, ymesh, zmesh]))

    # Save data that we've created so far
    with h5py.File(fdir+h5name, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('z', data=z)
        f.create_dataset('xmesh', data=xmesh)
        f.create_dataset('ymesh', data=ymesh)
        f.create_dataset('zmesh', data=zmesh)
        f.create_dataset('latitude', data=lat)
        f.create_dataset('longitude', data=lon)
        f.create_dataset('altitude', data=alt)

    # Set up name conversion dictionary
    name_conversion = {}
    for pair in file(f_var_rename):
        k,v = pair.split(',')
        name_conversion[k] = v

    #Process the rest of the fields
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



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:t:o:", ["dir=","test","outname="])
    except getopt.GetoptError:
        return

    test = False
    outname = "heliosares.h5"
    fdir = ""

    for opt, arg in opts:
        if opt in ("-t", "--test"):
            test = True
        elif opt in ("-d", "--dir"):
            fdir = arg
        elif opt in ("-o", "--outname"):
            outname = arg

    convert_dataset(fdir, outname)

if __name__ == "__main__":
    main(sys.argv[1:])


