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
    --indir (-i): Directory to find files in. Defaults to current dir
    --flip (-f): Flip the x-component of the magnetic for that buggy file
    --test (-t): Doesn't do anything right now, will eventually test

"""
import numpy as np
import netCDF4 as nc
import h5py
import spiceypy as sp
import glob
import sys
import getopt
from general_functions import *

f_var_rename = 'misc/name_conversion.txt'
mars_r = 3390
axis_labels = ['X_axis', 'Y_axis', 'Z_axis']
# axis_labels = ['x','y','z']

def convert_dataset(fdir, h5_name, flip=False):
    print 'Starting: {0}'.format(h5_name)
    fnames = glob.glob(fdir+'*.nc')

    # Get grid
    with nc.Dataset(fnames[0], mode='r') as ds:
        x = ds.variables[axis_labels[0]][:]+61.8125
        y = ds.variables[axis_labels[1]][:]+61.8125
        z = ds.variables[axis_labels[2]][:]-61.8125

    print 'Setting up grid'
    zmesh, ymesh, xmesh = np.meshgrid(z, y, x, indexing='ij')
    xmesh = xmesh.T
    ymesh = ymesh.T
    zmesh = zmesh.T

    # Set dims
    dims = (x.shape[0], y.shape[0], z.shape[0])

    # Going to make the lat/lon/alt fields
    print 'Converting cartesian to spherical'
    #lat, lon, alt = convert_coords_cart_sphere(np.array([xmesh, ymesh, zmesh]))

    # Save data that we've created so far
    with h5py.File(fdir+h5_name, 'w') as f:
        print 'Saving spatial data'
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        f.create_dataset('z', data=z)
        f.create_dataset('xmesh', data=xmesh)
        f.create_dataset('ymesh', data=ymesh)
        f.create_dataset('zmesh', data=zmesh)
        #f.create_dataset('latitude', data=lat)
        #f.create_dataset('longitude', data=lon)
        #f.create_dataset('altitude', data=alt)

    # Set up name conversion dictionary
    name_conversion = {}
    for pair in file(f_var_rename):
        k,v = pair.split(',')
        name_conversion[k] = v[:-1] #remove newline


    # Load in H+ data for combining
    data = {}
    for fname in fnames:
        print "Processing: ", fname
        with nc.Dataset(fname, mode='r') as ds:
            species = fname.split('/')[-1].split('_')[0]
            for k,v in ds.variables.items():
                if np.all(v.shape == dims) or np.all(v.shape == dims[::-1]):

                    if species+'_'+k in name_conversion:
                        key = name_conversion[species+'_'+k]
                    elif k in name_conversion:
                        key = name_conversion[k]
                    else:
                        key = species+"_"+k

                    if species in ['Hpl', 'Hsw']: 
                        data[key] = v[:].T
                    elif flip and key == 'magnetic_field_x':
                        v = -1*v[:]   
                    else:
                        with h5py.File(fdir+h5_name, 'r+') as f:
                            f.create_dataset(key, data=v[:].T)


    # Combine sw and pl H+
    data['H_p1_number_density'] = data['Hpl_number_density'] + data['Hsw_number_density']
    weights = [data['Hpl_number_density'], data['Hsw_number_density']]
    zeros = np.sum(weights,axis=0)==0 
    weights[0][zeros]=1
    weights[1][zeros]=1

    for f in ['temperature', 'velocity_x', 'velocity_y', 'velocity_z']:
        dd = [data['Hpl_{0}'.format(f)], data['Hsw_{0}'.format(f)]]
        data['H_p1_{0}'.format(f)] = np.average(dd, weights=weights, axis=0) 

    # Save H+
    with h5py.File(fdir+h5_name, 'r+') as f:
        for k, v in data.items(): f.create_dataset(k, data=v)



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:t:o:f", ["indir=","test","outname=", "flip"])
    except getopt.GetoptError:
        return

    test = False
    outname = "heliosares.h5"
    fdir = ""
    flip = False

    for opt, arg in opts:
        if opt in ("-t", "--test"):
            test = True
        elif opt in ("-i", "--indir"):
            fdir = arg
        elif opt in ("-o", "--outname"):
            outname = arg
        elif opt in ("-f", "--flip"):
            flip = True

    convert_dataset(fdir, outname, flip=flip)

if __name__ == "__main__":
    main(sys.argv[1:])


