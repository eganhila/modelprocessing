"""
Converts a vlsv file into an hdf5 file. Assumes
standard rhybrid output.
"""

from analysator import pytools as pt
import numpy as np
import h5py
import sys
import operator as oper
import getopt
import os

dirname = os.path.dirname(__file__)
f_var_rename =  os.path.join(dirname,'misc/name_conversion.txt')
# Set up name conversion dictionary
name_conversion = {}
for pair in open(f_var_rename):
    k,v = pair.split(',')
    name_conversion[k] = v[:-1] #remove newline


data_conversion = {"number_density": lambda x: x*1e-6,
                   "velocity": lambda x: x*1e-3,
                   "electric_field": lambda x: x*1e6, #put in microV/m to match nT km/s
                   "magnetic_field": lambda x: x*1e9, #nT
                   }


def convert_dataset(infile, outname, radius=3390):

    # Read in file using vlsv reader from pytools
    vr = pt.vlsvfile.VlsvReader(infile)

    # Grid dims
    [xmin,ymin,zmin,xmax,ymax,zmax] = vr.get_spatial_mesh_extent()
    [mx,my,mz] = vr.get_spatial_mesh_size() # how many blocks per direction
    [sx,sy,sz] = vr.get_spatial_block_size() # how many cells per block per direction
    nx = mx*sx # number of cells along x
    ny = my*sy # number of cells along y
    nz = mz*sz # number of cells along z

    # Cell locations
    locs = vr.get_cellid_locations()
    locs_sorted = sorted(locs.items(), key=oper.itemgetter(0))
    locs_sorted_idx = np.array([ii[1] for ii in locs_sorted])


    vars_1D_complete = ['n_O+_ave', 'n_O2+_ave', "T_tot", 'n_tot_ave']
    vars_3D_complete = ['v_O+_ave', 'v_O2+_ave', 'cellBAverage', 'cellUe', "cellJ", "nodeE"]
    vars_multi = [('H+', ('sw_ave', 'planet_ave'), 'H_p1'),
                  ('O+', ('io_ave', 'torus_ave'), 'O_p1'),
                  ('S+', ('io_ave', 'torus_ave'), 'S_p1'),
                  ]

    vars_spatial = ['x', 'y', 'z']


    base_shape = None

    with h5py.File(outname) as f:

        for v in vars_1D_complete:
            dat = vr.read_variable(v)
            if dat is None: continue 
            dat = dat[locs_sorted_idx].reshape(nz, ny, nx).T

            for dc in data_conversion.keys():
                if dc in name_conversion[v]:
                    dat = data_conversion[dc](dat)
            f.create_dataset(name_conversion[v], data=dat)

            base_shape = dat.shape

        for v in vars_3D_complete[::-1]:
            for x_i, x in enumerate(['_x','_y', '_z']):
                dat = vr.read_variable(v)
                if dat is None: continue 
                dat = dat[:, x_i]
                dat = dat[locs_sorted_idx].reshape(nz, ny, nx).T
                for dc in data_conversion.keys():
                    if dc in name_conversion[v]:
                        dat = data_conversion[dc](dat)
                if name_conversion[v] in data_conversion.keys():
                    dat = data_conversion[name_conversion[v]](dat)
                f.create_dataset(name_conversion[v]+x, data=dat)

        # Do the sw/pl ion averaging

        for base, ptypes, outbase in vars_multi:

            n_dat = np.zeros(base_shape)
            v_dat = [np.zeros(base_shape) for i in range(3)]

            for ptype in ptypes:
                temp = vr.read_variable('n_'+base+ptype)

                if temp is None: continue 


                temp = data_conversion['number_density'](temp)
                n_dat += temp[locs_sorted_idx].reshape(nz, ny, nx).T

                for x_i, x in enumerate(['_x','_y', '_z']):
                    tempv = vr.read_variable('v_'+base+ptype)[:, x_i]*temp
                    v_dat[x_i] += tempv[locs_sorted_idx].reshape(nz, ny, nx).T

            v_dat = [v/n_dat for v in v_dat]
            v_dat = data_conversion['velocity'](np.array(v_dat))

            f.create_dataset(outbase+'_number_density', data=n_dat)

            for x_i, x in enumerate(['_x','_y', '_z']):
                f.create_dataset(outbase+'_velocity'+x, data=v_dat[x_i])


        # Make spatial vars
        x = np.linspace(xmin, xmax, nx)/1000
        y = np.linspace(ymin, ymax, ny)/1000
        z = np.linspace(zmin, zmax, nz)/1000

        zmesh, ymesh, xmesh = np.meshgrid(z, y, x, indexing='ij')
        xmesh = xmesh.T
        ymesh = ymesh.T
        zmesh = zmesh.T

        for var, name in zip([x,y,z,xmesh,ymesh,zmesh],
                             ['x','y','z','xmesh','ymesh','zmesh']):
            f.create_dataset(name, data=var)

        f.attrs.create('radius', radius)
        print('Saving: ' + outname)
        f.close()




def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:t:o:r:", ["infile=","test","outname=", "radius="])
    except getopt.GetoptError:
        return

    test = False
    outname = "rhybrid.h5"
    fdir = ""
    radius = 3390

    for opt, arg in opts:
        if opt in ("-t", "--test"):
            test = True
        elif opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-o", "--outname"):
            outname = arg
        elif opt in ("-r", "--radius"):
            radius = arg

    convert_dataset(infile, outname, radius)

if __name__ == "__main__":
    main(sys.argv[1:])


