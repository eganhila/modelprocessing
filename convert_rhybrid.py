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

f_var_rename = 'misc/name_conversion.txt'
# Set up name conversion dictionary
name_conversion = {}
for pair in file(f_var_rename):
    k,v = pair.split(',')
    name_conversion[k] = v[:-1] #remove newline

def convert_dataset(infile, outname):

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
    locs_sorted = sorted(locs.iteritems(), key=oper.itemgetter(0))
    locs_sorted_idx = np.array([ii[1] for ii in locs_sorted])

    #variables to process
    vars_1D_complete = ['n_O+_ave', 'n_O2+_ave', 'n_He++sw_ave']
    vars_3D_complete = ['v_O+_ave', 'v_O2+_ave', 'v_He++sw_ave', 'cellBAverage']
    vars_1D_add = [('n_H+sw_ave','n_H+planet_ave')]
    vars_3D_ave = [('v_H+sw_ave','v_H+planet_ave')]
    vars_spatial = ['x', 'y', 'z']


    with h5py.File(outname) as f:

        for v in vars_1D_complete:
            dat = vr.read_variable(v)
            dat = dat[locs_sorted_idx].reshape(nz, ny, nx).T
            f.create_dataset(name_conversion[v], data=dat)

        for v in vars_3D_complete:
            for x_i, x in enumerate(['_x','_y', '_z']):
                dat = vr.read_variable(v)[:, x_i]
                dat = dat[locs_sorted_idx].reshape(nz, ny, nx).T
                f.create_dataset(name_conversion[v]+x, data=dat)

        for v_add_list, v_ave_list in zip(vars_1D_add, vars_3D_ave):
            dat_add = np.zeros_like(dat)
            dat_ave = np.zeros_like(dat)

            for v_add, v_ave in zip(v_add_list, v_ave_list): 
                dat = vr.read_variable(v_add)
                dat = dat[locs_sorted_idx].reshape(nz, ny, nx).T

                dat_all += dat

            f.create_dataset(name_conversion[v_list[0]], data=dat_all)

        for v in vars_3D_ave:
            dat_all = np.zeros_like(dat)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:t:o:", ["infile=","test","outname="])
    except getopt.GetoptError:
        return

    test = False
    outname = "heliosares.h5"
    fdir = ""

    for opt, arg in opts:
        if opt in ("-t", "--test"):
            test = True
        elif opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-o", "--outname"):
            outname = arg

    convert_dataset(infile, outname)

if __name__ == "__main__":
    main(sys.argv[1:])


