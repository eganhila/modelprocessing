"""
Combines multiple standard outputs into one
file by averaging them together.

Inputs:

    --outname (-o): filename to save h5 output to. Default heliosares.h5
    --indir (-i): Directory to find files in. Defaults to current dir

"""
import h5py
import numpy as np
import glob
import sys
import getopt

def combine_datasets(fdir, outname):

    fnames = glob.glob(fdir+'*.h5')
    N_files = len(fnames)

    with h5py.File(outname, 'w-') as f:
        print 'Creating output file: {0}'.format(outname)


    with h5py.File(fnames[0], 'r') as f:
        var_names = f.keys() 
        shape = f[var_names[0]].shape

    for var in var_names:
        if var in ['x','y','z', 'xmesh', 'ymesh', 'zmesh']: continue
        print "Processing variable: {0}".format(var)

        dat = np.zeros(shape)

        for fname in fnames:
            with h5py.File(fname, 'r') as f:
                dat += f[var][:]


        dat = dat/N_files

        with h5py.File(outname, 'r+') as f:
            f.create_dataset(var, data=dat)

    for var in ['x','y','z', 'xmesh', 'ymesh', 'zmesh']:
        with h5py.File(fnames[0], 'r') as f:
            dat = f[var][:]
        with h5py.File(outname, 'r+') as f:
            f.create_dataset(var, data=dat)



    


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:t:o:", ["indir=","test","outname="])
    except getopt.GetoptError:
        return

    test = False
    outname = "heliosares.h5"
    fdir = ""

    for opt, arg in opts:
        if opt in ("-i", "--indir"):
            fdir = arg
        elif opt in ("-o", "--outname"):
            outname = arg

    combine_datasets(fdir, outname)

if __name__ == '__main__':
    main(sys.argv[1:])
