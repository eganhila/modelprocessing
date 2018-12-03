import h5py
import sys
import glob
import numpy as np
import getopt

def create_averaged_ds(indir, method):
    
    files = glob.glob(indir+"/*.h5")

    # do a test open
    with h5py.File(files[0], 'r') as f:
        shape = f['magnetic_field_x'].shape
        fields = list(f.keys())

    if method == "average": func = np.average
    if method == "median": func = np.median

    newfile = indir+"/"+method+".h5"

    N_files = len(files)

    for field in fields:
        all_data = np.zeros((N_files,)+shape)

        for i, fname in enumerate(files):
            with h5py.File(fname, 'r') as f:
                all_data[i] = f[field][:]

        avg_data = func(all_data, axis=0)

        with h5py.File(newfile, 'a') as f:
            f.create_dataset(field, data=avg_data)
    


def main(argv):
    opts, args = getopt.getopt(argv, "", ["indir=", "method="])

    for opt, arg in opts:
        if opt == "--indir":
            indir = arg
        elif opt == "--method":
            method = arg

    create_averaged_ds(indir, method)

if __name__ == "__main__":
    main(sys.argv[1:])
