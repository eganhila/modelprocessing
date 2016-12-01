import numpy as np
import h5py
import glob

def main():
    fdir = '/Volumes/triton/Data/ModelChallenge/Maven/Traj/'
    fnames = glob.glob(fdir+'*.csv')
    orbit_strings = [s.split('/')[-1].split('_')[1].replace(' ', '') \
                     for s in fnames]
    orbits = [int(o.split('.')[0]) for o in orbit_strings]

    for i in range(len(orbits)):
        orb = orbits[i]
        fn_csv = fnames[i]
        fn_h5 = 'traj_{0:04d}.h5'.format(orb)

        with h5py.File(fdir+fn_h5, 'w') as f_h5:
            x, y, z, alt = np.loadtxt(fn_csv, delimiter=',',
                                      unpack=True)
            f_h5.create_dataset('x', data=x)
            f_h5.create_dataset('y', data=y)
            f_h5.create_dataset('z', data=z)
            f_h5.create_dataset('alt', data=alt)

if __name__ == "__main__":
    main()