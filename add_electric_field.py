import numpy as np
import h5py
import sys
import getopt


def add_electric_field(infile):

    # Load data
    ds = h5py.File(infile, 'a')

    # Get all the relevant velocities
    velocities = [k[:-2] for k in ds.keys() if 'velocity_x' in k] 

    # Get magnetic field
    mag_nd = np.array([ds['magnetic_field_x'][:],
                    ds['magnetic_field_y'][:],
                    ds['magnetic_field_z'][:]])

    # Original shape
    shape = mag_nd.shape

    # Flatten mag
    mag = mag_nd.reshape(3,-1).T

    for vel_prefix in velocities:

        vel_nd = np.array([ds[vel_prefix+'_x'][:],
                           ds[vel_prefix+'_y'][:],
                           ds[vel_prefix+'_z'][:]])

        vel = vel_nd.reshape(3,-1).T

        # Calculate the corresponding electric field
        E = -1*np.cross(vel, mag).T.reshape(shape)

        E_prefix = str.replace(str(vel_prefix), 'velocity', 'electric_field')
        if E_prefix+'_x' in ds.keys():
            del ds[E_prefix+'_x']
            del ds[E_prefix+'_y']
            del ds[E_prefix+'_z']

        # save the output 
        ds.create_dataset(E_prefix+'_x', data=E[0])
        ds.create_dataset(E_prefix+'_y', data=E[1])
        ds.create_dataset(E_prefix+'_z', data=E[2])

    ds.close()


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"i:",
                                    ["infile="])
    except getopt.GetoptError:
        print getopt.GetoptError()
        print 'error'
        return
    
    infile = None
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg

    add_electric_field(infile)


if __name__ == '__main__':
    main(sys.argv[1:])
