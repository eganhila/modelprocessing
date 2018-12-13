import numpy as np
import h5py
import sys
import getopt


def add_electric_field(infile):

    # Load data
    ds = h5py.File(infile, 'a')

    # Get all the relevant velocities
    vel_nd = np.array([ds["electron_velocity_x"][:],
                              ds["electron_velocity_y"][:],
                              ds["electron_velocity_z"][:]])


    # Get magnetic field
    mag_nd = np.array([ds['magnetic_field_x'][:],
                    ds['magnetic_field_y'][:],
                    ds['magnetic_field_z'][:]])

    # Original shape
    shape = mag_nd.shape

    # Flatten mag
    mag = mag_nd.reshape(3,-1).T
    vel = vel_nd.reshape(3,-1).T


    # Calculate the corresponding motional electric field
    E = -1*np.cross(vel, mag).T.reshape(shape)


    # save the output 
    ds.create_dataset('electric_field_x', data=E[0])
    ds.create_dataset('electric_field_y', data=E[1])
    ds.create_dataset('electric_field_z', data=E[2])

    ds.close()


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"i:",
                                    ["infile="])
    except getopt.GetoptError:
        print(getopt.GetoptError())
        print('error')
        return
    
    infile = None
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg

    add_electric_field(infile)


if __name__ == '__main__':
    main(sys.argv[1:])
