"""
Converts an ascii tecplot file into an hdf5 file. Also
changes field names to be consistent across all file 
formats. 

Inputs:
    --outname (-o): filename to save h5 output to. Defaults to
                    the infile name but extension changed to *.h5
    --infile (-i): filename to convert
    --test (-t): Doesn't do anything yet
"""

import numpy as np
import h5py 
import spiceypy as sp
import glob
import sys
import getopt
from general_functions import *


#data_conversion = {'H_p1_number_density':lambda x: 0.5*x/1.00794, 
#                   'O2_p1_number_density':lambda x: 0.5*x/(2*15.9994), 
#                   'O_p1_number_density':lambda x: 0.5*x/15.9994, 
#                   'CO2_p1_number_density':lambda x: 0.5*x/(15.9994*2+12.0107),
#                   }
data_conversion = {'H_p1_number_density':lambda x: 0.5*x, 
                   'O2_p1_number_density':lambda x: 0.5*x, 
                   'O_p1_number_density':lambda x: 0.5*x, 
                   'CO2_p1_number_density':lambda x: 0.5*x}
f_var_rename = 'Misc/name_conversion.txt'


def convert_file(fname, h5_name):

    # read in file
    with file(fname) as dat_file: 
        
        #Process header inforation
        read_header = True
        dat_vars = []
        while read_header:
            l = dat_file.readline()
            
            # Grab the first variable thats at the end of the line
            if "VARIABLES" in l: dat_vars.append(l.replace("VARIABLES = ", ''))
            
            # If the line begins with " then its another variable
            if l[0] == '"': dat_vars.append(l)

            # Contains the number of elements to read
            if "Nodes" in l: N = int( (l.split(',')[0]).split("=")[-1] )

            # Begining of last line before data
            if "DT" in l: read_header = False

        dat_vars = [v.replace('"', '').replace('\n','') for v in dat_vars]

        # Setup empty data structure
        data = {var:np.empty(N, dtype=float) for var in dat_vars}

        # Iterate through the data
        for i, line in enumerate(dat_file):
            line_dat = line.split(' ')
            line_dat = filter(None, line_dat)
            for j, key in enumerate(dat_vars):
                data[key][i] = float(line_dat[j])

            # There are bonus garbage lines at the end so we have to
            # manually exit the loop
            if i == N-1: break

                
    print data.keys()

    # Going to make the lat/lon/alt fields
    lat, lon, alt = convert_coords_cart_sphere(
            np.array([data['X [R]'], data['Y [R]'], data['Z [R]']]))


    # Set up name conversion dictionary
    name_conversion = {}
    for pair in file(f_var_rename):
        k,v = pair.split(',')
        name_conversion[k] = v[:-1] #remove newline

    # Save data
    with h5py.File(h5_name, 'w') as f:
        print 'Writing to {0}'.format(h5_name)
        for k, v in data.items():
            if  name_conversion[k] in data_conversion.keys():
                v = data_conversion[name_conversion[k]](v)
            f.create_dataset(name_conversion[k], data=v)


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:t:o:", ["infile=","test","outname="])
    except getopt.GetoptError:
        return

    test = False
    outname = ""
    fname = ""

    for opt, arg in opts:
        if opt in ("-t", "--test"):
            test = True
        elif opt in ("-i", "--infile"):
            fname = arg
        elif opt in ("-o", "--outname"):
            outname = arg

    if outname == "": outname = fname.split('.')[0]+'.h5'

    convert_file(fname, outname)

if __name__ == "__main__":
    main(sys.argv[1:])


