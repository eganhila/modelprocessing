import numpy as np
import pandas as pd
import sys
import getopt
import spiceypy as sp

sp.furnsh("/Users/hilaryegan/Projects/ModelChallenge/ModelProcessing/misc/maven_spice.txt")

col_dict = {1:    'time', 
            190:  'x',
            191:  'y',
            192:  'z',
            197:  'altitude',
            #SWEA
            23: 'electron_number_density',

            #SWIA
            41:   'H_p1_number_density', 
            43:    'H_p1_velocity_x',
            45:    'H_p1_velocity_y',
            47:    'H_p1_velocity_z',
            #STATIC
            #          54:     'H_p1_number_density_STATIC',
            #56:    'O_p1_number_density',
            #58:    'O2_p1_number_density',

            #NGIMS
            172:    'O_p1_number_density',
            163:    'O2_p1_number_density',
            166:    'CO2_p1_number_density',

            #MAG
            128:    'magnetic_field_x',
            130:   'magnetic_field_y',
            132:   'magnetic_field_z',
        }

def convert_kp_csv(infile, outname, tlim=False):

    cols =  sorted(col_dict.keys())
    
    # load data
    in_dat = np.loadtxt(infile, usecols = np.array(cols)-1, converters = {0: sp.str2et}, unpack=True) 
    
    # Time limit if desired
    if tlim:
        time = in_dat[0]
        tlim_str = ('2015-12-14/16:27:57','2015-12-14/20:59:35')
        tlim_et = (sp.str2et(tlim_str[0]), sp.str2et(tlim_str[1]))

        i0 = np.argmax(time > tlim_et[0])
        i1 = np.argmax(time > tlim_et[1])
    else:
        i0 = 0
        i1 = -1

    header = ''
    for col in cols: 
        header+=(col_dict[col]+',')

    np.savetxt(outname, in_dat[:,i0:i1].T, delimiter=',',header=header, comments='')

        



def main(argv):

    try:
        opts, args = getopt.getopt(argv, "i:o:t", ["infile=", "outname=", "tlim"])
    except getopt.GetoptError:
        print 'err'

    tlim = False
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-o", "--outname"):
            outname = arg
        elif opt in ("-t", "--tlim"):
            tlim = True

    convert_kp_csv(infile, outname, tlim=tlim)


if __name__ == '__main__':
    main(sys.argv[1:])

