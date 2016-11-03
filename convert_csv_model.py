import numpy as np
import h5py
import pandas as pd
import spiceypy as sp

mars_r = 3390
names = {u'X [R]':'x', u'Y [R]':'y', u'Z [R]':'z',
           u'`r [amu/cm^3]':'number_density', u'Hp [amu/cm^3]':'H_p1_number_density',
           u'O2p [amu/cm^3]':'O2_p1_number_density', u'Op [amu/cm^3]':'O_p1_number_density',
           u'CO2p [amu/cm^3]':'CO2_p1_number_density', u'U_x [km/s]':'velocity_x',
           u'U_y [km/s]':'velocity_y', u'U_z [km/s]':'velocity_z', 
           u'B_x [nT]':'magnetic_field_x', u'B_y [nT]':'magnetic_field_y',
           u'B_z [nT]':'magnetic_field_z', u'p [nPa]':'pressure',
           u'J_x [`mA/m^2]':'current_x', u'J_y [`mA/m^2]':'current_y',
           u'J_z [`mA/m^2]':'current_z'}
units = {u'X [R]':'km', u'Y [R]':'km', u'Z [R]':'km',
           u'`r [amu/cm^3]':'cm**-3', u'Hp [amu/cm^3]':'cm**-3',
           u'O2p [amu/cm^3]':'cm**-3', u'Op [amu/cm^3]':'cm**-3',
           u'CO2p [amu/cm^3]':'cm**-3', u'U_x [km/s]':'km/s',
           u'U_y [km/s]':'km/s', u'U_z [km/s]':'km/s', 
           u'B_x [nT]':'nT', u'B_y [nT]':'nT',
           u'B_z [nT]':'nT', u'p [nPa]':'nPa',
           u'J_x [`mA/m^2]':'mA/m**2', u'J_y [`mA/m^2]':'mA/m**2',
           u'J_z [`mA/m^2]':'mA/m**2'}

conversion = {u'X [R]':lambda x: mars_r*x,
              u'Y [R]':lambda x: mars_r*x,
              u'Z [R]':lambda x: mars_r*x,
              u'`r [amu/cm^3]':lambda x:x,
              u'Hp [amu/cm^3]':lambda x: x/1.00794,
              u'O2p [amu/cm^3]':lambda x: x/(2*15.9994), 
              u'Op [amu/cm^3]':lambda x: x/(15.9994),
              u'CO2p [amu/cm^3]':lambda x: x/(2*15.9994+12.0107), 
              u'U_x [km/s]':lambda x:x,
              u'U_y [km/s]':lambda x:x, 
              u'U_z [km/s]':lambda x:x, 
              u'B_x [nT]':lambda x:x, 
              u'B_y [nT]':lambda x:x,
              u'B_z [nT]':lambda x:x, 
              u'p [nPa]':lambda x:x,
              u'J_x [`mA/m^2]':lambda x:x, 
              u'J_y [`mA/m^2]':lambda x:x,
              u'J_z [`mA/m^2]':lambda x:x}

def main():
    fdir = '/Volumes/triton/Data/ModelChallenge/R2349/'
    fname = 'batrus_3dmhd_2349.csv'
    df = pd.read_csv(fdir+fname)

    h5name = fname.replace('.csv', '.h5')
    with h5py.File(fdir+h5name, 'w') as f:
	for c in df.columns:
	    if c== 'Unnamed: 18': continue
	    d = f.create_dataset(names[c], data=conversion[c](df[c].values))
	    d.attrs['units'] = units[c]

if __name__ == '__main__':
    main()
