import numpy as np
import h5py 
import spiceypy as sp
import glob

mars_r = 3390

dat_vars = ['x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'p', 'rhp',
            'uxh', 'uyh', 'uzh', 'php', 'rop2', 'uxo2', 'uyo2', 'uzo2', 'po2',
            'rop', 'uxo', 'uyo', 'uzo', 'pop', 'rco2', 'uxco2', 'uyco2', 'uzco2',
            'pco2p', 'b1x', 'b1y', 'b1z', 'E', 'jx', 'jy', 'jz']

name_conversion = {'x':'x', 'y':'y', 'z':'z',
                  'r':'number_density', 'vx':'velocity_x', 'vy':'velocity_y', 'vz':'velocity_z',
                  'bx':'magnetic_field_x', 'by':'magnetic_field_y', 'bz':'magnetic_field_z',
                  'p':'pressure', 'rhp':'H_p1_number_density', 'uxh':'H_p1_velocity_x', 'uyh':'H_p1_velocity_y', 'uzh':'H_p1_velocity_z',
                  'php':'H_p1_pressure', 'rop2':'O2_p1_number_density', 'uxo2':'O2_p1_velocity_x', 'uyo2':'O2_p1_velocity_y', 'uzo2':'O2_p1_velocity_z',
                  'po2':'O2_p1_pressure', 'rop':'O_p2_number_density', 'uxo':'O_p2_velocity_x', 'uyo':'O_p2_velocity_y', 'uzo':'O_p2_velocity_z',
                  'pop':'O_p2_pressure', 'rco2':'CO2_p1_number_density', 'uxco2':'CO2_p1_velocity_x', 'uyco2':'CO2_p1_velocity_y','uzco2':'CO2_p1_velocity_z',
                  'pco2p':'CO2_p1_pressure',
                  'b1x':'bx1', 'b1y':'by1', 'b1z':'bz1', 'E':'electric_field_magnitude',
                  'jx':'current_x', 'jy':'current_y', 'jz':'current_z',
                  'altitude':'altitude', 'latitude':'latitude', 'longitude':'longitude'}

data_conversion = {'x':lambda x: mars_r*x, 'y':lambda x: mars_r*x, 'z':lambda x: mars_r*x,
        'rhp':lambda x: x/1.00794, 'rop2': lambda x: x/(2*15.9994), 'rop': lambda x:x/15.9994, 'rco2':lambda x: x/(15.9994*2+12.0107)}




def convert_file(fdir, fname):

    data = {var:np.empty(5688000, dtype=float) for var in dat_vars}

    dat_file = file(fdir+fname)
    i = -1
    for line in dat_file:
        i+=1
        line_dat = line.split(' ')
        if i < 64: continue
        elif i< 5688064:
            for j, key in enumerate(dat_vars):
                data[key][i-64] = float(line_dat[j+1])
        else:
            break
                
    dat_file.close()

    x,y,z = data['x'], data['y'], data['z']

    N = z.shape[0]

    lat, lon, alt = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        p_rec = [x[i], y[i], z[i]]
        p_lat = sp.spiceypy.reclat(p_rec)
        alt[i], lon[i], lat[i] = p_lat
        
    lat = lat*180/np.pi
    lon = lon*180/np.pi
    alt = alt*3390 

    data['latitude'] = lat
    data['longitude'] = lon
    data['altitude'] = alt

    h5name = fname.split('.')[0]+'.h5'

    with h5py.File(fdir+h5name, 'w') as f:
        for k, v in data.items():
            if k in data_conversion.keys():
                v = data_conversion[k](v)
            f.create_dataset(name_conversion[k], data=v)


def main():

    fdir = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/BATSRUS/'
    fnames = glob.glob(fdir+'*.dat')

    for fname in fnames:
        print fname
        convert_file(fdir, fname.split('/')[-1])

if __name__ == '__main__':
    main()
