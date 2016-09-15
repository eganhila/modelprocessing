import numpy as np
import h5py 
import spiceypy as sp

fdir = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/BATSRUS/'
fname = '3d__ful_4_n00060000_PERmin-SSLONG180.dat'

dat_vars = ['x', 'y', 'z', 'r', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'p', 'rhp',
            'uxh', 'uyh', 'uzh', 'php', 'rop2', 'uxo2', 'uyo2', 'uzo2', 'po2',
            'rop', 'uxo', 'uyo', 'uzo', 'pop', 'rco2', 'uxco2', 'uyco2', 'uzco2',
            'pco2p', 'bx', 'by', 'bz', 'E', 'jx', 'jy', 'jz']

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
alt = alt - 3390 

data['latitude'] = lat
data['longitude'] = lon
data['altitude'] = alt

h5name = fname.split('.')[0]+'.h5'

with h5py.File(fdir+h5name, 'w') as f:
    for k, v in data.items():
        f.create_dataset(k, data=v)

