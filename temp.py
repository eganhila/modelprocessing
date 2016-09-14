import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fdir = '/Volumes/triton/Data/ModelChallenge/BATSRUS/'
fname = '3d__ful_4_n00060000_AEQNmax-SSLONG0.dat' 

dat_vars = ['x', 'y', 'z']#, 'r', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'p', 'rhp',
           # 'uxh', 'uyh', 'uzh', 'php', 'rop2', 'uxo2', 'uyo2', 'uzo2', 'po2', 
           # 'rop', 'uxo', 'uyo', 'uzo', 'pop', 'rco2', 'uxco2', 'uyco2', 'uzco2',
           # 'pco2p', 'bx', 'by', 'bz', 'E', 'jx', 'jy', 'jz']

data = {var:np.empty(11347200, dtype=float) for var in dat_vars}

dat_file = file(fdir+fname)

i = -1
for line in dat_file:
    i+=1
    if i < 64: 
        print line 
        continue
    line_dat = line.split(' ')[1:]
    data['x'][i-64] = float(line_dat[0])
    data['y'][i-64] = float(line_dat[1])
    data['z'][i-64] = float(line_dat[2])
    


plt.subplots(221)
plt.scatter(data['x'], data['y'], rasterized=True)
ax = plt.gca()
ax.set_yscale('symlog')
ax.set_xscale('symlog')

plt.subplots(222)
plt.scatter(data['z'], data['y'], rasterized=True)
ax = plt.gca()
ax.set_yscale('symlog')
ax.set_xscale('symlog')

plt.subplots(223)
plt.scatter(data['x'], data['z'], rasterized=True)
ax = plt.gca()
ax.set_yscale('symlog')
ax.set_xscale('symlog')

plt.savefig('Output/temp.pdf')

