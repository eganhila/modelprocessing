import numpy as np
import matplotlib.pyplot as plt

upstream_dat = np.loadtxt('Output/upstream.csv', delimiter=',', unpack=True)
names =  ['bsw_x', 'bsw_y', 'bsw_z', 'bsw_mag', 'psw', 'npsw', 'nasw', 'vpsw', 'tpmagsw', 'vvecsw_x', 'vvecsw_y', 'vvecsw_z']

for i in range(12):
    name = names[i]
    dat = upstream_dat[i+1]

    plt.hist(dat, bins=100)
    plt.savefig('Output/drivershist_{0}.pdf'.format(name))
    plt.close()
