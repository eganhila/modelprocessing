import numpy as np
import matplotlib.pyplot as plt
from general_functions import *
from mayavi import mlab

orbit = 371
trange_utc = get_orbit_times(orbits=np.array([371]))
coords, ltimes = get_path_pts(trange_utc, Npts=250, units_mr=True)

line = np.linspace(0,2,50)
zero = np.zeros_like(line)
mlab.plot3d(line, zero, zero, color=(0,0,0), opacity=0.8)
mlab.plot3d(zero, line, zero, color=(0,0,0), opacity=0.8)
mlab.plot3d(zero, zero, line, color=(0,0,0), opacity=0.8)
mlab.text3d(2.1,0,0, text='X',color=(0,0,0), scale=(0.2,0.2,0.2))
mlab.text3d(0,2.1,0, text='Y',color=(0,0,0), scale=(0.2,0.2,0.2))
mlab.text3d(0,0,2.1, text='Z',color=(0,0,0), scale=(0.2,0.2,0.2))

mlab.plot3d(coords[0], coords[1], coords[2], ltimes, colormap='inferno')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

mlab.mesh(x,y,z, scalars=x, colormap='copper', opacity=0.95)
mlab.view(azimuth=10)
mlab.savefig('Output/test.png')
