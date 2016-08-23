import yt
from load_heliosares import *
import spiceypy as sp
import matplotlib.pyplot as plt
import numpy as np


# Set up spice orbits
sp.furnsh("maven_spice.txt")
step = 40#100
utc = ['2015-12-14/16:27:57','2015-12-14/20:59:35']
etOne = sp.str2et(utc[0])
etTwo = sp.str2et(utc[1])
times = [x*(etTwo-etOne)/step + etOne for x in range(step)]
positions, lightTimes = sp.spkpos('Maven', times, 'J2000', 'NONE', 'MARS BARYCENTER')


# load simulation data
fdir = '/Volumes/triton/Data/ModelChallenge/MGITM/'
fname = 'MGITM_LS180_F070_150615.nc'
ds = load_mgitm(fdir=fdir, fname=fname)

# Get orbits in simulation
ray_path = [ds.ray(positions[i], positions[i+1]) for i in range(positions.shape[0]-1)]
ray_lens = np.array([ray["t"].shape for ray in ray_path])
ray_idx = np.r_[0, np.cumsum(ray_lens)]
dat_path = np.zeros(np.sum(ray_lens))
dat_pos = np.zeros_like(dat_path)

Nrays = float(positions.shape[0]-1)
for i, ray in enumerate(ray_path):
    t = ray["t"]
    sort = np.argsort(t)

    dat_path[ray_idx[i]:ray_idx[i+1]] = ray["n_e"][sort]
    dat_pos[ray_idx[i]:ray_idx[i+1]] = t[sort]*(times[i+1]-times[i])+times[i]

plt.plot(dat_pos, dat_path)
plt.semilogy()
plt.savefig('Output/orbit_density.pdf')


#   # Setup axes
#   import matplotlib.gridspec as gridspec
#   fig = plt.figure()
#   gs = gridspec.GridSpec(3,3)
#   ax1 = plt.subplot(gs[0,:])
#   ax2 = plt.subplot(gs[1:,:])
#
#
#   # Make a slice plot
#   slc = yt.SlicePlot(ds, 'x', ('Hesw','Density'))
#   slc.set_zlim(('Hesw', 'Density'), 2E2, 1E-2)
#   slc.set_xlabel('$\mathrm{y\;(km)}$')
#   slc.set_ylabel('$\mathrm{x \;(km)}$')
#   slc.set_colorbar_label(('Hesw', 'Density'),'$\mathrm{He\;Density}$')
#   for ray in ray_path: 
#       slc.annotate_ray(ray, plot_args={'color':'k', 'lw':2})
#
#   slc.save('Output/heliosares_orbit.pdf')
#
#   plot = slc.plots[("Hesw", "Density")]
#   plot.figure = fig
#   plot.axes=ax2
#   #grid.cbar_axes[1]
#
#   slc._setup_plots()
#
#   ax1.plot(dat_pos, dat_path)
#   #ax.set_yscale('log')
#
#
#   #plt.savefig('Output/heliosares_orbit.pdf')
