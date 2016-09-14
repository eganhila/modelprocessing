import numpy as np
import matplotlib.pyplot as plt

upstream_dat = np.loadtxt('Output/upstream.csv', delimiter=',', unpack=True)
orbit_times = np.loadtxt('Output/orbit_times.csv', delimiter=',', unpack=True)

nearest_orbit = np.zeros_like(upstream_dat[0])
nearest_time = np.zeros_like(upstream_dat[0])
for i in range(upstream_dat.shape[1]):
    idx = np.argmin(np.abs(upstream_dat[0, i]-orbit_times[1]))
    nearest_orbit[i] = orbit_times[0, idx] 
    nearest_time[i] = orbit_times[1, idx]

orb_dat = np.empty((upstream_dat.shape[0], orbit_times.shape[1]))
orb_dat[:] = np.NAN

for i, orb in enumerate(nearest_orbit):
    orb_dat[1:, int(orb)] = upstream_dat[1:, i]

orb_dat[0] = range(orb_dat.shape[1])
fmt = ['%.8e' for i in range(orb_dat.shape[0]-1)]
fmt = ['%04d'] + fmt

np.savetxt('Output/orbits_upstream.csv', orb_dat.T, fmt=fmt , delimiter=', ', newline='\n')


