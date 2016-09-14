import numpy as np
import matplotlib.pyplot as plt

upstream_dat = np.loadtxt('Output/orbits_upstream.csv', delimiter=', ', unpack=True)
orbits = np.loadtxt('/Volumes/triton/Data/maven/orbit_plots/final_orbits/orbitN.dat', unpack=True)

param = 7

plt.plot(upstream_dat[0], upstream_dat[param])
plt.scatter(orbits, upstream_dat[param, orbits.astype(int)])
plt.xlim(min(orbits), max(orbits))
plt.show()
