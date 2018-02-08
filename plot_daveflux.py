from spherical_flux import *
import numpy as np

fdat = np.loadtxt("IDL_Tools/flux_map.csv", delimiter=',').T
r = 1.8
xy, coords, rhat, area = create_sphere_mesh(r)


field = "total_flux"
fname = "Output/dave_flux.pdf"

create_plot(field, xy, fdat, r, fname=fname)