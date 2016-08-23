import yt
from load_heliosares import load_heliosares
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import spiceypy as sp

fields = [(s, "Density") for s in [#['CO2pl', 'Hesw',  'Hsw',   'O2pl', 'Thew',
                      'Hpl', 'Opl']]
radius = 4500
all_lims = {'Thew':(10E-4, 5E4),
        'CO2pl':(1E-2, 1E4),
        'Hesw':(1E-3, 2E2),
        'Hsw':(1E-3, 1E2),
        'O2pl':(1E-4, 1E5),
        'Hpl':(1E-4, 5E1),
        'Opl':(1E-2, 5E3)}



# load simulation data

lat_range = np.linspace(-np.pi, np.pi, 400)
lon_range = np.linspace(-np.pi/2, np.pi/2, 200)
lat_mesh, lon_mesh = np.meshgrid(lat_range, lon_range)
pts_geo = np.zeros((3, lat_mesh.size))
pts_geo[0, :] = radius
pts_geo[1, :] = lat_mesh.flatten() 
pts_geo[2, :] = lon_mesh.flatten() 

pts = np.zeros_like(pts_geo)
sp.furnsh("maven_spice.txt")
for i in range(lat_mesh.size):
    pts[[2,1,0], i] = sp.spiceypy.latrec(pts_geo[0, i], pts_geo[2,i], pts_geo[1, i])

for field in fields:
    lims = all_lims[field[0]]
    ds = load_heliosares('/Volumes/triton/Data/ModelChallenge/Heliosares/test/', subset=[field[0]])
    print 'Plotting field: ', field

    vals = ds.find_field_values_at_points(field, pts.T)
    vals = vals.reshape((lat_range.shape[0], lon_range.shape[0]))
    print vals
    if lims is not None:
        pcol = plt.pcolormesh(lat_range, lon_range, vals.T,cmap='arbre',  norm=LogNorm(vmin=lims[0], vmax=lims[1]), rasterized=True)
    else:
        pcol = plt.pcolormesh(lat_range, lon_range, vals.T,cmap='arbre', rasterized=True)
    plt.axis([lat_range.min(), lat_range.max(), lon_range.min(), lon_range.max()])
    plt.colorbar()
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["$-\pi$","$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"])
    plt.yticks([ -np.pi/2, 0, np.pi/2], [ "$-\pi/2$", "$0$", "$\pi/2$"])
    #pcol.set_edgecolor('face')
    plt.savefig('Output/spherical_slice_{0}.pdf'.format(field[0]), dpi=600)
    plt.close()



    for  ax in ['x', 'y', 'z']:
        slc = yt.SlicePlot(ds, ax, field)

        if lims is not None:
            slc.set_zlim(field, lims[0], lims[1])
        #slc.set_xlabel('$\mathrm{y\;(km)}$')
        #slc.set_ylabel('$\mathrm{x \;(km)}$')
        #slc.set_colorbar_label(('Hesw', 'Density'),'$\mathrm{He\;Density}$')
        slc.annotate_sphere([0,0,0], radius, circle_args={'lw':2, 'color':'k'})
        slc.annotate_line((0,0,0), (0,0,2000), coord_system='data', plot_args={'color':'k'})

        slc.save('Output/slice_annotated_{0}_{1}.pdf'.format(ax, field[0]))


    del ds



