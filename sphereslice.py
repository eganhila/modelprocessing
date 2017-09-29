from sliceplot import *
from general_functions import *
from slice_grid import setup_slicegrid_plot as setup_spslc_plot
import numpy as np
import matplotlib.pyplot as plt

ds_names, ds_types = get_datasets('temp', False)


def finalize_spslc_plot():
    for ax_i in range(plot['Nds']):
        for ax_j in range(plot['Nfields']):
            ax = plot['axes_grid'][ax_j,ax_i]

            if ax_i == plot['Nds'] -1:
                ax.set_ylabel(label_lookup[plot['fields'][ax_j]])
                ax.yaxis.set_label_position("right")
            if ax_j == 0:
                ax.set_title(label_lookup[plot['ds_keys'][ax_i]])

            if ax_j != plot['Nfields']-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('$\mathrm{Latitude}$')

            if ax_i != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('$\mathrm{Altitude} \;(km)$')


    h = 15.0
    w = plot['Nds']*h/plot['Nfields'] 

    plot['figure'].set_size_inches(w,h)
    print 'Saving: '+fname
    plt.savefig(fname)

def create_spslc_grid():
    
    lat = np.linspace(90,0,25)
    alt = (np.logspace(np.log10((100+3390)/3390.0), np.log10(5), 60)-1)*3390

    lat_grid, alt_grid = np.meshgrid(lat, alt)

    slc_shape = lat_grid.shape

    lat_grid = lat_grid.flatten()
    alt_grid = alt_grid.flatten()

    R = alt_grid/3390+1
    phi = lat_grid*np.pi/180.0

    x = R*np.cos(phi) 
    y = np.zeros_like(x)
    z = R*np.sin(phi)

    coords = np.array([x,y,z])

    return (coords, (lat_grid, alt_grid), slc_shape)


def main():

    fields ['O2_p1_v_cross_B_z']
    ds_keys = ['batsrus_mf_lr']

    plot = setup_spslc_plot(fields, ds_keys)

    coords, bins, shape = create_spslc_grid()
    indxs = get_path_idxs(coords, ds_names, ds_types)
    data = get_all_data(ds_names, ds_types, indxs, fields)

    for dsk in ds_keys:
        for f in fields:
            plot_ax = plot['axes'][(dsk, f)]
            slc = (bins[0], bins[1], data[dsk][f])

            plot_data_scalar(plot_ax, slc, None, f, cbar=False)  

    finalize_spslc_plot(plot)


if __name__ == '__main__':
    main()
