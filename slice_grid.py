from sliceplot import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib import ticker
from general_functions import *
import getopt
import sys
import cmocean
import glob
import ast
from sliceplot_helper import *
import matplotlib.gridspec as gridspec


axes = {'x':2,'y':1,'z':0}
ds_names, ds_types = get_datasets('R2349', False)
regrid_data = ['batsrus_mf_lr', 'batsrus_electron_pressure', 'batsrus_multi_species'] 

def setup_slicegrid_plot(ds_keys, fields):
    plot = {}

    Nfields, Nds = len(fields), len(ds_keys)
    gs = gridspec.GridSpec(Nfields, Nds, hspace=0.05, wspace=0.05)

    
    axes = {}
    axes_grid = [[0 for i in range(Nds)] for j in range(Nfields)] 
    for i, dsk in enumerate(ds_keys):
        for j, f in enumerate(fields):
            ax = plt.subplot(gs[j,i])
            axes[(dsk, f)] = ax
            axes_grid[j][i] = ax

    axes_grid = np.array(axes_grid)


    plot['figure'] = plt.gcf()
    plot['axes'] = axes
    plot['axes_grid'] = axes_grid
    plot['Nds'] = Nds
    plot['Nfields'] = Nfields
    plot['ds_keys'] = ds_keys
    plot['fields'] = fields

    return plot


def finalize_slicegrid_plot(plot, ax_plt, center):
    ax_labels = [['Y','Z'],['X','Z'],['X','Y']]

    for ax_i in range(plot['Nds']):
        for ax_j in range(plot['Nfields']):
            ax = plot['axes_grid'][ax_j,ax_i]
            ax.set_aspect('equal')

            mars_frac = np.real(np.sqrt(1-center[ax_plt]**2))
            alpha = np.nanmax([mars_frac, 0.1])
            add_mars(ax_plt, ax=ax, alpha=alpha)

           
            if ax_i == plot['Nds'] -1:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(label_lookup[plot['fields'][ax_j]])
            if ax_j == 0:
                ax.set_title(label_lookup[plot['ds_keys'][ax_i]])

            if ax_j != plot['Nfields']-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('$\mathrm{'+ax_labels[ax_plt][0]+'} \;(R_M)$')

            if ax_i != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('$\mathrm{'+ax_labels[ax_plt][1]+'} \;(R_M)$')

    plt.savefig('Output/test.pdf')


def make_slicegrid_plot(fields, ds_keys, ax, center):

    plot = setup_slicegrid_plot(ds_keys, fields)

    for dsk in ds_keys:
        ds_name = ds_names[dsk]

        for field in fields:
            ds = load_data(ds_name,field=field)
            slc = slice_data(ds, ax, field, regrid_data=dsk in regrid_data, center=center)
            plot_data(plot['axes'][(dsk, field)], slc, ax, field, cbar=False)

    finalize_slicegrid_plot(plot, ax, center)


def main():

    fields = ['O2_p1_number_density', 'O2_p1_v_cross_B_z']
    ds_keys = ['batsrus_mf_lr', 'batsrus_electron_pressure', 'rhybrid']
    ax = 'x'
    center = [0.0,0.0,0.0]



    make_slicegrid_plot(fields, ds_keys, axes[ax], center)

if __name__ == '__main__':
    main()
