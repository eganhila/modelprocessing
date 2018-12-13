from modelprocessing.sliceplot import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib import ticker
from modelprocessing.general_functions import *
import getopt
import sys
import glob
import ast
from modelprocessing.sliceplot_helper import *
import matplotlib.gridspec as gridspec
import matplotlib as mpl
plt.style.use('seaborn-poster')
#mpl.rcParams['text.usetex'] = False
#mpl.rcParams['axes.labelsize'] = 36 
#mpl.rcParams['xtick.labelsize'] = 36
#mpl.rcParams['ytick.labelsize'] = 36


north_ax = {'v':"E", 'E':"Bperp", "Bperp":"E"}
centers = {'v':[-3.5,0,0], "E":[0.0,0.0,0.0], "Bperp":[0,0,0]}
vecs = {'nominal':{'v':[1,0,0],
                'E':[0,0.17,0.98],
                'Bperp':[0,0.98,0.18]},
        'qparallel':{'v':[1,0,0],
              'E':[0,0.95,0.28],
              'Bperp':[0,-0.28,-0.95]}}
vecs['euv'] = vecs['qparallel']
vecs['pbeta'] = vecs['qparallel']
vecs['sw_str'] = vecs['qparallel']

axes = {'x':0,'y':1,'z':2}
ds_names, ds_types = get_datasets('exo_comparisonA', False)
regrid_data = ['batsrus_mf_lr', 'batsrus_electron_pressure', 'batsrus_multi_species'] 

def get_offax_data(ds_names, fields, ax, center=None, use_gfields=False):

    all_dat = {}

    if center is None: center = centers[ax]
    for dsk, ds_name in ds_names.items():
        all_dat[dsk] = {}

        if use_gfields: load_fields = fields
        else: load_fields = "all"

        ds = yt_load(ds_name, fields=load_fields)


        slc = yt.OffAxisSlicePlot(ds, vecs[dsk][ax], fields, center,
                north_vector=vecs[dsk][north_ax[ax]])

        frb = slc.data_source.to_frb((8, "code_length"), 512)
        for f in fields:
            all_dat[dsk][f] = frb[f].d[::-1]

    plt.clf()

    return all_dat


def plot_offax_grid(ds_names, ds_keys,  fields, all_dat,ax,add_mars_ax=None,
                    fname=None, normalize=False, override_lims=None, show=False,
                    override_cmaps=None):
    plot = setup_slicegrid_plot(ds_keys, fields)
    base_fields = fields
    if normalize:
        fields = [f+'_normalized' for f in base_fields]
    if override_lims is None: override_lims = {}
    if override_cmaps is None: override_cmaps = {}
    if add_mars_ax is None: add_mars_ax=ax

    for dsk in ds_names.keys():
        for bf, f in zip(base_fields,fields):
            
            if f in override_lims.keys(): olims = override_lims[f] #[IC]
            else: olims = None

            if normalize:
                with h5py.File(ds_names[dsk]) as ds:
                    data = all_dat[dsk][bf]/ds.attrs[normalize]
            else:
                data = all_dat[dsk][bf]
                data[data==0]=np.nan

            norm, cmap, tick_locations, symlogscale = apply_scalar_lims(f, data,  override_lims=olims)
            
            if f in override_cmaps.keys(): cmap = override_cmaps[f]
            
            im = plot['axes'][(dsk,f)].imshow(data, extent=[-4,4,-4,4],
                    cmap=cmap, norm=norm)

    finalize_slicegrid_plot(plot, 0,centers[ax], fname=fname,  add_mars_ax=add_mars_ax, show=show)
    plt.show()
    plt.clf()


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


def finalize_slicegrid_plot(plot, ax_plt, center, fname=None, boundaries=False, show=False,
                           add_mars_ax=None):
    ax_labels = [['Y','Z'],['X','Z'],['X','Y']]
    if add_mars_ax is None: add_mars_ax = ax_plt

    for ax_i in range(plot['Nds']):
        for ax_j in range(plot['Nfields']):
            ax = plot['axes_grid'][ax_j,ax_i]
            ax.set_aspect('equal')
            ax.set_xlim(-4,4)
            ax.set_ylim(-4,4)

            mars_frac = np.real(np.sqrt(1-center[ax_plt]**2))
            alpha = np.nanmax([mars_frac, 0.1])
            add_mars(add_mars_ax, ax=ax, alpha=alpha)
            if boundaries: add_boundaries(ax, ax_plt, center)

           
            if ax_i == plot['Nds'] -1:
                if plot['fields'][ax_j] in label_lookup.keys():
                    ax.set_ylabel(label_lookup[plot['fields'][ax_j]])
                else:
                    ax.set_ylabel(plot['fields'][ax_j])
                ax.yaxis.set_label_position("right")
            if ax_j == 0:
                if plot['ds_keys'][ax_i] in label_lookup.keys():
                    ax.set_title(label_lookup[plot['ds_keys'][ax_i]])
                else:
                    ax.set_title(plot['ds_keys'][ax_i])
            if ax_j != plot['Nfields']-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('$\mathrm{'+ax_labels[ax_plt][0]+'} \;(R_P)$')

            if ax_i != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('$\mathrm{'+ax_labels[ax_plt][1]+'} \;(R_P)$')


    h = 15.0
    w = plot['Nds']*h/plot['Nfields'] 

    plot['figure'].set_size_inches(w,h)
    if show:
        plt.show()
    else:
        print('Saving: '+fname)
        plt.savefig(fname)


def make_slicegrid_plot(fields, ds_keys, ax, center, fname='Output/test.pdf'):

    plot = setup_slicegrid_plot(ds_keys, fields)

    for dsk in ds_keys:
        ds_name = ds_names[dsk]

        for field in fields:
            ds = load_data(ds_name,field=field)
            slc = slice_data(ds, ax, field, regrid_data=dsk in regrid_data, center=center)
            plot_data(plot['axes'][(dsk, field)], slc, ax, field, cbar=False)

    finalize_slicegrid_plot(plot, ax, center, fname)#, boundaries=True)


def main():

    fields = ['H_p1_number_density', "O_p1_number_density","O2_p1_number_density", "magnetic_field_total"]
    ds_keys = ['2349_1RM_225km', '2349_2RM_450km',
                '2349_4RM_900km'] 
    #ds_keys = ds_names.keys()
    ax = 'z'
    center = [-2,0.0,0.0]
#    for i, x in enumerate([1,0.5,-0.5,-1.25,-2]):
#        print i
#        center = [x,0,0]

    make_slicegrid_plot(fields, ds_keys, axes[ax], center, fname='Output/slicegrid.pdf')#_{0:02d}.pdf'.format(i))

if __name__ == '__main__':
    main()
