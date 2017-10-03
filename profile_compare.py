"""
Similar to flythrough compare but with emphasis on
linear profiles
"""

import numpy as np
import matplotlib.pyplot as plt
from general_functions import *
import sys
import getopt
from flythrough_compare import *
plt.style.use('seaborn-poster')


def setup_profile_plot(fields, ds_names):
    Nfields = len(fields)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(Nfields, 1, hspace=0.05, wspace=3)
    axes = [plt.subplot(gs[i, 0]) for i in range(Nfields)]
    f = plt.gcf()

    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes)}
    plot['kwargs'] = {ds:{'lw':1.5} for ds in ds_names}
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = Nfields

    for ds in ds_names: 
        if ds in colors: plot['kwargs'][ds]['color']=colors[ds]

    return plot

def finalize_profile_plot(plot):
    
    adjust_field_axes(plot, True)

    for ax in plot['ax_arr']:
        ax.set_xscale('log')


    plot['ax_arr'][-1].set_xlabel('Altitude (km)')
    plt.savefig('Output/test.pdf')


def plot_profile(xvals, data, fields, ds_names):

    plot = setup_profile_plot(fields, ds_names)

    for field in fields:
        for dsk, ds_dat in data[field].items():

            if ds_dat.size != 0:
                plot_field_ds(xvals, ds_dat, plot['axes'][field], plot['kwargs'][dsk])

    finalize_profile_plot(plot)

def create_profile():
    ds_names_all, ds_types_all = get_datasets(load_key='R2349') 
    ds_keys = ['batsrus_mf_lr', 'batsrus_electron_pressure', 'rhybrid']

    ds_names = {dsk:dsn for dsk, dsn in ds_names_all.items() if dsk in ds_keys}
    ds_types = {dsk:[dsk] for dsk in ds_keys}

    R = np.logspace(np.log10((100+3390)/3390.0), np.log10(5), 100)

    #theta = np.linspace(-np.pi/6, np.pi/6, 15)
    theta = np.array([55*np.pi/180]) 

    fields = ['O2_p1_velocity_z', 'O2_p1_velocity_x', 'O2_p1_v_cross_B_z', 'O2_p1_v_cross_B_x']



    data = {f:{dsk:np.zeros((theta.shape[0], R.shape[0])) for dsk in ds_names.keys()} for f in fields}

    for i, theta_i in enumerate(theta):
        coords = np.array([ np.sin(theta_i)*R, np.zeros_like(R),  np.cos(theta_i)*R])
        indxs = get_path_idxs(coords, ds_names, ds_types)

        data_i = get_all_data(ds_names, ds_types, indxs, fields)

        for f in fields:
            for dsk in ds_names.keys():
                if data_i[f][dsk].size != 0:
                    data[f][dsk][i] = data_i[f][dsk]

    plot_profile((R-1)*3390, data, fields,  ds_names)

if __name__ == '__main__':
    create_profile()
