"""
Makes plots that fly through selected models for a
fields/group of fields.

Inputs:
    --field (-f): field to flythrough. Can also put "mag" to
        flythrough all magnetic field fields or "ion" for all
        ion fields
    --orbit (-o): orbit to flythrough. Integer to pick a single
        orbit, or predefine a group like G1
    --new_models (-n): Use the new models, flag passed to get
        datasets
    --helio_multi (-h): use the multi helio models

"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
from general_functions import *
import sys
import getopt
from matplotlib import cm
import pandas as pd
import datetime
plt.style.use('seaborn-poster')

def setup_plot(fields, ds_names, coords, tlimit=None, add_altitude=False, single_out=None):
    """
    Setup plotting environment and corresponding data structures
    """
    if add_altitude and False:
        fields = fields[:]
        fields.insert(0,'altitude')
    Nfields = len(fields)

    hrs = [1 for i in range(Nfields)]

    hrs.insert(0,0.3)
    hrs.insert(0,0.1)
    hrs.insert(0,0.1)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(Nfields+3, 1,
                           height_ratios=hrs, hspace=0.05, wspace=3)
    axes = [plt.subplot(gs[i, 0]) for i in range(3, Nfields+3)]
    f = plt.gcf()

    #f, axes = plt.subplots(len(fields), 1)
    colors = {'maven':'k',
              'maven_low_alt':'k',
              'maven_plume':'k',
              'rhcsv':'red',
              'bats_min_LS270_SSL0':'CornflowerBlue',
              'bats_min_LS270_SSL180':'DodgerBlue',
              'bats_min_LS270_SSL270':'LightSkyBlue',
              'batsrus_multi_species':'MediumBlue',
              'batsrus_multi_fluid':'DodgerBlue',
              'batsrus_electron_pressure':'LightSeaGreen',
              'heliosares': 'MediumVioletRed',
              'rhybrid':'orchid',
              'helio_1':'LimeGreen',
              'helio_2':'ForestGreen'}

    for i in range(550,660,10): colors['t00{0}'.format(i)] = cm.rainbow((i-550)/10.0)

    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes)}
    plot['kwargs'] = {ds:{ 'label':label_lookup[ds], 'color':colors[ds], 'lw':1.5}
            for ds in ds_names }

    if single_out is not None:
        for ds in plot['kwargs'].keys():
            if ds != single_out: plot['kwargs'][ds]['alpha']=0.2

    plot['kwargs']['rhybrid']['alpha']=0



    #plot['kwargs']['maven']['alpha'] = 0.6
    #plot['kwargs']['maven']['lw'] = 1
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = Nfields #len(fields)
    plot['shadowbar'] = plt.subplot(gs[0,0])
    plot['timebar'] = plt.subplot(gs[1,0])
    plot['tlimit'] = tlimit
    plot['shadow'] = np.logical_and(coords[0]<0,
                                    np.sqrt(coords[1]**2+coords[2]**2)<3390)
    plot['altitude'] = (np.sqrt(np.sum(coords**2,axis=0))-1)*3390
    return plot

def add_tbars(plot, reset_timebar):
    tb = plot['timebar']
    sb = plot['shadowbar']

    t_xv, t_yv = np.meshgrid(np.linspace(0,1, 100), [0,1])
    s_xv, s_yv = np.meshgrid(np.linspace(0, 1, plot['shadow'].shape[0]),[0,1])

    t_dat = np.array([np.linspace(0,1,100), np.linspace(0,1,100)])
    s_dat = np.array([plot['shadow'], plot['shadow']])

    tb.pcolormesh(t_xv, t_yv, t_dat, cmap='inferno',rasterized=True)
    sb.pcolormesh(s_xv, s_yv, s_dat, cmap='inferno_r',rasterized=True,
                  vmin=-0.1,vmax=1.2)

    if plot['tlimit'] is None: tlim = (0,1)
    else: tlim = plot['tlimit']

    if not reset_timebar:
        tb.set_xlim(tlim)
    sb.set_xlim(tlim)

    tb.axis('off')
    sb.axis('off')
    
    
def change_xticks(plot):
    ax = plot['ax_arr'][-1]
    tick_locs = ax.get_xticks().tolist()
    t = plot['time']
    alt = plot['altitude']
    
    alt_vals = [alt[np.argmin(np.abs(t-tick))] for tick in tick_locs]
    time_vals = [datetime.datetime.fromtimestamp(t*4.5*60*60+1450110477).strftime("%H:%M") for t in tick_locs]
    ax.set_xticklabels(['{1:02d}'.format(tv, int(av)) for tv, av in zip(time_vals, alt_vals)])
    ax.set_xticks(tick_locs)

    ax = plot['ax_arr'][0]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(time_vals[::2])
    ax2.set_xticks(tick_locs[::2])

    
def finalize_plot(plot, xlim=None, fname=None, show=False, zeroline=False, 
                    reset_timebar=False):
    """
    Make final plotting adjustments and save/show image
    """
    for f, ax in plot['axes'].items():
        if f in label_lookup:
            ax.set_ylabel(label_lookup[f])
        else:
            ax.set_ylabel(f)
        if zeroline:
            ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyle=':', alpha=0.4)
        if f in field_lims: ax.set_ylim(field_lims[f])
        if f in log_fields2: ax.set_yscale('log')

    for i in range(plot['N_axes']):
        ax = plot['ax_arr'][i]
        if i == plot['N_axes']-1: ax.set_xlabel('$\mathrm{Altitude}$')
        else: ax.set_xticks([])

        if plot['tlimit'] is not None: ax.set_xlim(plot['tlimit'])
        else: ax.set_xlim(0,1)

    add_tbars(plot, reset_timebar)
    change_xticks(plot)


    plot['figure'].set_size_inches(8,10)
    #plot['figure'].set_size_inches(8, 16)
    plot['figure'].subplots_adjust(left=0.2)

    if show:
        plt.show()
    elif fname is None:
        plt.savefig('Output/test.pdf')
    else:
        plt.savefig(fname)


def get_path_idxs(coords, ds_names, ds_types):
    indxs = {}
    for ds_type, keys in ds_types.items():
        if ds_type == 'maven': continue
        if len(keys) == 0: continue
        print 'getting indxs: '+ds_type
        indxs[ds_type] = bin_coords(coords, ds_names[keys[0]], 
                                    grid='helio' in ds_type)
                                    #grid=ds_type=='heliosares')
    indxs['maven'] = 'all'
    return indxs

def plot_field_ds(x, data, ax, kwargs):
    if data.ndim<2:
        ax.plot(x, data, **kwargs)
    else:
        mean_dat = np.nanmedian(data,axis=0)
        max1_dat = np.nanpercentile(data, 75, axis=0)
        min1_dat = np.nanpercentile(data, 25, axis=0)
        max0_dat = np.nanmax(data, axis=0)
        min0_dat = np.nanmin(data, axis=0)
        ax.plot(x, mean_dat, **kwargs)
        
        ax.fill_between(x, min1_dat, max1_dat, alpha=0.2, color=kwargs['color'])
        lim = ax.get_ylim()
        ax.fill_between(x, min0_dat, max0_dat, alpha=0.05, color=kwargs['color'])
        ax.set_ylim(lim)


def make_flythrough_plot(fields, data, ds_names, title='flythrough', 
                         coords=None,  subtitle=None, tlimit=None, single_out=None, **kwargs):
    """
    Main function for creating plot, must have already found
    data
    """
    plot = setup_plot(fields, ds_names.keys(), coords,
                      tlimit=tlimit, add_altitude=True, single_out=single_out)
    plot['time'] = data['time']['time']

    for field in fields:
        for dsk, ds_dat in data[field].items():
            if ds_dat.size != 0:
                plot_field_ds(data['time'][dsk], ds_dat, plot['axes'][field], plot['kwargs'][dsk])
            else:
                plot_field_ds(np.array([0]),np.array([0]),
                              plot['axes'][field], plot['kwargs'][dsk])

    finalize_plot(plot, zeroline=True, fname='Output/{0}_{1}.pdf'.format(title, subtitle), **kwargs)
    print 'Saving: ', 'Output/{0}_{1}.pdf'.format(title, subtitle)

def flythrough_orbit(orbits, ds_names, ds_types, field, region, **kwargs):
    """
    Setup an orbit, find the appropriate date, and make
    a flythrough plot.
    """
    #coords, times = get_path_pts(trange, Npts=150)
    if orbits[0].isdigit():
        coords, time = get_orbit_coords(int(orbits[0]), Npts=250, return_time=True)
    elif orbits[0] == 'z':
        z = np.linspace(1, 2, 250)
        coords = np.array([np.zeros_like(z), np.zeros_like(z), z])
        idx = []
    else:
        print 'Orbit not supported'
        raise(RuntimeError)

    coords = rotate_coords_simmso(coords)

    indxs = get_path_idxs(coords, ds_names, ds_types)
    indxs['maven'] = []


    if region == 'plume':
        tlimit = (0.3,0.45)
    elif region == 'low_alt':
        tlimit = (0.43, 0.57)
    elif region == 'shemi':
        tlimit = (0.55, 0.7) 
    else:
        tlimit = (0,1)
        region = 'all'

    if field == 'all_ion':
        fields =['H_p1_number_density',
                'O2_p1_number_density',
                'O_p1_number_density',
                'CO2_p1_number_density'] 
    elif field == 'O2':
        fields =['O2_p1_number_density',
                 'O2_p1_velocity_x',
                 'O2_p1_velocity_y',
                 'O2_p1_velocity_z',
                 'O2_p1_velocity_total']
    elif field == 'O':
        fields = ['O_p1_number_density',
             'O_p1_velocity_x',
             'O_p1_velocity_y',
             'O_p1_velocity_z',
             'O_p1_velocity_total']
    elif field == 'H':
        fields = ['H_p1_number_density',
             'H_p1_velocity_x',
             'H_p1_velocity_y',
             'H_p1_velocity_z',
             'H_p1_velocity_total']
    elif field == 'mag':
        fields = ['magnetic_field_x',
             'magnetic_field_y',
             'magnetic_field_z',
             'magnetic_field_total']
    elif field == 'plume1': fields  = ['O2_p1_number_density', 'O2_p1_velocity_total', 'O_p1_number_density', "O_p1_velocity_total"]
    elif field == 'plume2': fields  = ['O2_p1_number_density','O2_p1_velocity_x', 'O2_p1_velocity_y', 'magnetic_field_x']
    elif field == 'plume': fields = ['O_p1_number_density', 'O_p1_velocity_total', 'O2_p1_number_density', 'O_p1_velocity_total', 'O2_p1_velocity_x', 'O2_p1_velocity_y', 'magnetic_field_x']
    elif field == 'pressure':
        fields = ['magnetic_pressure', 'pressure', 'electron_pressure', 'total_pressure']
    elif field == 'current':
        fields = ['current_y', 'electron_pressure','pressure', 'hall_velocity_y', 'electron_velocity_y',  'velocity_y']
    else:
        fields = [field]

    data = get_all_data(ds_names, ds_types, indxs, fields)
    data['time'] = {dsk:time for dsk in ds_names.keys()}
    data['time']['time'] = time
    make_flythrough_plot(fields, data, ds_names, coords=coords,  
                         subtitle='{0}_{1}_{2}'.format(region, field,orbits[0]),  tlimit=tlimit, **kwargs)


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"f:o:nhz",["field=", "orbit=", "new_models", "helio_models", "region=", "reset_timebar", "single_out="])
    except getopt.GetoptError:
        print 'error'
        return


    field, orbit, new_models, helio_models, region, reset_timebar, single_out = None, None, False, False, None, False, None

    for opt, arg in opts:
        if opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--orbit"):
            orbit = arg
        elif opt in ("-n", "-new_models"):
            print 'Using new models'
            new_models=True
        elif opt in ('-h', '--helio_multi'):
            helio_models = True
        elif opt in ("--region"):
            region = arg
        elif opt in ("--reset_timebar", "-z"):
            reset_timebar = True
        elif opt in ("--single_out"):
            single_out = arg

    if orbit == 371:
        orbit_groups = np.array([353, 360, 363, 364, 364, 365, 366, 367, 367, 368, 369, 370, 371,375, 376, 376, 380, 381, 381, 382, 386, 386, 387, 390, 391])
    
    # Get Datasets setup
    ds_names, ds_types = get_datasets(R2349=new_models, helio_multi=helio_models)
    print ds_names, ds_types

    flythrough_orbit([orbit], ds_names, ds_types, field, region, reset_timebar=reset_timebar, single_out=single_out)


if __name__ == "__main__":
    main(sys.argv[1:])

