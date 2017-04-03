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
plt.style.use('seaborn-poster')

def setup_plot(fields, ds_names, coords, tlimit=None, add_altitude=False):
    """
    Setup plotting environment and corresponding data structures
    """
    if add_altitude: 
        fields = fields[:]
        fields.insert(0,'altitude')
    Nfields = len(fields)
        
    hrs = [1 for i in range(Nfields)]
        
    hrs.insert(0,0.1)
    hrs.insert(0,0.1)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(Nfields+2, 1,
                           height_ratios=hrs, hspace=0.05)
    axes = [plt.subplot(gs[i, 0]) for i in range(2, Nfields+2)]
    f = plt.gcf()
    
    #f, axes = plt.subplots(len(fields), 1)
    colors = {'maven':'k', 
              'maven1':'k', 
              'maven2':'r', 
              'bats_min_LS270_SSL0':'CornflowerBlue',
              'bats_min_LS270_SSL180':'DodgerBlue',
              'bats_min_LS270_SSL270':'LightSkyBlue',
              'batsrus_multi_species':'MediumBlue',
              'batsrus_multi_fluid':'DodgerBlue',
              'batsrus_electron_pressure':'LightSeaGreen',
              'heliosares':'MediumVioletRed',
              'helio_1':'LimeGreen',
              'helio_2':'ForestGreen'}

    for i in range(550,660,10): colors['t00{0}'.format(i)] = cm.rainbow((i-550)/10.0)
    
    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes)}
    plot['kwargs'] = {ds:{ 'label':label_lookup[ds], 'color':colors[ds], 'lw':1.5}
                        for ds in ds_names}
   # plot['kwargs']['maven']['alpha'] = 1#0.6
#    plot['kwargs']['maven']['lw'] = 1
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = Nfields #len(fields)
    plot['shadowbar'] = plt.subplot(gs[0,0])
    plot['timebar'] = plt.subplot(gs[1,0])
    plot['tlimit'] = tlimit
    plot['shadow'] = np.logical_and(coords[0]<0,
                                    np.sqrt(coords[1]**2+coords[2]**2)<3390)
    return plot

def finalize_plot(plot, xlim=None, fname=None, show=False, zeroline=False):
    """
    Make final plotting adjustments and save/show image
    """
    if 'altitude' in plot.keys():
        plot['axes']['altitude'].plot(np.linspace(plot['time'][0], plot['time'][-1],
                                                  plot['altitude'].shape[0]), 
                                                  plot['altitude'],
                                                  **plot['kwargs']['maven'])
    for f, ax in plot['axes'].items():
        ax.set_ylabel(label_lookup[f])
        if zeroline:
            ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyle=':', alpha=0.4)
        if f in field_lims: ax.set_ylim(field_lims[f])
        if f in log_fields2: ax.set_yscale('log')
            
    for i in range(plot['N_axes']):
        ax = plot['ax_arr'][i]
        if i == plot['N_axes']-1: ax.set_xlabel('$\mathrm{Time}$')
        else: ax.set_xticks([])
        
        if plot['tlimit'] is not None:
            #lim, t = plot['tlimit'], plot['time']
            #tlim = (t[int(lim[0]*t.shape[0])], t[int(lim[1]*t.shape[0])])
            tlim = plot['tlimit']
            ax.set_xlim(tlim)
        else:
            ax.set_xlim(0,1)
            
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
    
    tb.set_xlim(tlim)
    sb.set_xlim(tlim)
    
    tb.axis('off')
    sb.axis('off')
    
    #plot['ax_arr'][-1].legend()#(bbox_to_anchor=(1.4, 1))
    handles, labels = plot['ax_arr'][-1].get_legend_handles_labels()
    #plot['ax_arr'][0].legend(handles, labels)
    plot['ax_arr'][0].set_zorder(1)
    #plot['figure'].set_size_inches(8,10)
    plot['figure'].set_size_inches(8, 16)
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
                                    grid=ds_type=='heliosares')
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
                         coords=None,  subtitle=None, tlimit=None, **kwargs):
    """
    Main function for creating plot, must have already found
    data
    """
    plot = setup_plot(fields, ds_names.keys(), coords,
                      tlimit=tlimit, add_altitude=True)

    for field in fields:
        for dsk, ds_dat in data[field].items():
                        
            if ds_dat.size != 0:
                plot_field_ds(data['time'][dsk], ds_dat, plot['axes'][field], plot['kwargs'][dsk])
            else:
                plot_field_ds(np.array([0]),np.array([0]),
                              plot['axes'][field], plot['kwargs'][dsk])

    finalize_plot(plot, zeroline=True, fname='Output/{0}_{1}.pdf'.format(title, subtitle), **kwargs)
    print 'Saving: ', 'Output/{0}_{1}.pdf'.format(title, subtitle)
    

def flythrough_orbit(orbits, ds_names, ds_types, field, **kwargs):
    """
    Setup an orbit, find the appropriate date, and make
    a flythrough plot.
    """

    #coords, times = get_path_pts(trange, Npts=150)
    if orbits[0].isdigit():
        coords, idx = get_orbit_coords(int(orbits[0]), Npts=250, return_idx=True)
    elif orbits[0] == 'z':
        z = np.linspace(1, 2, 250)
        coords = np.array([np.zeros_like(z), np.zeros_like(z), z])
        idx = []
    else:
        print 'Orbit not supported'
        raise(RuntimeError)

    indxs = get_path_idxs(coords, ds_names, ds_types)
    indxs['maven'] = idx
    

    if field == 'low_ion':
        fields =['H_p1_number_density',
                'O2_p1_number_density',
                'O_p1_number_density',
                'CO2_p1_number_density'] 
        tlimit = (0.43, 0.57)
        title = 'low_alt_ion'
        skip = 1
    elif field == 'plume':
        fields =['O2_p1_number_density',
                 'O2_p1_velocity_xy',
                 'O2_p1_velocity_z',
                 'O_p1_number_density',
                 'O_p1_velocity_xy',
                 'O_p1_velocity_z',
                 'CO2_p1_number_density',
                 'CO2_p1_velocity_xy',
                 'CO2_p1_velocity_z',
                 'H_p1_velocity_xy',
                 'H_p1_velocity_z',
                 ]
        tlimit =(0.2, 0.45)
        title = 'plume'
        skip = 1

    elif field == 'mag':
        fields = ['magnetic_field_total', 'magnetic_field_x', 'magnetic_field_y',
              'magnetic_field_z']
        tlimit = None#(0.3, 0.7)
        title = 'mag'
        skip = 1
    else:
        fields = [field]
        tlimit = None
        title = field
        skip=1

    data = get_all_data(ds_names, ds_types, indxs, fields)
    make_flythrough_plot(fields, data, ds_names, coords=coords,  
                         subtitle='{0}_{1}'.format(title,orbits[0]),  tlimit=tlimit)


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"f:o:nh",["field=", "orbit=", "new_models", "helio_models"])
    except getopt.GetoptError:
        print 'error'
        return


    field, orbit, new_models, helio_models = None, None, False, False

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

    if orbit == 371:
        orbit_groups = np.array([353, 360, 363, 364, 364, 365, 366, 367, 367, 368, 369, 370, 371,375, 376, 376, 380, 381, 381, 382, 386, 386, 387, 390, 391])
    
    # Get Datasets setup
    ds_names, ds_types = get_datasets(R2349=new_models, helio_multi=helio_models)
    print ds_names, ds_types

    flythrough_orbit([orbit], ds_names, ds_types, field)


if __name__ == "__main__":
    main(sys.argv[1:])
