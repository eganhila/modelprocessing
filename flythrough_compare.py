import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp
import h5py
from general_functions import *
plt.style.use('seaborn-talk')

def setup_plot(fields, ds_names):
    f, axes = plt.subplots(len(fields), 1)
    colors = {'maven':'k', 
              'bats_min_LS270_SSL0':'CornflowerBlue',
              'bats_min_LS270_SSL180':'DodgerBlue',
              'bats_min_LS270_SSL270':'LightSkyBlue',
              'helio_run1':'LimeGreen',
              'helio_run2':'ForestGreen'}
    
    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes)}
    plot['kwargs'] = {ds:{ 'label':ds, 'color':colors[ds], 'lw':1.5}
                        for ds in ds_names}
    plot['kwargs']['maven']['alpha'] = 0.6
    plot['kwargs']['maven']['lw'] = 1
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = len(fields)


    return plot

def finalize_plot(plot, xlim=None, fname=None, show=False, zeroline=False):
    for f, ax in plot['axes'].items():
        ax.set_ylabel(label_lookup[f])
        if zeroline:
            ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyle=':', alpha=0.4)
        ax.set_yscale('symlog', linthreshy=10)
    for i in range(plot['N_axes']):
        ax = plot['ax_arr'][i]
        if i == plot['N_axes']-1:
            ax.set_xlabel('Time')
        else:
            ax.set_xticks([])
        #ax.set_ylabel(ax.get_ylabel(), labelpad=30*(i%2))    
        if xlim is not None:
            ax.set_xlim(xlim)
        
    plot['ax_arr'][0].legend()#(bbox_to_anchor=(1.4, 1))
    plot['ax_arr'][0].set_zorder(1)
    plt.gcf().set_size_inches(10,10)

    if show:
        plt.show()
    if fname is None:
        plt.savefig('Output/test.pdf')
    else:
        plt.savefig(fname)

def get_path_idxs(coords, ds_names, ds_types):
    indxs = {}
    for ds_type, keys in ds_types.items():
        if ds_type == 'maven': continue
        print 'getting indxs: '+ds_type
        indxs[ds_type] = bin_coords(coords, ds_names[keys[0]])
    indxs['maven'] = 'all'
    return indxs

def plot_field_ds(x, data, ax, kwargs):
    if data.ndim<2:
        ax.plot(x, data, **kwargs)
    else:
        mean_dat = np.nanmedian(data,axis=0)
        max_dat = np.nanmax(data, axis=0)
        min_dat = np.nanmin(data, axis=0)
        ax.plot(x, mean_dat, **kwargs)
        ax.fill_between(x, min_dat, max_dat, alpha=0.2, color=kwargs['color'])


def make_plot(times, fields, orbits, title, indxs, ds_names, ds_types, skip=1):
    plot = setup_plot(fields, ds_names.keys())

    for ds_type, keys in ds_types.items():
        for key in keys:
            dsf = ds_names[key]

            for field in fields:
                with h5py.File(dsf, 'r') as ds:
                    ds_dat = get_ds_data(ds, field, indxs[ds_type])
                        
                    if ds_dat.size != 0:
			plot_field_ds(times-times[0], ds_dat, plot['axes'][field], plot['kwargs'][key])
		    else:
			plot_field_ds(np.array([0]),np.array([0]),
                                      plot['axes'][field], plot['kwargs'][key])
    for field in fields:
	dsf = ds_names['maven'] 
	mav_data = []

	for i, orb in enumerate(orbits):
	    with h5py.File(dsf.format(orb), 'r') as ds:
		dat = get_ds_data(ds, field, 'all')
		if np.sum(dat.shape) != 0: mav_data.append(dat)     

	if len(mav_data) == 0: 
	    plot_field_ds(np.array([0]), np.array([0]), plot['axes'][field], plot['kwargs']['maven'])
	    continue

	L = min([d.shape[0] for d in mav_data])
	data = np.zeros((len(mav_data),L))
	for i in range(len(mav_data)):
	    data[i] = mav_data[i][:L]
	t = np.linspace(times[0], times[-1], data.shape[1])-times[0]
	plot_field_ds(t[::skip], data[:,::skip], plot['axes'][field], plot['kwargs']['maven'])

    finalize_plot(plot, zeroline=True, fname='Output/{0}_{1}.pdf'.format(title, orbits[0]))
    


def flythrough_orbit(orbits, trange, ds_names, ds_types):

    coords, times = get_path_pts(trange, Npts=150)
    indxs = get_path_idxs(coords, ds_names, ds_types)

    ion_fields =['H_p1_number_density',
          'O2_p1_number_density',
          'O_p1_number_density',
          'O_p2_number_density',
          'CO2_p1_number_density'] 
    mag_fields = ['magnetic_field_radial',
          'magnetic_field_x', 'magnetic_field_y',
          'magnetic_field_z']
    make_plot(times, ion_fields, orbits, 'ion_flythrough', indxs, ds_names, ds_types)
    make_plot(times, mag_fields, orbits, 'mag_flythrough', indxs, ds_names, ds_types, skip=20)


def main():

    orbit_groups = [np.array([415,412,397,363])]

    # Get Datasets setup
    ds_names, ds_types = get_datasets()

    for orbits in orbit_groups:
        tranges = get_orbit_times(orbits)

        flythrough_orbit(orbits, tranges[:,0], ds_names, ds_types)





if __name__ == "__main__":
    main()
