import numpy as np
import matplotlib.pyplot as plt
import h5py
from general_functions import *
plt.style.use('seaborn-talk')

def setup_plot(fields, ds_names, coords, tlimit=None):
    hrs = [1 for i in range(len(fields))]
    hrs.insert(0,0.1)
    hrs.insert(0,0.1)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(len(fields)+2, 1,
                           height_ratios=hrs)
    axes = [plt.subplot(gs[i, 0]) for i in range(2, len(fields)+2)]
    f = plt.gcf()
    
    #f, axes = plt.subplots(len(fields), 1)
    colors = {'maven':'k', 
              'bats_min_LS270_SSL0':'CornflowerBlue',
              'bats_min_LS270_SSL180':'DodgerBlue',
              'bats_min_LS270_SSL270':'LightSkyBlue',
              'batsrus_3dmhd':'CornflowerBlue',
              'helio_1':'LimeGreen',
              'helio_2':'ForestGreen'}
    
    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes)}
    plot['kwargs'] = {ds:{ 'label':ds.replace('_', ' '), 'color':colors[ds], 'lw':1.5}
                        for ds in ds_names}
    plot['kwargs']['maven']['alpha'] = 0.6
    plot['kwargs']['maven']['lw'] = 1
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = len(fields)
    plot['timebar'] = plt.subplot(gs[0,0])
    plot['tlimit'] = tlimit
    plot['altitude'] = np.sqrt(np.sum(coords**2,axis=0))
    return plot

def finalize_plot(plot, xlim=None, fname=None, show=False, zeroline=False):
    for f, ax in plot['axes'].items():
        ax.set_ylabel(label_lookup[f])
        ax.set_xlim(plot['time'][0], plot['time'][-1])
        if zeroline:
            ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyle=':', alpha=0.4)
        if 'number_density' in f:
            ax.set_yscale('symlog', linthreshy=10)
    for i in range(plot['N_axes']):
        ax = plot['ax_arr'][i]
        if i == 0:
            twinax = ax.twiny()
            alt = plot['altitude']
            #twinax.plot(alt, np.zeros_like(alt), alpha=0)

            tick_locs = np.array([0,0.25,0.5,0.75,0.99])
            if plot['tlimit'] is not None: 
                lim = plot['tlimit']
                tick_locs = tick_locs*(lim[1]-lim[0])+lim[0]
            twinax.set_xticks(tick_locs)
            twinax.set_xticklabels(['$'+str(int(alt[int(tl*alt.shape[0])]))+'$' for tl in tick_locs])
            twinax.set_xlabel('$\textrm{Altitude}$')

        if i == plot['N_axes']-1:
            ax.set_xlabel('$\texrm{Time}$')
        else:
            ax.set_xticks([])
        #ax.set_ylabel(ax.get_ylabel(), labelpad=30*(i%2))    
        if plot['tlimit'] is not None:
            lim = plot['tlimit']
            t = plot['time']
            tlim = (t[int(lim[0]*t.shape[0])], t[int(lim[1]*t.shape[0])])
            ax.set_xlim(tlim)
    
    tb = plot['timebar']
    if plot['tlimit'] is None:
        xv, yv = np.meshgrid(plot['time'], [0,1])
        m = np.array([plot['time'], plot['time']]).reshape(-1,2, order='F')
        tb.set_xlim(plot['time'][0], plot['time'][-1])
        tb.pcolormesh(xv, yv, m.T, cmap='inferno',rasterized=True)
    else:
        xv, yv = np.meshgrid(np.linspace(tlim[0],tlim[1],1e3), lim)
        m = np.array([np.linspace(tlim[0],tlim[1],1e3),
                      np.linspace(tlim[0],tlim[1],1e3)]).reshape(-1,2, order='F')
        tb.set_xlim(tlim[0], tlim[1])
    
    tb.axis('off')
    
    
    
    plot['ax_arr'][0].legend()#(bbox_to_anchor=(1.4, 1))
    plot['ax_arr'][0].set_zorder(1)
    plot['figure'].set_size_inches(10,10)
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
        if len(keys) == 0: continue
        print 'getting indxs: '+ds_type
        indxs[ds_type] = bin_coords(coords, ds_names[keys[0]])
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


def make_plot(times, fields, orbits, title, indxs, coords, ds_names, ds_types, skip=1, subtitle=None):
    plot = setup_plot(fields, ds_names.keys(), coords, tlimit=(0.4,0.7))

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
    plot['time'] = t
    if subtitle is None: subtitle= orbits[0]
    print 'Saving: ', 'Output/{0}_{1}.pdf'.format(title, subtitle)
    finalize_plot(plot, zeroline=True, fname='Output/{0}_{1}.pdf'.format(title, subtitle))
    


def flythrough_orbit(orbits, trange, ds_names, ds_types, **kwargs):

    coords, times = get_path_pts(trange, Npts=150)
    indxs = get_path_idxs(coords, ds_names, ds_types)

    ion_fields =['H_p1_number_density',
          'O2_p1_number_density',
          'O_p1_number_density',
          'CO2_p1_number_density'] 
    mag_fields = ['magnetic_field_x', 'magnetic_field_y',
          'magnetic_field_z']
    make_plot(times, ion_fields, orbits, 'ion_flythrough', indxs, coords, ds_names, ds_types, **kwargs)
    #make_plot(times, mag_fields, orbits, 'mag_flythrough', indxs, coords, ds_names, ds_types, skip=20, **kwargs)


def main():

    orbit_groups =[np.array([353, 360, 363, 364, 364, 365, 366, 367, 367, 368, 369, 370, 371,
            375, 376, 376, 380, 381, 381, 382, 386, 386, 387, 390, 391])] 
    
    # Get Datasets setup
    ds_names, ds_types = get_datasets(new_models=False)

    for gi, orbits in enumerate(orbit_groups):
        tranges = get_orbit_times(orbits)
        mid_tr = tranges[:, orbits.shape[0]/2]
        flythrough_orbit(orbits, mid_tr, ds_names, ds_types, subtitle='G{0}'.format(gi+1))

        break



if __name__ == "__main__":
    main()
