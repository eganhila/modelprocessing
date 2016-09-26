import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp
import h5py

colors ={'AEQNmin-SSLONG0':'DodgerBlue', 'AEQNmax-SSLONG0':'MidnightBlue',
                  'run1':'Maroon', 'run2':'Crimson'}
log_fields = []

def get_path_pts(trange, geo=False, Npts=50):
    sp.furnsh("maven_spice.txt")
    et1, et2 = sp.str2et(trange[0]), sp.str2et(trange[1])
    times = np.linspace(et1, et2, Npts)
    
    positions, lightTimes = sp.spkpos('Maven', times, 'J2000', 
                                'NONE', 'MARS BARYCENTER')

    if not geo:
        return positions, times+946080000 +647812 

    geo_coords = np.zeros_like(positions)

    for i,p in enumerate(positions):
        geo_coords[i,:] = sp.spiceypy.reclat(p)

    geo_coords[:,0] = (geo_coords[:,0]-3388.25)
    geo_coords[:,1] = (geo_coords[:,1]+np.pi)*180/np.pi
    geo_coords[:,2] = (geo_coords[:,2])*180/np.pi
    geo_coords = geo_coords[:,[0,2,1]]

    return geo_coords, times+946080000+647812 


def bin_coords(coords, dsf):
    with h5py.File(dsf, 'r') as dataset:

        idx = np.zeros(coords.shape[0])
        x = dataset['x'][:].flatten()
        y = dataset['y'][:].flatten()
        z = dataset['z'][:].flatten()

    for i in range(coords.shape[0]): 
        dx2  = (coords[i, 0] - x)**2
        dy2  = (coords[i, 1] - y)**2
        dz2  = (coords[i, 2] - z)**2

        dr = np.sqrt(dx2+dy2+dz2)
        idx[i] = np.argmin(dr)

    return idx.astype(int)


def get_datasets(fdir='/Volumes/triton/Data/ModelChallenge/SDC_Archive/'):
    ds_names = {}
    ds_names['AEQNmax-SSLONG0'] = \
            fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmax-SSLONG0.h5'
    ds_names['AEQNmin-SSLONG0'] = \
            fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmin-SSLONG0.h5'
    ds_names['run1'] = \
            fdir+'HELIOSARES/Hybrid/Run1/'+'run1.h5'
    ds_names['run2'] = \
            fdir+'HELIOSARES/Hybrid/Run2/'+'run2.h5'

    ds_types = {'batrus':['AEQNmin-SSLONG0', 'AEQNmax-SSLONG0'],
                'heliosares':['run1', 'run2']}
    ds_classes = {'batrus.euv':['AEQNmin-SSLONG0', 'AEQNmax-SSLONG0'],
                  'heliosares.?':['run1', 'run2']}
    return (ds_names, ds_types, ds_classes)

def plot_field_ds(x, data, ax, kwargs):
    ax.plot(x, data, **kwargs)


def setup_plot(fields, ds_names):
    f, axes = plt.subplots(len(fields), 1)

    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes.flatten())}
    plot['kwargs'] = {ds:{'color':colors[ds], 'label':ds}
                        for ds, c in zip(ds_names, colors)}
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = len(fields)

    return plot


def finalize_plot(plot, fname=None, show=False):
    for f, ax in plot['axes'].items():
        ax.set_ylabel(f.replace('_', ' '))
        if f in log_fields:
            ax.set_yscale('log')
    for i in range(plot['N_axes']):
        if i == plot['N_axes']-1:
            plot['ax_arr'][i].set_xlabel('Time')
        else:
            plot['ax_arr'][i].set_xticks([])
    plt.legend()
    
    if show:
        plt.show()
    if fname is None:
        plt.savefig('Output/test.pdf')
    else:
        plt.savefig(fname)


def main():
    trange = ['2015-12-14/16:30:00', '2015-12-14/21:00:00']
    fields = ['O2_p1_number_density', 'O2_p1_velocity_x',
             'O2_p1_velocity_x',  'O2_p1_velocity_z', 
              'magnetic_field_x', 'magnetic_field_y',
              'magnetic_field_z']

    ds_names, ds_types, ds_classes = get_datasets()
    print 'getting coords'
    coords, times = get_path_pts(trange, Npts=50) 
    print 'setting up plot'
    plot = setup_plot(fields, ds_names.keys())

    for ds_type, keys in ds_types.items():
        print 'getting indxs: '+ds_type
        indxs = bin_coords(coords, ds_names[keys[0]])

        for key in keys:
            print 'doing ds: '+key
            dsf = ds_names[key]

            for field in fields:
                print 'getting field: ', field
                with h5py.File(dsf, 'r') as ds:
                    print field, ds[field][:].flatten()[indxs]
                    plot_field_ds(times, ds[field][:].flatten()[indxs],
                                  plot['axes'][field],
                                  plot['kwargs'][key])

    finalize_plot(plot)



if __name__ == '__main__':
    main()






