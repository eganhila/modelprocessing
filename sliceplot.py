import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
from general_functions import *
from matplotlib.collections import LineCollection
import getopt
import sys

def setup_plot():
    plot = {}
    
    f, axes = plt.subplots(3,1)
    
    plot['figure'] = f
    plot['axes'] = axes
    
    return plot

def add_orbit(ax, ax_i, orbit):
    off_ax = [[1,2],[0,2],[0,1]]
    trange = get_orbit_times(orbit)
    coords, ltimes = get_path_pts(trange,Npts=250)
    
    x,y = coords[off_ax[ax_i][0]]/3390, coords[off_ax[ax_i][1]]/3390
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('inferno'),
                        norm=plt.Normalize(ltimes.min(), ltimes.max()))
    lc.set_array(ltimes)
    lc.set_linewidth(1.8)
    
    ax.add_collection(lc)

def finalize_plot(plot, orbit=None, fname='Output/test.pdf'):
    plot['figure'].set_size_inches(5,10)
    ax_labels = [['Y','Z'],['X','Z'],['X','Y']]
    for ax_i, ax in enumerate(plot['axes']):
        ax.set_aspect('equal')
        circle = plt.Circle((0, 0), 1, color='k')
        ax.add_artist(circle)
        if orbit is not None: add_orbit(ax,ax_i, orbit)
        ax.set_xlabel('$\mathrm{'+ax_labels[ax_i][0]+'} \;(R_M)$')
        ax.set_ylabel('$\mathrm{'+ax_labels[ax_i][1]+'} \;(R_M)$')
        
    plt.tight_layout()
    print 'Saving: {0}'.format(fname)
    plt.savefig(fname)

def load_data(ds_name, field, vec_field):
    ds = {}
    with h5py.File(ds_name, 'r') as f:
        if 'xmesh' in f.keys():
            ds['x'] = f['xmesh'][:]/3390
            ds['y'] = f['ymesh'][:]/3390
            ds['z'] = f['zmesh'][:]/3390
        else:
            ds['x'] = f['x'][:]/3390
            ds['y'] = f['y'][:]/3390
            ds['z'] = f['z'][:]/3390            
        
        if vec_field:
            ds[field+'_x'] = f[field+'_x'][:]
            ds[field+'_y'] = f[field+'_y'][:]
            ds[field+'_z'] = f[field+'_z'][:]
        else:
            ds[field] = f[field][:]
            
    return ds

def get_offgrid_slice(ds, ax_i, field, vec_field):
    if ax_i == 0:
        idx = ds['x'] == 0
        c_fields = ['y', 'z']
    elif ax_i == 1:
        idx = ds['y'] == 0
        c_fields = ['x', 'z']
    elif ax_i == 2:
        idx = ds['z'] == 0
        c_fields = ['x', 'y']

    all_fields = [f for f in c_fields]
    if vec_field: 
        for ax in c_fields:
            all_fields.append(field+'_'+ax)
    else: all_fields.append(field)
   
    ds_slice = {}
    for field in all_fields:
        ds_slice[field] = ds[field][idx]
  
    return (ds_slice, c_fields)
 

def slice_regrid(ds, ax_i, field, vec_field, test):
    if test: lin = np.linspace(-4, 4, 50)
    else: lin = np.linspace(-4, 4, 250)
    grid_0, grid_1 = np.meshgrid(lin, lin)
    g0_flat, g1_flat = grid_0.flatten(), grid_1.flatten()
    
    ds_slice, c_fields = get_offgrid_slice(ds, ax_i, field, vec_field) 
    c0_slice = ds_slice[c_fields[0]]
    c1_slice = ds_slice[c_fields[1]]
    
    idxs = np.zeros_like(g0_flat,dtype=int)
    
    for i, c in enumerate(zip(g0_flat, g1_flat)):
        c0, c1 = c
        dr = np.sqrt((c0_slice-c0)**2+(c1_slice-c1)**2)
        idxs[i] = np.argmin(dr)
        
    if vec_field:
        field = [ds_slice[field+'_'+c_fields[0]][idxs].reshape(grid_0.shape),\
                 ds_slice[field+'_'+c_fields[1]][idxs].reshape(grid_0.shape)]
        in_mars = np.sqrt(grid_0**2+grid_1**2)<1.2
        field[0][in_mars] = 0
        field[1][in_mars] = 0
    else:
        field = ds_slice[field][idxs].reshape(grid_0.shape)
        
    return (grid_0, grid_1, field)

def slice_onax(ds, ax_i, field, vec_field):    
    idx = ds['x'].shape[ax_i]/2
    if ax_i == 0: 
        slc_0 = ds['y'][idx, :,:]
        slc_1 = ds['z'][idx, :,:]
        if vec_field: field = [ds[field+'_y'][idx, :,:], ds[field+'_z'][idx, :,:]]
        else: field = ds[field][idx, :,:]
    if ax_i == 1: 
        slc_0 = ds['x'][:, idx, :]
        slc_1 = ds['z'][:, idx, :]
        if vec_field: field = [ds[field+'_x'][:,idx,:], ds[field+'_z'][:, idx, :]]
        else: field = ds[field][:,idx,:]
    if ax_i == 2: 
        slc_0 = ds['x'][:,:,idx]
        slc_1 = ds['y'][:,:,idx]
        if vec_field: field = [ds[field+'_x'][:,:,idx], ds[field+'_y'][:,:,idx]]
        else: field = ds[field][:,:,idx]
    
    return (slc_0, slc_1, field)
    
def slice_data(ds, ax_i, field, regrid_data, vec_field, test):
    if regrid_data: return slice_regrid(ds, ax_i, field, vec_field, test)
    else: return slice_onax(ds, ax_i, field, vec_field)

def plot_data_vec(plot, slc, ax_i):
    slc_0, slc_1, field_dat = slc
    Ns = slc_0.shape[0]/25
    lws = np.sqrt(np.sum(np.array(field_dat)**2,axis=0))
    lws = 1.2*lws[::Ns, ::Ns].flatten()/lws.max()
    #lws[lws==0] = 0.001

    plot['axes'][ax_i].quiver(slc_0.T[::Ns, ::Ns], slc_1.T[::Ns, ::Ns],
                              field_dat[0].T[::Ns, ::Ns], 
                              field_dat[1].T[::Ns, ::Ns],
                              linewidth=0.5)

def plot_data_scalar(plot, slc, ax_i, field):
    slc_0, slc_1, field_dat = slc
    im = plot['axes'][ax_i].pcolormesh(slc_0.T, slc_1.T, field_dat.T,
                norm=LogNorm(vmax=field_dat.max(),
                             vmin=1e-3),
                cmap='viridis', rasterized=True)
    plot['axes'][ax_i].set_xlim(slc_0.min(), slc_0.max())
    plot['axes'][ax_i].set_ylim(slc_1.min(), slc_1.max())
    plt.colorbar(im, ax=plot['axes'][ax_i],label=label_lookup[field])
    
def plot_data(plot, slc, ax_i, vec_field, field):
    if vec_field: plot_data_vec(plot, slc, ax_i)
    else: plot_data_scalar(plot, slc, ax_i, field)
    

def make_plot(ds_name, field, orbit=None, regrid_data=False,
              vec_field=False, fname=None, test=False):
    ds = load_data(ds_name, field, vec_field)
    plot = setup_plot()
    
    for ax in [0,1,2]:
        slc = slice_data(ds, ax, field, regrid_data, vec_field, test=test)
        plot_data(plot, slc, ax, vec_field, field)
    finalize_plot(plot, orbit=orbit, fname=fname)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:i:o:t:",["field=","infile=", "orbit=", "test"])
    except getopt.GetoptError:
        return
    
    infile, field, orbit, test = None, None, None, False
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--orbit"):
            orbit = int(arg)
        elif opt in ("-t", "--test"):
            test = True
    if infile is None: infile = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/run1.h5'
    if field is None: field = "electron_number_density"
    
    
    if 'velocity' in field or 'magnetic_field' in field:
        vec_field = True
    else: vec_field = False
    if 'Heliosares' in infile: regrid_data = False
    else: regrid_data = True
    
    if field == 'all_ion':
        with h5py.File(infile, 'r') as f: fields = f.keys()
        fields = [f for f in fields if 'number_density' in f]
    else: fields = [field]
    
    for field in fields:
        make_plot(infile, field, orbit=orbit, test=test,
                  regrid_data=regrid_data, vec_field=vec_field,
                  fname='Output/slice_{0}_{1}_{2}.pdf'.format(field, infile.split('/')[-1][:-3], orbit))
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    