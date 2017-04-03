"""
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib import ticker
from general_functions import *
from matplotlib.collections import LineCollection
import getopt
import sys
import cmocean
import glob
plt.style.use('seaborn-poster')

def setup_sliceplot():
    plot = {}
    
    f, axes = plt.subplots(3,1)
    
    plot['figure'] = f
    plot['axes'] = axes[::-1]
    
    return plot

def orbit_intersect_plane(coords, center, ax_i):
    if center is None: center = [0,0,0]
    # test to see if there is any intersection
    
    ax_c = center[ax_i]
    pt_argmin  = np.argsort(np.abs(coords[ax_i, :-1]-ax_c))
    
    p1_idx = pt_argmin[0]
    if p1_idx == coords.shape[1]: p1_idx = p1_idx
    p1 = coords[:,p1_idx]
    
    p1_into_plane = ((coords[ax_i, p1_idx] - coords[ax_i, p1_idx+1])>0)
    
    i = 1
    while True:
        p2_idx = pt_argmin[i]
        p2_into_plane = ((coords[ax_i, p2_idx] - coords[ax_i, p2_idx+1])>0)
        if p2_into_plane == p1_into_plane: 
            i+=1
        else: break
            
    return (p1, coords[:,p2_idx])

def add_orbit(ax, ax_i, orbit, center=None, show_intersect=False,
              show_center=False, lw=2.8):
    off_ax = [[1,2],[0,2],[0,1]]
    coords = get_orbit_coords(orbit, Npts=250)
    ltimes = np.linspace(0,1,coords.shape[1])
    #ltimes[np.logical_and(coords[ax_i]<0, np.sqrt(np.sum(coords[off_ax[ax_i]],axis=1))<1)] = 0
    
    x,y = coords[off_ax[ax_i][0]], coords[off_ax[ax_i][1]]
    
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('inferno'),
                        norm=plt.Normalize(ltimes.min(), ltimes.max()))
    lc.set_array(ltimes)
    lc.set_linewidth(lw)
    
    ax.add_collection(lc)
    
    if show_intersect:
        p1, p2 = orbit_intersect_plane(coords, center, ax_i)
        ax.scatter([p1[off_ax[ax_i][0]], p2[off_ax[ax_i][0]]],
                   [p1[off_ax[ax_i][1]], p2[off_ax[ax_i][1]]],
                   marker='x', color='grey', zorder=20)

def finalize_sliceplot(plot, orbit=None, center=None, show_center=False,
                       show_intersect=False, fname='Output/test.pdf'):
    plot['figure'].set_size_inches(5,10)
    ax_labels = [['Y','Z'],['X','Z'],['X','Y']]
    ax_title_lab = ["X", "Y", "Z"]
    
    for ax_i, ax in enumerate(plot['axes']):
        ax.set_aspect('equal')
        
        if center is None: 
            mars_frac = 1
            ax.set_title('$\mathrm{'+ax_title_lab[ax_i]+'= 0}\;(R_M)$')
        else: 
            mars_frac = np.real(np.sqrt(1-center[ax_i]**2))
            ax.set_title('$\mathrm{'+ax_title_lab[ax_i]+'= '+"{0:0.02}".format(center[ax_i])+'}\;(R_M)$')
        
        circle = plt.Circle((0, 0), mars_frac, color='k')
        ax.add_artist(circle)
        circle = plt.Circle((0, 0), 1, color='k', alpha=0.1, zorder=0)
        ax.add_artist(circle)
        
        if orbit is not None: add_orbit(ax,ax_i, orbit, center, show_center=show_center, show_intersect=True)
            
        ax.set_xlabel('$\mathrm{'+ax_labels[ax_i][0]+'} \;(R_M)$')
        ax.set_ylabel('$\mathrm{'+ax_labels[ax_i][1]+'} \;(R_M)$')

        off_ax = [[1,2],[0,2],[0,1]]
        if show_center:
            ax.scatter([center[off_ax[ax_i][0]]],
                       [center[off_ax[ax_i][1]]],
                       marker='.', color='grey', zorder=20, s=3)
        
    plt.tight_layout()
    print 'Saving: {0}'.format(fname)
    plt.savefig(fname)

def get_offgrid_slice(ds, ax_i, field, vec_field, center, extra_fields=None):
    if center is None: ax_c = 0
    else: ax_c = center[ax_i]
        
    
    all_fields = ['x','y','z']
    ax_lab = all_fields[ax_i]
    c_fields = [['y','z'],['x','z'],['x','y']][ax_i]

    # Need to divide this into two sections, 
    # inside and outside 2 R_M

    r = np.sqrt(ds['x']**2+ds['y']**2+ds['z']**2)
    outer_pts = r>=1.5
        

    tol = 0.01
    idx0 = np.abs(ds[ax_lab]-ax_c) < np.min(np.abs(ds[ax_lab]-ax_c))+tol
    while sum(idx0) < 10000:
        tol *=5
        idx0 = np.abs(ds[ax_lab]-ax_c) < np.min(np.abs(ds[ax_lab]-ax_c))+tol

    idx1 = np.logical_and(outer_pts, idx0)
    while sum(idx1) < 10000:
        tol *=5
        idx1 = np.abs(ds[ax_lab][outer_pts]-ax_c) <\
                np.min(np.abs(ds[ax_lab][outer_pts]-ax_c))+tol

    idx1 = np.abs(ds[ax_lab]-ax_c) <\
            np.min(np.abs(ds[ax_lab]-ax_c))+tol
    idx = np.logical_or(idx0, np.logical_and(idx1, outer_pts))
    
    if vec_field: 
        for ax in c_fields: all_fields.append(field+'_'+ax)
    else: all_fields.append(field)
   
    ds_slice = {}
    for f in all_fields:
        ds_slice[f] = ds[f][idx]
    if extra_fields is not None:
        for f in extra_fields:
            ds_slice[f] = ds[f][idx]
  
    return (ds_slice, c_fields)
 

def slice_regrid(ds, ax_i, field, vec_field=False, test=False, center=None, extra_fields=None):
    if test: lin = np.linspace(-4, 4, 50)
    else: lin = np.linspace(-4, 4, 250)
    grid_0, grid_1 = np.meshgrid(lin, lin)
    g0_flat, g1_flat = grid_0.flatten(), grid_1.flatten()
    
    ds_slice, c_fields = get_offgrid_slice(ds, ax_i, field, vec_field, center, extra_fields=extra_fields) 
    coords_slice = np.array([ds_slice[c_fields[0]],ds_slice[c_fields[1]],ds_slice[['x','y','z'][ax_i]]])
    idxs = np.zeros_like(g0_flat,dtype=int)
    
    if center is None: ax_c = 0
    else: ax_c = center[ax_i]
    
    for i, c in enumerate(zip(g0_flat, g1_flat)):
        c0, c1 = c
        
        dr = np.sqrt((coords_slice[0]-c0)**2+(coords_slice[1]-c1)**2+(coords_slice[2]-ax_c)**2)
        idxs[i] = np.argmin(dr)
        
    if vec_field:
        field_dat = [ds_slice[field+'_'+c_fields[0]][idxs], ds_slice[field+'_'+c_fields[1]][idxs]]
        
        field_dat[0][idxs==-1] = 0
        field_dat[1][idxs==-1] = 0
        field_dat[0] = field_dat[0].reshape(grid_0.shape)
        field_dat[1] = field_dat[1].reshape(grid_0.shape)
        
        in_mars = np.sqrt(grid_0**2+grid_1**2+ax_c**2)<1.1
        field_dat[0][in_mars] = 0
        field_dat[1][in_mars] = 0

    else:
        if extra_fields is None:
            field_dat = ds_slice[field][idxs]
            field_dat[idxs==-1] = 0
            field_dat = field_dat.reshape(grid_0.shape)
        
        else:
            field_dat = {}
            for f in extra_fields+[field]:
                field_dat[f] = ds_slice[f][idxs].reshape(grid_0.shape)
        
    return (grid_0, grid_1, field_dat)

def slice_onax(ds, ax_i, field, vec_field=False, idx=None, center=None, test=False):
    if idx is not None and center is not None:
        print 'Cannot supply both idx and center to slice_onax'
        raise(RuntimeError)
    
    if center is not None:
        ax_center = center[ax_i]
        slc_v = ['x', 'y', 'z'][ax_i]
        #print np.argmin(np.abs(ds[slc_v]-ax_center), axis=ax_i)
        idx = np.argmin(np.abs(ds[slc_v]-ax_center), axis=ax_i)[0,0]
                        
    if idx is None:
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
    print field

    return (slc_0, slc_1, field)
    
def slice_data(ds, ax_i, field, regrid_data, **kwargs):
    if regrid_data: return slice_regrid(ds, ax_i, field, **kwargs)
    else: return slice_onax(ds, ax_i, field, **kwargs)

def plot_data_vec(plot, slc, ax_i):
    slc_0, slc_1, field_dat = slc
    Ns = np.array(slc_0.shape, dtype=int)/15
    lws = np.sqrt(np.sum(np.array(field_dat)**2,axis=0))
    lws = 1.2*lws[::Ns[0], ::Ns[1]].flatten()/lws.max()
    #lws[lws==0] = 0.001

    plot['axes'][ax_i].quiver(slc_0.T[::Ns[0], ::Ns[1]], slc_1.T[::Ns[0], ::Ns[1]],
                              field_dat[0].T[::Ns[0], ::Ns[1]], 
                              field_dat[1].T[::Ns[0], ::Ns[1]],
                              linewidth=0.5)

def plot_data_scalar(plot, slc, ax_i, field, logscale=True, zlim=None, cbar=True, diverge_cmap=False):
    slc_0, slc_1, field_dat = slc
    #diverge_cmap, logscale, zlim = True, False, (-30,30)
    #zlim = (-180,180)
    #if zlim is None: vmin, vmax = 1e-3, field_lims[field][1] #np.nanmax(field_dat), 1e-3
    #else: vmin, vmax = zlim
    field_lims = field_lims_slices

    if field in field_lims.keys(): vmin, vmax = field_lims_slices[field]
    else: vmin, vmax = np.nanmin(field_dat), np.nanmax(field_dat)

    diverging, logscale, symlogscale=False, False, False
    if sum([1 for dfk in diverging_field_keys if dfk in field])>0: diverging=True
    if sum([1 for lfk in log_field_keys if lfk in field])>0: logscale=True
    if sum([1 for sfk in symlog_field_keys if sfk in field])>0: symlogscale=True

    
    if logscale: norm = LogNorm(vmax=vmax, vmin=vmin)
    elif symlogscale: 
        linthresh=1e5
        norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)
	maxlog=int(np.ceil( np.log10(vmax) ))
	minlog=int(np.ceil( np.log10(-vmin) ))
	linlog=int(np.ceil(np.log10(linthresh)))

	#generate logarithmic ticks 
	tick_locations=([-(10**x) for x in xrange(minlog,linlog-1,-1)]
			+[0.0]
			+[(10**x) for x in xrange(linlog,maxlog+1)] )
    elif diverging: 
        vm = np.max(np.abs([vmin, vmax]))
        norm = Normalize(vmax=vm, vmin=-1*vm)
    else: norm = Normalize(vmax=vmax, vmin=vmin)

    if diverging: cmap='RdBu' # cmap=cmocean.cm.delta
    else: cmap='viridis'

    if field in label_lookup: label=label_lookup[field]
    else: label = field

    if field_dat.max() != field_dat.min(): 
    
        im = plot['axes'][ax_i].pcolormesh(slc_0.T, slc_1.T, field_dat.T,
                                           norm=norm, cmap=cmap, rasterized=True)
        if cbar:
            if symlogscale:
                plt.colorbar(im, ax=plot['axes'][ax_i], label=label,
                             ticks=tick_locations,
                             format=ticker.LogFormatterMathtext())
            else:
                plt.colorbar(im, ax=plot['axes'][ax_i],label=label)

    
    plot['axes'][ax_i].set_xlim(slc_0.min(), slc_0.max())
    plot['axes'][ax_i].set_ylim(slc_1.min(), slc_1.max())
    
def plot_data(plot, slc, ax_i, vec_field, field,**kwargs):
    if vec_field: plot_data_vec(plot, slc, ax_i, **kwargs)
    else: plot_data_scalar(plot, slc, ax_i, field, **kwargs)
    

def make_plot(ds_name, field, center=None, orbit=None, regrid_data=False,
              vec_field=False, fname=None, test=False, mark=False):
    ds = load_data(ds_name,field=field, vec_field=vec_field)
    plot = setup_sliceplot()
    
    for ax in [0,1,2]:
        slc = slice_data(ds, ax, field, regrid_data=regrid_data, vec_field=vec_field, test=test, center=center)
        plot_data(plot, slc, ax, vec_field, field)
    finalize_sliceplot(plot, orbit=orbit, center=center, fname=fname,show_center=mark)

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:i:o:t:c:d:m",["field=","infile=", "orbit=", "center=", "test","dir=", "mark"])
    except getopt.GetoptError:
        print getopt.GetoptError()
        print 'error'
        return
    
    infile, field, orbit, center, test, fdir, mark = None, None, None, None, False, None, False
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--orbit"):
            orbit = int(arg)
        elif opt in ("-t", "--test"):
            test = True
        elif opt in ("-d", "--dir"):
            fdir = arg
        elif opt in ("-c", "--center"):
            import ast
            center = np.array(ast.literal_eval(arg))
        elif opt in ('-m', '--mark'):
            mark=True
    
    if infile is None and fdir is None: 
        print 'must supply file'
        raise(RuntimeError)
    
    if 'velocity' == field[-8:] or 'magnetic_field' == field:
        vec_field = True
    else: vec_field = False

    if fdir is not None: infiles = glob.glob(fdir+"*.h5")
    else: infiles = [infile]

    if 'Heliosares' in infiles[0] or 'helio' in infiles[0]: regrid_data = False
    else: regrid_data = True

    
    if field == 'all_ion':
        with h5py.File(infile, 'r') as f: fields = f.keys()
        fields = [f for f in fields if 'p1_number_density' in f]
    else: fields = [field]


    
    for infile in infiles:
        for field in fields:
            print infile, field
            make_plot(infile, field, orbit=orbit, test=test,
                      regrid_data=regrid_data, vec_field=vec_field, center=center, mark=mark,
                      fname='Output/slice_{0}_{1}_{2}.pdf'.format(field, infile.split('/')[-1][:-3], orbit))
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
