import yt
import spiceypy as sp
import matplotlib.pyplot as plt
import numpy as np
from yt.frontends.netcdf.api import NetCDFDataset
from yt.units.yt_array import YTArray


def plot_slicepath(ds, coords, times, coord_limit=False):

    from test_geocart import slice_lon, slice_lat0

    slc_0 = slice_lon(ds, 'oplus', lon=0, return_slc=True)
    slc_1 = slice_lon(ds, 'oplus', lon=90, return_slc=True)
    slc_2 = slice_lat0(ds, 'oplus', return_slc=True)
    slcs = [slc_0, slc_1, slc_2]

    from matplotlib.collections import LineCollection
    t = np.cumsum(times)
    c = (t-t[0])/(t[-1]-t[0])
    f, axes = plt.subplots(3,1)

    for x,y in ([0,1], [1,2], [2,0]):

        points = np.array([coords[:,x], coords[:,y]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('viridis'),
                                    norm=plt.Normalize(0, 1))
        lc.set_array(c)
        lc.set_linewidth(3)

        axes[x].imshow(slcs[x], cmap='viridis', extent=[-500,500,-500,500])
        axes[x].add_collection(lc)
        if coord_limit:
            axes[x].set_xlim(coords[:,x].min(), coords[:,x].max())
            axes[x].set_ylim(coords[:,y].min(), coords[:,y].max())

    plt.savefig('Output/slicepath.pdf')



def get_maven_path(geo=False, step=100, trange=None):
    sp.furnsh("maven_spice.txt")

    if trange is None:
        utc = ['2015-12-14/16:30:00', '2015-12-14/21:00:00']
    else:
        utc = trange
    et1, et2 = sp.str2et(utc[0]), sp.str2et(utc[1])

    times = np.linspace(et1, et2, step)

    positions, lightTimes = sp.spkpos('Maven', times, 'J2000', 'NONE', 'MARS BARYCENTER')

    if not geo:
        return positions, times+946080000 +647812 
 #lightTimes

    geo_coords = np.zeros_like(positions)

    for i,p in enumerate(positions):
        geo_coords[i,:] = sp.spiceypy.reclat(p)

    geo_coords[:,0] = (geo_coords[:,0]-3388.25)
    geo_coords[:,1] = (geo_coords[:,1]+np.pi)*180/np.pi
    geo_coords[:,2] = (geo_coords[:,2])*180/np.pi
    geo_coords = geo_coords[:,[0,2,1]]

    return geo_coords, times+946080000+647812 




def plot_geo(coords, times):
    from matplotlib.collections import LineCollection
    t = np.cumsum(times)
    c = (t-t[0])/(t[-1]-t[0])
    f, axes = plt.subplots(3,1)

    for x,y in ([0,1], [1,2], [2,0]):


        points = np.array([coords[:,x], coords[:,y]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('viridis'),
                                    norm=plt.Normalize(0, 1))
        lc.set_array(c)
        lc.set_linewidth(3)

        axes[x].add_collection(lc)
        axes[x].set_xlim(coords[:,x].min(), coords[:,x].max())
        axes[x].set_ylim(coords[:,y].min(), coords[:,y].max())

    plt.savefig('Output/geopath.pdf')


def plot_field(t,dat, add_coords=None):
    from matplotlib.collections import LineCollection
    c = (t-t[0])/(t[-1]-t[0])
    


    points = np.array([t, dat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('viridis'),
                                norm=plt.Normalize(0, 1))
    lc.set_array(c)
    #lc.set_linewidth(3)

    if add_coords is not None:
        f, axes = plt.subplots(4,1)
        axes[0].add_collection(lc)
        axes[0].set_yscale('log')

        for coord_set in add_coords:

            axes[1].plot(t, coord_set[:,0])
            axes[2].plot(t, coord_set[:,1])
            axes[3].plot(t, coord_set[:,2])
            
        for ax in axes:
            ax.set_xlim(t.min(), t.max()) 
            ax.set_xlim(t[dat==dat].min(), t[dat==dat].max())

    else:
        ax = plt.gca()
        ax.add_collection(lc)
        ax.set_yscale('log')


    plt.savefig('Output/pathdat.pdf')
    #plt.show()


def get_path_arrays(ds, coords, times, fields, use_pts=False):

    if use_pts:
        return get_path_arrays_points(ds, coords, times, fields)
    else:
        return get_path_arrays_rays(ds, coords, times, fields)


def get_path_arrays_points(ds, coords, times, fields):

    field_dat = {field:YTArray(np.ones(len(times))*np.nan, 'cm**-3') for field in fields}

    for i, coord in enumerate(coords):

        vals = ds.find_field_values_at_point(fields, coord) 
        
        if len(vals) == 0: continue
        for f_i, f  in enumerate(fields):
            if len(vals[f_i]) > 0:
                field_dat[f][i] = vals[f_i].in_units('cm**-3')

    return (times, field_dat, coords, range(len(times))) 


def get_path_arrays_rays(ds, coords, times, fields):
    paths = [ds.ray(coords[i], coords[i+1]) for i in range(coords.shape[0]-2)]

    N_empty = np.sum([len(path['t']) == 0 for path in paths]) 
    length = np.sum([path['t'].shape for path in paths])
    N_pts = length+N_empty


    arr_times = np.zeros(N_pts, dtype=times.dtype)
    arr_idx = np.empty(N_pts, dtype=int)
    arr_coords = np.empty((N_pts,3), dtype=int)
    field_dat = {field:np.ones(N_pts)*np.nan for field in fields}


    arr_i = 0
    for path_i, path in enumerate(paths):
        l = path[fields[0]].shape[0]

        sort = np.argsort(path['t'])

        arr_times[arr_i:arr_i+l] = path['t'][sort]*(times[path_i+1] -times[path_i])+times[path_i] 

        if l > 0:
            for field in fields:
                field_dat[field][arr_i:arr_i+l] = path[field][sort].in_units('cm**-3')
            arr_i = arr_i + l 
        else:
            arr_i = arr_i + 1

    plt.plot(np.linspace(0,1,len(times)), times)
    plt.plot(np.linspace(0,1,len(arr_times)), arr_times)
    #plt.semilogy()
    plt.show()

    return (arr_times, field_dat, arr_coords, arr_idx)


def test_geo():

    fdir = '/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/Run1/'
    fname = 'Hsw_18_06_14_t00600.nc'
    ds = NetCDFDataset(fdir+fname, model='heliosares')


    #coords, times = get_maven_path(geo=True, step=2000, trange =['2015-02-28/22:45:47', '2015-03-01/03:16:07'])
    positions, times = get_maven_path(geo=False, trange = ['2015-02-28/22:45:47', '2015-03-01/03:16:07'])

    plot_slicepath(ds, positions, times)


    #t, dat, arr_coords, idx = get_path_arrays(ds, coords, times, ['oplus'], use_pts=True)
    #plot_field(t, dat['oplus'], add_coords=[arr_coords, coords[idx]])





    #plt.plot(density)
    #plt.show()

test_geo()
