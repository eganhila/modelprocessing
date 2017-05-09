import numpy as np
import spiceypy as sp
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from misc.labels import *
from misc.field_default_params import *

sp.furnsh("/Users/hilaryegan/Projects/ModelChallenge/ModelProcessing/misc/maven_spice.txt")
mars_r = 3390
orbit_dir = '/Volumes/triton/Data/OrbitDat/Flythroughs/'
model_dir = '/Volumes/triton/Data/ModelChallenge/'

def load_data(ds_name, field=None, fields=None, vec_field=False):
    """
    Load data for a standard hdf5 dataset into a dictionary

    ds_name (string): full file name to be loaded
    field (optional, string): field to load
    fields (optional, list): fields to load
    vec_field (boolean, default False): if true, automatically load 
        field + "_x", "_y", and "_z". Must be used with field, not fields
    """
    ds = {}
    with h5py.File(ds_name, 'r') as f:
        if 'xmesh' in f.keys():
            ds['x'] = f['xmesh'][:]/3390
            ds['y'] = f['ymesh'][:]/3390
            ds['z'] = f['zmesh'][:]/3390
            grid=True
        else:
            ds['x'] = f['x'][:]
            ds['y'] = f['y'][:]
            ds['z'] = f['z'][:]
            grid=False

        p = np.array([ds['x'], ds['y'], ds['z']])
        norm = p/np.sqrt(np.sum(p**2, axis=0))

        if 'O2_p1_velocity_x' in f.keys(): ion_v = True
        else: ion_v = False
            
        if vec_field:
            ds[field+'_x'] = f[field+'_x'][:]
            ds[field+'_y'] = f[field+'_y'][:]
            ds[field+'_z'] = f[field+'_z'][:]
        elif fields is not None:
            for field in fields:
                ds[field] = get_ds_data(f, field, None, grid=grid, normal=norm, 
                            ion_velocity=ion_v)

        elif field is not None:
            ds[field] = get_ds_data(f, field, None, grid=grid, normal=norm, 
                        ion_velocity=ion_v)
            
    return ds

def get_datasets( R2349=False, SDC_G1=False, maven=True, helio_multi=False):
    """
    Get datasets and related information (which datasets are the same type, etc)

    Set the flag of the set of datasets you want to load as
    True. Can only load one dataset at once. Includes maven
    data by default but can turn that off separately

    Most important is setting up ds_types for code speed.
    Set a new type for every type of dataset that isn't
    identical grid setup. Indexes of coordinates are only
    found once for each type of dataset. If in doubt, set
    a new type for each dataset.
    """
    ds_names = {}
    if R2349:
        ds_names['batsrus_multi_fluid'] =  model_dir+'R2349/batsrus_3d_multi_fluid.h5'
        ds_names['batsrus_multi_species'] =  model_dir+'R2349/batsrus_3d_multi_species.h5'
        #ds_names['batsrus_electron_pressure'] =  model_dir+'R2349/batsrus_3d_pe.h5'
        ds_names['batsrus_electron_pressure'] =  model_dir+'R2349/batsrus_pe_test.h5'
        ds_names['heliosares'] ='/Volumes/triton/Data/ModelChallenge/R2349/heliosares_multi.h5'#  model_dir+'R2349/heliosares.h5'
        ds_names['rhybrid'] ='/Volumes/triton/Data/ModelChallenge/R2349/rhybrid.h5'
        
        ds_types = {'batsrus1':[key for key in ds_names.keys() if 'multi_fluid' in key],
                    'batsrus2':[key for key in ds_names.keys() if 'multi_species' in key],
                    'batsrus3':[key for key in ds_names.keys() if 'electron_pressure' in key],
                    'heliosares':[key for key in ds_names.keys() if 'helio' in key],
                    'rhybrid_helio':['rhybrid']}
        if maven:
            ds_names['maven']=orbit_dir+'orbit_2349.csv'
            ds_types['maven']=['maven']

    elif helio_multi:
        ds_names['t00550'] = model_dir+'R2349/Heliosares_Multi/t00550.h5'
        ds_names['t00560'] = model_dir+'R2349/Heliosares_Multi/t00560.h5'
        ds_names['t00570'] = model_dir+'R2349/Heliosares_Multi/t00570.h5'
        ds_names['t00580'] = model_dir+'R2349/Heliosares_Multi/t00580.h5'
        ds_names['t00590'] = model_dir+'R2349/Heliosares_Multi/t00590.h5'
        ds_names['t00600'] = model_dir+'R2349/Heliosares_Multi/t00600.h5'
        ds_names['t00610'] = model_dir+'R2349/Heliosares_Multi/t00610.h5'
        ds_names['t00620'] = model_dir+'R2349/Heliosares_Multi/t00620.h5'
        ds_names['t00630'] = model_dir+'R2349/Heliosares_Multi/t00630.h5'
        ds_names['t00640'] = model_dir+'R2349/Heliosares_Multi/t00640.h5'
        ds_names['t00650'] = model_dir+'R2349/Heliosares_Multi/t00650.h5'

        ds_types = {'heliosares':[key for key in ds_names.keys()]}
        if maven:
            ds_names['maven'] = orbit_dir+'orbit_2349.csv'
            ds_types['maven']=['maven']
    elif SDC_G1:
        #BATSRUS
        ds_names['bats_min_LS270_SSL0'] = \
                model_dir+'SDC_Archive/BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG0.h5'
        ds_names['bats_min_LS270_SSL180'] = \
            model_dir+'SDC_Archive/BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG180.h5'
        ds_names['bats_min_LS270_SSL270'] = \
                model_dir+'SDC_Archive/BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG270.h5'        
        
        #HELIOSARES
        ds_names['helio_1'] = \
                model_dir+'SDC_Archive/HELIOSARES/Hybrid/'+'helio_1.h5'
        
        ds_names['helio_2'] = \
                model_dir+'SDC_Archive/HELIOSARES/Hybrid/'+'helio_2.h5'
            
        
        ds_types = {'batsrus1':[key for key in ds_names.keys() if 'bats' in key],
                    'heliosares':[key for key in ds_names.keys() if 'helio' in key]}
        if maven:
            pass
            #ds_names['maven'] = orbit_dir+'orbit_2349.csv'
            #ds_types['maven']=['maven']

    else:
        print 'No datasets selected'
    

    return (ds_names, ds_types)

def cart_geo_vec_transform(ds, prefix, indx):
    if 'xmesh' in ds.keys():
        x, y, z = ds['xmesh'][:].flatten()[indx], ds['ymesh'][:].flatten()[indx], ds['zmesh'][:].flatten()[indx]
    else:
        x, y, z = ds['x'][:].flatten()[indx], ds['y'][:].flatten()[indx], ds['z'][:].flatten()[indx]
    vx, vy, vz = ds[prefix+'_x'][:].flatten()[indx], ds[prefix+'_y'][:].flatten()[indx], ds[prefix+'_z'][:].flatten()[indx]
    v = np.array([vx,vy,vz])
    
    lat = -1*(np.arctan2(np.sqrt(x**2+y**2), z))+np.pi/2  #theta
    lon = np.arctan2(y, x)   #phi
    #alt = (np.sqrt(x**2+y**2+z**2)-1)*3390

    #lat, lon = ds['latitude'][:].flatten()[indx], ds['longitude'][:].flatten()[indx]
    #lat, lon = lat*np.pi/180.0, lon*np.pi/180.0

    rot_mat = np.zeros((3,3,lat.shape[0]))
    rot_mat[0,:,:] = np.sin(lat)*np.cos(lon), np.sin(lat)*np.sin(lon), np.cos(lat)
    rot_mat[1,:,:] = np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), -1*np.sin(lat)
    rot_mat[2,:,:] = -1*np.sin(lon), np.cos(lon), np.zeros_like(lat)

    vr = np.einsum('lij,kl->il', rot_mat.T, v)

    return vr

def apply_flat_indx(ds, field, indx):
    return ds[field][:].flatten()[indx]

def apply_all_indx(ds, field, indx):
    return ds[field][:]

def apply_grid_indx(ds, field, indx):
    dat = np.zeros(indx.shape[1])
    dat_flat = ds[field][:].flatten()
    dat_shape = ds[field].shape
    indx_flat = indx[0]*dat_shape[1]*dat_shape[2]+indx[1]*dat_shape[2]+indx[2]
    dat = dat_flat[indx_flat] 
    return dat

def apply_maven_indx(ds, field, indx):
#    return ds.loc[indx, field].values
    return ds.loc[:,field].values


def get_ds_data(ds, field, indx, grid=True, normal=None, ion_velocity=False,
                area=None, maven=False):
    """
    Get data from a dataset for a particular field and set of points
    Can interpret suffixes to get total or  normal values for vector
    fields, or flux values for a number density field.

    ds : loaded data in dictionary form
    field : field to get data for
    indx : indexes of the points get the data for
    grid : boolean indicating if the dataset was saved in array or
        list of points forp 
    normal : hacky way to get nhat for surface
    velocity_field : what velocity field to use in calculation of
        flux values
    area : hacky way to get area for surface
    """

    if indx is None: apply_indx = apply_all_indx
    elif grid: apply_indx = apply_grid_indx
    elif maven: apply_indx = apply_maven_indx
    else: apply_indx = apply_flat_indx

    if ion_velocity and '_' in field: 
        ion = (field.split('_')[0])+'_'+(field.split('_')[1])
        velocity_field = '{0}_velocity'.format(ion)
    else:
        velocity_field = 'velocity'

    if field in ds.keys():
        return apply_indx(ds, field, indx)
    elif field == 'electron_pressure':
        return get_ds_data(ds, ' electron_pressure', indx, grid=grid, maven=maven)
    elif '_total' in field and field.replace('_total', '_x') in ds.keys():
        x = apply_indx(ds, field.replace('_total', '_x'), indx)**2
        y = apply_indx(ds, field.replace('_total', '_y'), indx)**2
        z = apply_indx(ds, field.replace('_total', '_z'), indx)**2
        return np.sqrt(x+y+z)
    elif '_normal' in field: # field.replace('_normal', '_x') in ds.keys():
        vx = apply_indx(ds, field.replace('_normal', '_x'), indx)
        vy = apply_indx(ds, field.replace('_normal', '_y'), indx)
        vz = apply_indx(ds, field.replace('_normal', '_z'), indx)
        v = np.array([vx,vy,vz])
        vn = np.sum(normal*v, axis=0)
        return vn
    elif '_flux' in field:
        vn = 1e5*get_ds_data(ds, velocity_field+'_normal', indx, grid, 
                             normal, ion_velocity, maven=maven)
        dens = get_ds_data(ds, field.replace('flux', "number_density"), indx, grid, maven=maven)
        return vn*dens
    elif 'area' == field:
        return area
                                   
#    elif '_radial' in field and field.replace('_radial', '_x') in ds.keys():
#        return cart_geo_vec_transform(ds,field.replace('_radial', ''), indx)[0]
#    elif '_latitudinal'in field and field.replace('_latitudinal', '_x') in ds.keys(:
#        return cart_geo_vec_transform(ds,field.replace('_latitudinal', ''), indx)[1]
#    elif 'longitudinal' in field and field.replace('_longitudinal', '_x') in ds.keys():
#        return cart_geo_vec_transform(ds,field.replace('_longitudinal', ''), indx)[2]
    elif 'xy' in field:
        prefix = '_'.join(field.split('_')[:-1])
        x = get_ds_data(ds, prefix+'_x', indx, grid=grid, maven=maven)
        y = get_ds_data(ds, prefix+'_y', indx, grid=grid, maven=maven)
        return np.sqrt(x**2+y**2)

    elif '_'.join(field.split('_')[2:]) in ds.keys() and '_'.join(field.split('_')[2:]) not in ['x','y','z']:
        return apply_indx(ds, '_'.join(field.split('_')[2:]), indx)
    elif field == 'magnetic_pressure':
        return get_ds_data(ds, 'magnetic_field_total', indx, grid=grid, maven=maven)**2/(2*1.26E-6*1e9)
    elif field == 'total_pressure':
        if maven: return np.array([])
        pe = get_ds_data(ds, 'electron_pressure', indx, grid=grid, maven=maven)
        pt = get_ds_data(ds, 'thermal_pressure', indx, grid=grid, maven=maven)
        pb = get_ds_data(ds, 'magnetic_pressure', indx, grid=grid, maven=maven)
        p = pb
        if pe.shape == p.shape: p += pe
        if pt.shape == p.shape: p += pt

        return p
    elif 'electron_velocity' in field and 'current_x' in ds.keys():
        print 'evel'
        vec = field[-1]
        u = get_ds_data(ds, 'velocity_'+vec, indx, grid=grid, maven=maven)
        J = get_ds_data(ds, 'current_'+vec, indx, grid=grid, maven=maven)
        n = get_ds_data(ds, 'number_density', indx, grid=grid, maven=maven)
        return u-(J/n)/6.24e6

    elif 'hall_velocity' in field and 'current_x' in ds.keys():
        print 'hvel'
        vec = field[-1]
        J = get_ds_data(ds, 'current_'+vec, indx, grid=grid, maven=maven)
        n = get_ds_data(ds, 'number_density', indx, grid=grid, maven=maven)
        return (J/n)/6.24e6

    elif 'density' in field and field != 'density':
        return get_ds_data(ds, 'density', indx, grid=grid, maven=maven)
    elif 'velocity_x' in field and field != 'velocity_x':
        return get_ds_data(ds, 'velocity_x', indx, grid=grid, maven=maven)
    elif 'velocity_y' in field and field != 'velocity_y':
        return get_ds_data(ds, 'velocity_y', indx, grid=grid, maven=maven)
    elif 'velocity_z' in field and field != 'velocity_z':
        return get_ds_data(ds, 'velocity_z', indx, grid=grid, maven=maven)
    else:
        if maven: dstype = 'maven'
        elif grid: dstype = 'heliosares'
        else: dstype= 'batsrus'
        print "Field {0} not found in {1}".format(field, dstype)
        return np.array([])
        #raise(ValueError)

def adjust_spherical_positions(pos, alt, Rm0):
    alt1 = min(alt)
    R = np.sqrt(np.sum(pos**2, axis=0))
    alt2 = min(R)-Rm0
    Rm = Rm0+alt2-alt1 
    
    return pos/Rm*Rm0

        

def get_orbit_coords(orbit, geo=False, Npts=50, units_rm=True, sim_mars_r=3396.0,
                 adjust_spherical=True, return_time=False):
    """
    A function that returns coordinates of the spacecraft for
    a given orbit.

    orbit (int): Orbit #
    geo (bool, default = False): Return coordinates in spherical geographic
        system. Otherwise return in cartesian MSO. !! Doesn't
        currently work
    Npts (int, default = 50): Number of points to sample orbit with. Only
        choose N that 10000  is easily divisible by
    units_rm (bool, default=True): Return coordinates in units of
        mars radius. Otherwise return in km
    sim_mars_r (float, default=3396.0): radius of planet to assume in simulation
    adjust_spherical (bool, default=True): Adjust the coordinates to
        account for a non-spherical mars
    """
    Nskip = 10000/Npts
    data = pd.read_csv(orbit_dir+'orbit_{0:04d}.csv'.format(orbit))#[::Nskip]
    pos = np.array([data['x'], data['y'], data['z']])
    time = data['time'].values
    time = (time-time[0])/(time[-1]-time[0])
    alt = data['altitude']
     
    if adjust_spherical:
        pos = adjust_spherical_positions(pos, alt, sim_mars_r)

    if units_rm:
        pos = pos/sim_mars_r

    if return_time: return (pos,time)
    else: return pos


def bin_coords(coords, dsf, grid=True):
    """
    Get indexes of the dataset points for specified
    set of coordinates.

    coords (3, N): array of points assumed to be in same
        coordinate system as datasets, cartesian MSO
    dsf: filename of ds
    grid (boolean, default True): if the dataset was saved in array or
        list of points form

    """
    if grid: return bin_coords_grid(coords, dsf)
    else: return bin_coords_nogrid(coords, dsf)
    
def bin_coords_grid(coords, dsf):
    with h5py.File(dsf, 'r') as dataset:
        x = dataset['xmesh'][:,0,0]/3390
        y = dataset['ymesh'][0,:,0]/3390
        z = dataset['zmesh'][0,0,:]/3390
        mesh_shape = (dataset['xmesh'].shape)
        
    idx = np.zeros((3, coords.shape[-1]))
    
    for i in range(coords.shape[-1]):
        idx_x = np.argmin((coords[0,i]-x)**2)
        idx_y = np.argmin((coords[1,i]-y)**2)
        idx_z = np.argmin((coords[2,i]-z)**2)
        
        idx[:, i] = [idx_x, idx_y, idx_z]
            
    return idx.astype(int)
        
def bin_coords_nogrid(coords, dsf):
    with h5py.File(dsf, 'r') as dataset:
        x = dataset['x'][:].flatten()
        y = dataset['y'][:].flatten()
        z = dataset['z'][:].flatten()

    idx = np.zeros(coords.shape[-1])

    for i in range(coords.shape[-1]):
        dx2  = (coords[0, i] - x)**2
        dy2  = (coords[1, i] - y)**2
        dz2  = (coords[2, i] - z)**2
        
        dr = np.sqrt(dx2+dy2+dz2)
        idx[i] = np.argmin(dr)

    return idx.astype(int)



def convert_coords_cart_sphere(coords_cart):
    """
    Converts a set of coordinates in a cartesian 
    coordinate system to a spherical one. Returns in
    order (lat, lon, alt). 

    coords_cart (3, ...): numpy array with the first
        dimension indication x,y,z
    """
    shape = coords_cart.shape
    coords = coords_cart.reshape(3,-1)

    lat, lon, alt = np.zeros_like(coords)
    for i in range(coords.shape[1]):
        p_rec = [coords[0, i], coords[1, i], coords[2, i]]
        p_lat = sp.spiceypy.reclat(p_rec)
        alt[i], lon[i], lat[i] = p_lat
        
    lat = lat*180/np.pi
    lon = lon*180/np.pi
    alt = alt - mars_r 

    coords_sphere = np.array([lat, lon, alt]).reshape(shape)
    return coords_sphere


def get_all_data(ds_names, ds_types, indxs, fields, **kwargs):
    """
    Get data for all fields for indexes that were 
    already found.
    """
    data = {f:{} for f in fields+['time']}

    for ds_type, keys in ds_types.items():
        for dsk in keys:

            dsf = ds_names[dsk]

            if ds_type == 'maven':
                ds = pd.read_csv(dsf)
                for field in fields:
                    ds_dat = get_ds_data(ds, field, indxs[ds_type],
                            maven=True, grid=False)
                    data[field][dsk] = ds_dat
                    time = get_ds_data(ds, 'time', indxs[ds_type],
                        maven=True, grid=False)
                time = time-time[0]
                time = time/time[-1]
                data['time'][dsk] = time
            


            else:
                for field in fields:
                    with h5py.File(dsf, 'r') as ds:
                        ds_dat = get_ds_data(ds, field, indxs[ds_type],
                                             grid=('helio' in ds_type) or ('rhybrid' in ds_type), **kwargs)
                                             #grid=ds_type=='heliosares', **kwargs)
                        data[field][dsk] = ds_dat

                data['time'][dsk] = np.linspace(0, 1, np.max(indxs[ds_type].shape))

    return data

