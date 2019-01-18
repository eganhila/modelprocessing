import numpy as np
import yt
#import spiceypy as sp
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from modelprocessing.misc.labels import *
from modelprocessing.misc.field_default_params import *

#sp.furnsh("/Users/hilaryegan/Projects/ModelChallenge/ModelProcessing/misc/maven_spice.txt")
mars_r = None
orbit_dir = '/Volumes/triton/Data/OrbitDat/Flythroughs/'
model_dir = '/Volumes/triton/Data/ModelChallenge/'
exo_dir = "/Users/hilaryegan/Data/PlanetSize/" #'/Volumes/triton/Data/Exoplanets/'

mH = 1.66053904e-27
ion_mass = {'H_p1':mH, 'O_p1':16*mH, 'O2_p1':32*mH}

def load_data(ds_name, field=None, fields=None, vec_field=None):
    """
    Load data for a standard hdf5 dataset into a dictionary

    ds_name (string): full file name to be loaded
    field (optional, string): field to load
    fields (optional, list): fields to load
    vec_field (boolean, default False): if not none, automatically load 
        vec_field + "_x", "_y", and "_z".
    """
    ds = {}
    ds['attrs'] = {}
    with h5py.File(ds_name, 'r') as f:
        try:
            mars_r = float(f.attrs['radius'])
        except KeyError:
            print("Can't load radius from h5 file, assuming Mars Radius")
            mars_r = 3390.0
        for k,v in f.attrs.items():
            ds['attrs'][k]=v
        #ds['attrs']['scale_height'] = 300*3390/mars_r
        if 'xmesh' in f.keys():
            ds['x'] = f['xmesh'][:]/mars_r
            ds['y'] = f['ymesh'][:]/mars_r
            ds['z'] = f['zmesh'][:]/mars_r
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

        if fields is None:
            fields = []
        elif fields == 'all':
            fields = [k for k in f.keys() if 'mesh' not in k and k not in ['x','y','z']]

        if vec_field is not None:
            for v in ['_x', '_y', '_z']: fields.append(vec_field+v)
            
        if field is not None: fields.append(field)

        for field in fields:
            ds[field] = get_ds_data(f, field, None, grid=grid, normal=norm, 
                        ion_velocity=ion_v)
            
    return ds

def get_datasets(load_key=None, maven=False):
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
    if load_key == 'R2349': 
        ds_names['batsrus_mf_lr'] =  model_dir+'R2349/batsrus_3d_multi_fluid_lowres.h5'
        ds_names['batsrus_multi_species'] =  model_dir+'R2349/batsrus_3d_multi_species.h5'
        ds_names['batsrus_electron_pressure'] =  model_dir+'R2349/batsrus_3d_pe.h5'
        ds_names['heliosares'] ='/Volumes/triton/Data/ModelChallenge/R2349/heliosares_multi.h5'
        #ds_names['rhybrid'] ='/Volumes/triton/Data/ModelChallenge/R2349/rhybrid.h5'
        
        ds_types = {'batsrus1':[key for key in ds_names.keys() if 'multi_fluid' in key],
                    'batsrus2':[key for key in ds_names.keys() if 'multi_species' in key],
                    'batsrus3':[key for key in ds_names.keys() if 'electron_pressure' in key],
                    'batsrus4':[key for key in ds_names.keys() if 'mf_lr' in key],
                    'heliosares':[key for key in ds_names.keys() if 'helio' in key],
                    'rhybrid_helio':[key for key in ds_names.keys() if 'rhybrid' in key ]}
        if maven or True:
            ds_names['maven']=orbit_dir+'orbit_2349.csv'
            #ds_names['maven'] = orbit_dir+'orbit_plume_2349.csv'
            ds_types['maven']=['maven']
    elif load_key == 'batsrus_mf_lowres':
        ds_names['batsrus_mf_lr'] =  model_dir+'R2349/batsrus_3d_multi_fluid_lowres.h5'
        ds_types = {'batsrus_mf_lr' : ['batsrus_mf_lr']}


    elif load_key ==  'helio_multi':
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
            #ds_names['maven'] = orbit_dir+'orbit_2349.csv'
            ds_types['maven']=['maven']
    elif load_key == 'SDC_BATS':
        ds_names['LS180_SSL000_max'] = model_dir+'SDC_Archive/BATSRUS/LS180_SSL000_max.h5'
        ds_names['LS270_SSL000_max'] = model_dir+'SDC_Archive/BATSRUS/LS270_SSL000_max.h5'
        ds_names['LS090_SSL000_max'] = model_dir+'SDC_Archive/BATSRUS/LS090_SSL000_max.h5'
        ds_names['LS180_SSL270_max'] = model_dir+'SDC_Archive/BATSRUS/LS180_SSL270_max.h5'
        ds_names['LS270_SSL270_max'] = model_dir+'SDC_Archive/BATSRUS/LS270_SSL270_max.h5'
        ds_names['LS090_SSL270_max'] = model_dir+'SDC_Archive/BATSRUS/LS090_SSL270_max.h5'
        ds_names['LS180_SSL180_max'] = model_dir+'SDC_Archive/BATSRUS/LS180_SSL180_max.h5'
        ds_names['LS270_SSL180_max'] = model_dir+'SDC_Archive/BATSRUS/LS270_SSL180_max.h5'
        ds_names['LS090_SSL180_max'] = model_dir+'SDC_Archive/BATSRUS/LS090_SSL180_max.h5'
        ds_names['LS180_SSL000_min'] = model_dir+'SDC_Archive/BATSRUS/LS180_SSL000_min.h5'
        ds_names['LS270_SSL000_min'] = model_dir+'SDC_Archive/BATSRUS/LS270_SSL000_min.h5'
        ds_names['LS090_SSL000_min'] = model_dir+'SDC_Archive/BATSRUS/LS090_SSL000_min.h5'
        ds_names['LS180_SSL270_min'] = model_dir+'SDC_Archive/BATSRUS/LS180_SSL270_min.h5'
        ds_names['LS270_SSL270_min'] = model_dir+'SDC_Archive/BATSRUS/LS270_SSL270_min.h5'
        ds_names['LS090_SSL270_min'] = model_dir+'SDC_Archive/BATSRUS/LS090_SSL270_min.h5'
        ds_names['LS180_SSL180_min'] = model_dir+'SDC_Archive/BATSRUS/LS180_SSL180_min.h5'
        ds_names['LS270_SSL180_min'] = model_dir+'SDC_Archive/BATSRUS/LS270_SSL180_min.h5'
        ds_names['LS090_SSL180_min'] = model_dir+'SDC_Archive/BATSRUS/LS090_SSL180_min.h5'

        ds_types = {'batsrus':[key for key in ds_names.keys()]}

    elif load_key == 'SDC_G1':
        #BATSRUS
        ds_names['bats_min_LS270_SSL0'] = \
                model_dir+'SDC_Archive/BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG0.h5'
        ds_names['bats_min_LS270_SSL180'] = \
            model_dir+'SDC_Archive/BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG180.h5'
        ds_names['bats_min_LS270_SSL270'] = \
                model_dir+'SDC_Archive/BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG270.h5'        
        
        #HELIOSARES
        #ds_names['helio_1'] = \
        #        model_dir+'SDC_Archive/HELIOSARES/Hybrid/'+'helio_1.h5'
        
        #ds_names['helio_2'] = \
        #        model_dir+'SDC_Archive/HELIOSARES/Hybrid/'+'helio_2.h5'
            
        
        ds_types = {'batsrus1':[key for key in ds_names.keys() if 'bats' in key],
                    'heliosares':[key for key in ds_names.keys() if 'helio' in key]}
        if maven:
            pass
            #ds_names['maven'] = orbit_dir+'orbit_2349.csv'
            #ds_types['maven']=['maven']

    elif load_key == 'rhybrid_res':
        ds_names = {'rhybrid240':'/Volumes/triton/Data/ModelChallenge/R2349/rhybrid.h5',
                    'rhybrid120':'/Volumes/triton/Data/ModelChallenge/R2349/HYB/state00030000.h5'}
        ds_types = {'rhybrid1':['rhybrid240'], 'rhybrid2':['rhybrid120']}
    elif load_key == 'batsrus_tseries':
        ds_names = {'batsrus_mf':'/Volumes/triton/Data/ModelChallenge/R2349/BATSRUS/10km_mf/3d__ful_4_n00040000.h5',
                    'batsrus_ms':'/Volumes/triton/Data/ModelChallenge/R2349/BATSRUS/10km_ms/3d__mhd_6_n0050000.h5'}
        ds_types = {'batsrus_mf':['batsrus_mf'], 'batsrus_ms':['batsrus_ms']}

    elif load_key == 'maven':
        ds_names, ds_types = {},{}
        ds_names['maven'] = orbit_dir+'orbit_2349.csv'
        ds_types['maven']=['maven']
    elif load_key == 'exo_2349':
        keys = ['2349_1RM_225km','2349_1RM_450km', '2349_2RM_450km',
                '2349_2RM_900km','2349_4RM_900km'] 
        ds_names = {k:exo_dir+'/'+k+'/'+k+'.h5' for k in keys}
        ds_types = {k:[k] for k in keys}
    elif load_key == 'exo_comparisonA':
        keys = ['2349_1RM_225km', '2349_2RM_450km',
                '2349_1.5RM_338km'] 
        ds_names = {k:exo_dir+'/ComparisonA/'+k+'.h5' for k in keys}
        ds_types = {k:[k] for k in keys}
    elif load_key == 'exo_comparisonB':
        keys = ['2349_1RM_225km', 'T0_1RM_225km', 'T1_1RM_225km', "T2_1RM_225km"] 
        ds_names = {k:exo_dir+'/ComparisonB/'+k+'.h5' for k in keys}
        ds_types = {k:[k] for k in keys}

    elif load_key == 'exo_t1':
        keys = ['T1_1RM_112km', 'T1_1RM_225km', #'T1_1RM_450km',
                'T1_2RM_225km', 'T1_2RM_450km', #'T1_2RM_900km',
                'T1_4RM_900km']

        ds_names = {k:exo_dir+'/'+k+'/'+k+'.h5' for k in keys}
        ds_types = {k:[k] for k in keys}

    else:
        print('No datasets selected')
    

    return (ds_names, ds_types)

def yt_load(ds_name, fields, use_ftype=False, periodic=True):

    try:
        with h5py.File(ds_name) as ds:
            mars_r = ds.attrs['radius']
    except:
        print("Can't load radius from h5 file, assuming Mars Radius")
        mars_r = 3390.0

    data = load_data(ds_name, fields=fields)

    xlim = (data['x'].min(), data['x'].max())
    ylim = (data['y'].min(), data['y'].max())
    zlim = (data['z'].min(), data['z'].max())

    bbox = np.array([xlim, ylim, zlim])

    for test_field in data.keys():
        if test_field not in ["attrs","x","y","z"]: break

    shape = data[test_field].shape #data['H_p1_number_density'].shape
    attrs = data['attrs']

    data = {("gas",k):v for k,v in data.items() if k not in ["attrs", 'x', 'y', 'z']} 

    for f in data.keys():
        if "velocity" in f[1]: data[f] = (data[f], "km/s")
        if "number_density" in f[1]: data[f] = (data[f], "cm**-3")
        if "magnetic_field" in f[1]: data[f] = (data[f], "nT")

    if periodic: periodicity = (True, True,True)
    else: periodicity = (False, False, False)


    ds = yt.load_uniform_grid(data, shape, mars_r*1e5, 
                              bbox=bbox, periodicity=periodicity)
    ds.my_attributes = attrs 
    return ds


def cart_geo_vec_transform(ds, prefix, indx):
    if 'xmesh' in ds.keys():
        x, y, z = ds['xmesh'][:].flatten()[indx], ds['ymesh'][:].flatten()[indx], ds['zmesh'][:].flatten()[indx]
    else:
        x, y, z = ds['x'][:].flatten()[indx], ds['y'][:].flatten()[indx], ds['z'][:].flatten()[indx]
    vx, vy, vz = ds[prefix+'_x'][:].flatten()[indx], ds[prefix+'_y'][:].flatten()[indx], ds[prefix+'_z'][:].flatten()[indx]
    v = np.array([vx,vy,vz])
    
    lat = -1*(np.arctan2(np.sqrt(x**2+y**2), z))+np.pi/2  #theta
    lon = np.arctan2(y, x)   #phi
    #alt = (np.sqrt(x**2+y**2+z**2)-1)*mars_r

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
    return ds.loc[:,field].values[::2]

def get_path_idxs(coords, ds_names, ds_types):
    indxs = {}
    for ds_type, keys in ds_types.items():
        if ds_type == 'maven': continue
        if len(keys) == 0: continue
        print('getting indxs: '+ds_type)
        indxs[ds_type] = bin_coords(coords, ds_names[keys[0]], 
                                    grid='batsrus' not in ds_type)
                                    #grid=ds_type=='heliosares')
    indxs['maven'] = 'all'
    return indxs


def get_ds_data(ds, field, indx, grid=True, normal=None, ion_velocity=True,
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
        if field == 'H_p1_number_density' and grid==False and maven==False:
            return 2*  apply_indx(ds, field, indx)
        return apply_indx(ds, field, indx)
    elif 'weighted' in field:
        a_field, b_field = field.split('_weighted_')
        a = get_ds_data(ds, a_field,  indx, grid, maven=maven)
        b = get_ds_data(ds, b_field,  indx, grid, maven=maven)
        
        return a*b/a.sum()
    elif '_normalized' == field[-11:]:
        base_fname = field[:-11]
        base_f = get_ds_data(ds, base_fname, indx, grid, 
                area=area, normal=normal,
                ion_velocity=ion_velocity, maven=maven)
        norm = ds.attrs[base_fname+'_norm']
        return base_f/norm

    elif field == 'electron_pressure':
        return get_ds_data(ds, ' electron_pressure', indx, grid=grid, maven=maven)
    elif '_total' in field and field.replace('_total', '_x') in ds.keys():
        vx = get_ds_data(ds, field.replace('_total', '_x'), indx, grid=grid, maven=maven)
        vy = get_ds_data(ds, field.replace('_total', '_y'), indx, grid=grid, maven=maven)
        vz = get_ds_data(ds, field.replace('_total', '_z'), indx, grid=grid, maven=maven)
        return np.sqrt(vx**2+vy**2+vz**2)
    elif '_normal' in field: # field.replace('_normal', '_x') in ds.keys():
        vx = get_ds_data(ds, field.replace('_normal', '_x'), indx, grid=grid, maven=maven)
        vy = get_ds_data(ds, field.replace('_normal', '_y'), indx, grid=grid, maven=maven)
        vz = get_ds_data(ds, field.replace('_normal', '_z'), indx, grid=grid, maven=maven)

        v = np.array([vx,vy,vz])
        vn = np.sum(normal*v, axis=0)
        return vn
    elif '_Bperp' == field[-6:]:
        Bperp_hat = ds.attrs['Bperp_hat']
        base_field = field[:-5]
        bf_y = get_ds_data(ds, base_field+'y', indx, grid=grid, maven=maven)
        bf_z = get_ds_data(ds, base_field+'z', indx, grid=grid, maven=maven)

        return Bperp_hat[1]*bf_y+Bperp_hat[2]*bf_z

    elif '_Esw' == field[-4:]:
        Esw_hat = ds.attrs['Esw_hat']
        base_field = field[:-4]
        bf_y = get_ds_data(ds, base_field+'_y', indx, grid=grid, maven=maven)
        bf_z = get_ds_data(ds, base_field+'_z', indx, grid=grid, maven=maven)

        return Esw_hat[1]*bf_y+Esw_hat[2]*bf_z



    elif '_flux' in field:
        vn = 1e5*get_ds_data(ds, velocity_field+'_normal', indx, grid, 
                             normal, ion_velocity, maven=maven)
        dens = get_ds_data(ds, field.replace('flux', "number_density"), indx, grid, maven=maven)
        
        return vn*dens
    elif "power" in field:
        flux =  get_ds_data(ds, field.replace("power", "flux"), indx, grid,
                            normal, ion_velocity, maven=maven)
        vmag = get_ds_data(ds, velocity_field+"_total", indx, grid, normal,
                            ion_velocity, maven=maven)

        return flux*vmag*vmag

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


    elif field == 'ram_pressure':
        #ions = ['O2_p1', 'O_p1', 'H_p1', 'CO2_p1']
        ions = ['H_p1']

        pr = np.zeros_like(get_ds_data(ds, 'magnetic_field_x', indx, grid=grid, maven=maven))
        for ion in ions:
            rho_i = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven)
            v_i = get_ds_data(ds, ion+'_velocity_total', indx, grid=grid, maven=maven)

            if rho_i.shape != v_i.shape:
                continue

            if rho_i.shape == pr.shape:
                pr += rho_i*v_i**2


        return pr/1e6
    elif 'kinetic_energy_density' in field:
        ion = field[:-23]
        vfield = ion+'_velocity'
        vtot = get_ds_data(ds, vfield+"_total", indx, grid=grid, maven=maven)
        dens = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven)

        return 0.5*vtot**2*dens*ion_mass[ion]/1e6

    elif 'gyroradius' in field:
        ion = field[:-11]
        v = np.array([get_ds_data(ds, ion+'_v_-_fluid_v_'+vec, indx, grid=grid, maven=maven) for vec in ['x','y','z']]) 
        n = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven) 
        mask = n==0

        B = np.array([get_ds_data(ds, 'magnetic_field_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])

        v_dot_B = v[0]*B[0]+v[1]*B[1]+v[2]*B[2]
        Btot = np.sqrt(B[0]**2+B[1]**2+B[2]**2)

        vperp = np.sqrt(np.sum(np.array([v[i] - v_dot_B[i]/Btot for i in range(3)])**2, axis=0))
        r = ion_mass[ion]*vperp/(1.6e-19*Btot)
        r[mask] = np.nan
        return r*1e6


    #elif '_'.join(field.split('_')[2:]) in ds.keys() and '_'.join(field.split('_')[2:]) not in ['x','y','z']:
    #    print 'what'+'_'.join(field.split('_')[2:])
    #    return apply_indx(ds, '_'.join(field.split('_')[2:]), indx)
    elif field == 'magnetic_pressure':
        return get_ds_data(ds, 'magnetic_field_total', indx, grid=grid, maven=maven)**2/(2*1.26E-6*1e9)
    elif field == 'electron_pressure_frac':
        pe = get_ds_data(ds, 'electron_pressure', indx, grid=grid, maven=maven)
        pt = get_ds_data(ds, 'thermal_pressure', indx, grid=grid, maven=maven)
        return pe/pt
    elif field == 'total_pressure':
        if maven: return np.array([])
        pe = get_ds_data(ds, 'electron_pressure', indx, grid=grid, maven=maven)
        pt = get_ds_data(ds, 'thermal_pressure', indx, grid=grid, maven=maven)
        pb = get_ds_data(ds, 'magnetic_pressure', indx, grid=grid, maven=maven)
        pr = get_ds_data(ds, 'ram_pressure', indx, grid=grid, maven=maven)
        p = pb
        if pe.shape == p.shape: p += pe
        if pt.shape == p.shape: p += pt
        if pr.shape == p.shape: p += pr

        return p
    elif 'temperature' in field:
        ion = '_'.join(field.split('_')[:2])
        if ion+'_pressure' in ds.keys() and ion + '_number_density' in ds.keys():
            p = get_ds_data(ds, ion+'_pressure', indx, grid=grid, maven=maven)
            n = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven)
            k = 1.38e-23

            return 1e-15*p/(n*k)

    elif field == "standoff_pressure_fraction":
        pb = get_ds_data(ds, 'magnetic_pressure', indx, grid=grid, maven=maven)
        pr = get_ds_data(ds, 'ram_pressure', indx, grid=grid, maven=maven)

        return pb/pr

    elif field == "velocity_fraction":
        pi = get_ds_data(ds, "O2_p1_velocity_total", indx,grid=grid)
        pe = get_ds_data(ds, "electron_velocity_total", indx, grid=grid)
        return pe/pi

    elif field == 'thermal_pressure':
        if 'pressure' in ds.keys(): 
            return get_ds_data(ds, 'pressure', indx, grid=grid, maven=maven)
        elif 'H_p1_temperature' in ds.keys():
            ions = ['H_p1', 'O_p1', 'O2_p1']
            n_i = np.nan_to_num(np.array([get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven) for ion in ions]))
            T_i = np.nan_to_num(np.array([get_ds_data(ds, ion+'_temperature', indx, grid=grid, maven=maven) for ion in ions]))
            k = 1.38*1e-4
            p = np.sum(n_i*k*T_i, axis=0)
            return p
        else:
            return np.array([])

    elif  'J_cross_B' in field:

        J = np.array([get_ds_data(ds, 'current_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])
        B = np.array([get_ds_data(ds, 'magnetic_field_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])
        ion = '_'.join(field.split('_')[:2])
        #n = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven) 

        if field[-1] == 'x': v = J[1]*B[2]-J[2]*B[1]
        if field[-1] == 'y': v = J[2]*B[0]-J[0]*B[2]
        if field[-1] == 'z': v = J[0]*B[1]-J[1]*B[0]
        if 'total' in field: 
            v0 = J[1]*B[2]-J[2]*B[1]
            v1 = J[2]*B[0]-J[0]*B[2]
            v2 = J[0]*B[1]-J[1]*B[0]
            v = np.sqrt(v0**2+v1**2+v2**2)

        return v*6.2#/n
    elif "motional_electric_field" in field:
        v = np.array([get_ds_data(ds, 'fluid_velocity_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])
        B = np.array([get_ds_data(ds, 'magnetic_field_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])

        if field[-1] == 'x': x = (v[1]*B[2]-v[2]*B[1])
        if field[-1] == 'y': x = (v[2]*B[0]-v[0]*B[2])
        if field[-1] == 'z': x = (v[0]*B[1]-v[1]*B[0])
        if 'total' in field: 
            x0 = v[1]*B[2]-v[2]*B[1]
            x1 = v[2]*B[0]-v[0]*B[2]
            x2 = v[0]*B[1]-v[1]*B[0]
            x = np.sqrt(x0**2+x1**2+x2**2)

        return x


    elif 'v_-_fluid_v' in field:
        ion = '_'.join(field.split('_')[:2])

        v_ion = np.array([get_ds_data(ds, ion+'_velocity_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])
        v_fluid = np.array([get_ds_data(ds, 'fluid_velocity_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])

        v = v_ion - v_fluid

        if field[-1] == 'x': return v[0] 
        if field[-1] == 'y': return v[1] 
        if field[-1] == 'z': return v[2] 
        if 'total' in field: return np.sqrt(np.sum(v**2, axis=0))

    elif 'fluid_velocity' in field:
        return get_ds_data(ds, field.replace('fluid', 'H_p1'), indx, grid=grid, maven=maven)
        if grid == True: 
            return get_ds_data(ds, field.replace('fluid', 'avg_ion'), indx, grid=grid, maven=maven)
        if 'total' in field:
            ue = [get_ds_data(ds, field.replace('fluid', 'electron').replace('total', ax), indx, grid=grid, maven=maven) for ax in ['x','y','z']]
            ubar_i = [get_ds_data(ds, field.replace('fluid', 'avg_ion').replace('total', ax), indx, grid=grid, maven=maven) for ax in ['x','y','z']]

            return 0.5*np.sqrt(np.sum((np.array(ue)+np.array(ubar_i))**2, axis=0))
        else:
            ue = get_ds_data(ds, field.replace('fluid', 'electron'), indx, grid=grid, maven=maven)
            ubar_i = get_ds_data(ds, field.replace('fluid', 'avg_ion'), indx, grid=grid, maven=maven)
            return 0.5*(ue+ubar_i)

    elif 'electron_velocity' in field:

       ax = field[-1:]

       J = get_ds_data(ds, 'current_'+ax, indx, grid=grid, maven=maven)
       ubar_i = get_ds_data(ds, 'avg_ion_velocity_'+ax, indx, grid=grid, maven=maven)

       return 2*J+ubar_i

    elif 'avg_ion_velocity' in field:
        if 'velocity_x' in ds.keys() and False:
            return get_ds_data(ds, field.replace('avg_ion_velocity', 'velocity'), indx, grid=grid, maven=maven)
        if 'xmesh' in ds.keys() and 'electron_velocity_z' in ds.keys():
            return get_ds_data(ds, field.replace('avg_ion_velocity', 'H_p1_velocity'), indx, grid=grid, maven=maven)

        ions = ['H_p1', 'O2_p1', 'O_p1']
        if 'CO2_p1_number_density' in ds.keys(): ions.append('CO2_p1')

        ax = field[-1:]

        data = np.zeros_like(get_ds_data(ds, 'H_p1_number_density', indx, grid=grid, maven=maven))
        nsum = np.zeros_like(data)

        for ion in ions:
            v = get_ds_data(ds, ion+'_velocity_'+ax, indx, grid=grid, maven=maven)
            n = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven)

            data += v*n
            nsum += n

        return data/nsum
        

    elif  'v_cross_B' in field:
        ion = '_'.join(field.split('_')[:2])

        v = np.array([get_ds_data(ds, ion+'_v_-_fluid_v_'+vec, indx, grid=grid, maven=maven) for vec in ['x','y','z']]) 

        B = np.array([get_ds_data(ds, 'magnetic_field_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])

        if field[-1] == 'x': x = (v[1]*B[2]-v[2]*B[1])
        if field[-1] == 'y': x = (v[2]*B[0]-v[0]*B[2])
        if field[-1] == 'z': x = (v[0]*B[1]-v[1]*B[0])
        if 'total' in field: 
            x0 = v[1]*B[2]-v[2]*B[1]
            x1 = v[2]*B[0]-v[0]*B[2]
            x2 = v[0]*B[1]-v[1]*B[0]
            x = np.sqrt(x0**2+x1**2+x2**2)

        n = get_ds_data(ds, ion+'_number_density', indx, grid=grid)
        x[n==0] = np.nan

        return x*1e-6


    elif field == 'v_sub_total':
        ion = 'O2_p1'
        v_ion = np.array([get_ds_data(ds, ion+'_velocity_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])
        v_fluid = np.array([get_ds_data(ds, 'velocity_'+vec, indx, grid=grid, maven=maven) \
                      for vec in ['x','y','z']])
        v = v_ion - v_fluid
        return np.sqrt(np.sum(v**2,axis=0))
        
    elif 'electron_velocity' in field and 'current_x' in ds.keys():
        vec = field[-1]
        u = get_ds_data(ds, 'velocity_'+vec, indx, grid=grid, maven=maven)
        J = get_ds_data(ds, 'current_'+vec, indx, grid=grid, maven=maven)
        n = get_ds_data(ds, 'number_density', indx, grid=grid, maven=maven)
        return u-(J/n)/6.24e6

    elif 'hall_velocity' in field and 'current_x' in ds.keys():
        vec = field[-1]
        J = get_ds_data(ds, 'current_'+vec, indx, grid=grid, maven=maven)
        n = get_ds_data(ds, 'number_density', indx, grid=grid, maven=maven)
        return (J/n)/6.24e6
    
    elif 'velocity_frac' in field:
        vec = field[-1]
        vfield = field[:-7]
        ion = field[:-16]
        vtot = get_ds_data(ds, vfield+"_total", indx, grid=grid, maven=maven)
        vvec = get_ds_data(ds, vfield+'_'+vec, indx, grid=grid, maven=maven)
        dens = get_ds_data(ds, ion+'_number_density', indx, grid=grid, maven=maven)
        return np.abs(vvec/vtot)

    elif field == "magnetic_field_orientation":
        Br = get_ds_data(ds, "magnetic_field_normal", indx, 
                         grid=grid, maven=maven, normal=normal)
        Btot = get_ds_data(ds, "magnetic_field_total", indx,
                            grid=grid, maven=maven)

        return Br/Btot

    elif field == 'magnetic_field_elevation_angle':
        B_frac = get_ds_data(ds, "magnetic_field_orientation", indx,
                grid=grid, maven=maven, normal=normal)
        return 180*np.arcsin(B_frac)/np.pi


    elif field == "magnetic_field_open":
        orientation = get_ds_data(ds, "magnetic_field_orientation", indx,
                                 grid=grid, maven=maven, normal=normal)

        o = np.abs(orientation)>0.25
        return o

        

    elif 'density' in field and field != 'density':
        return get_ds_data(ds, 'density', indx, grid=grid, maven=maven)
    elif 'velocity_x' in field and field != 'velocity_x':
        return get_ds_data(ds, 'velocity_x', indx, grid=grid, maven=maven)
    elif 'velocity_y' in field and field != 'velocity_y':
        return get_ds_data(ds, 'velocity_y', indx, grid=grid, maven=maven)
    elif 'velocity_z' in field and field != 'velocity_z':
        return get_ds_data(ds, 'velocity_z', indx, grid=grid, maven=maven)
    elif 'velocity_total' in field and field != 'velocity_total':
        return get_ds_data(ds, 'velocity_total', indx, grid=grid, maven=maven)
    else:
        if maven: dstype = 'maven'
        elif grid: dstype = 'heliosares'
        else: dstype= 'batsrus'
        print("Field {0} not found in {1}".format(field, dstype))
        return np.array([])
        #raise(ValueError)

def adjust_spherical_positions(pos, alt, Rm0):
    alt1 = min(alt)
    R = np.sqrt(np.sum(pos**2, axis=0))
    alt2 = min(R)-Rm0
    Rm = Rm0+alt2-alt1 
    
    return pos/Rm*Rm0

        

def get_orbit_coords(orbit, geo=False, Npts=250, units_rm=True, sim_mars_r=3396.0,
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
    Nskip = 2 #10000/Npts
    data = pd.read_csv(orbit_dir+'orbit_{0:04d}.csv'.format(orbit))[::Nskip]
    pos = np.array([data['x'], data['y'], data['z']])
    time = data['time'].values
    time_adj = (time-time[0])/(time[-1]-time[0])
    alt = data['altitude']
     
    if adjust_spherical:
        pos = adjust_spherical_positions(pos, alt, sim_mars_r)

    if units_rm:
        pos = pos/sim_mars_r

    if return_time: return (pos,time, time_adj)
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
    else: 
        with h5py.File(dsf, 'r') as dataset:
            x = dataset['x'][:].flatten()
            y = dataset['y'][:].flatten()
            z = dataset['z'][:].flatten()

        return bin_coords_nogrid(coords,np.array([x,y,z]))
    
def bin_coords_grid(coords, dsf, method='nearest'):
    with h5py.File(dsf, 'r') as dataset:
        mars_r = float(dataset.attrs['radius'])
        x = dataset['xmesh'][:,0,0]/mars_r
        y = dataset['ymesh'][0,:,0]/mars_r
        z = dataset['zmesh'][0,0,:]/mars_r
        mesh_shape = (dataset['xmesh'].shape)
        
    idx = np.zeros((3, coords.shape[-1]))
    
    for i in range(coords.shape[-1]):
        if method=='nearest':
            idx_x = np.argmin((coords[0,i]-x)**2)
            idx_y = np.argmin((coords[1,i]-y)**2)
            idx_z = np.argmin((coords[2,i]-z)**2)
        elif method == 'left':
            factors = [(coords[0,i]-x)/np.abs(coords[0,i]-x),
                       (coords[1,i]-y)/np.abs(coords[1,i]-y),
                       (coords[2,i]-z)/np.abs(coords[2,i]-z)]

            idx_x = np.argmin((coords[0,i]-x)**2*factors[0])
            idx_y = np.argmin((coords[1,i]-y)**2*factors[1])
            idx_z = np.argmin((coords[2,i]-z)**2*factors[2])
        elif method == 'right':
            factors = [(x-coords[0,i])/np.abs(coords[0,i]-x),
                       (y-coords[1,i])/np.abs(coords[1,i]-y),
                       (z-coords[2,i])/np.abs(coords[2,i]-z)]

            idx_x = np.argmin((coords[0,i]-x)**2*factors[0])
            idx_y = np.argmin((coords[1,i]-y)**2*factors[1])
            idx_z = np.argmin((coords[2,i]-z)**2*factors[2])

        idx[:, i] = [idx_x, idx_y, idx_z]
            
    return idx.astype(int)


"""
def trilinear_interpolate(coords, dsf, fields):
    # get nearest points
    idx_left = bin_coords_grid(coords, dsf, method='left')
    with h5py.File(dsf, 'r') as dataset:
        dx = (dataset['xmesh'][1,0,0]-dataset['xmesh'][0,0,0])/mars_r

    x0 = get_ds_data(ds, 'xmesh', idx_left)
    y0 = get_ds_data(ds, 'ymesh', idx_left)
    z0 = get_ds_data(ds, 'zmesh', idx_left)

    xd = (coords[0]-x0)/dx
    yd = (coords[1]-y0)/dx
    zd = (coords[2]-z0)/dx

    idx_matrix = np.zeros(idx_)

    # get weights
    weights_left = np.abs()

    # get vals for each grid

    # average using weights
"""
def bin_coords_nogrid(coords, incoords):

    x,y,z = incoords
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
            print('Getting data for: ',dsk)

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
                        
                        if '_x' in field or '_y' in field or '_z' in field:
                            get_data_func = get_rotated_data
                        else: get_data_func = get_ds_data
                        try:
                            ds_dat = get_data_func(ds, field, indxs[ds_type],
                                                 grid='batsrus' not in ds_type, **kwargs)
                                             #grid=ds_type=='heliosares', **kwargs)
                        except ValueError:
                            ds_dat = np.array([])
                        data[field][dsk] = ds_dat

                data['time'][dsk] = np.linspace(0, 1, np.max(indxs[ds_type].shape))

    return data

def get_rotated_data(ds, field, indxs, **kwargs):
    fprefix = field[:-2]
    fsuffix = field[-1]

    dat = np.array([get_ds_data(ds, fprefix+'_x', indxs, **kwargs), 
                    get_ds_data(ds, fprefix+'_y', indxs, **kwargs),
                    get_ds_data(ds, fprefix+'_z', indxs, **kwargs)])

    dat = rotate_vec_simmso(dat)

    if fsuffix == 'x': return dat[0] 
    if fsuffix == 'y': return dat[1]
    if fsuffix == 'z': return dat[2]

def rotate_vec_simmso(vec):
   xz_theta = -1*np.pi*4.18/180
   xy_theta = -1*np.pi*5.1/180
   Rxz = np.array([[np.cos(xz_theta), 0, np.sin(xz_theta)],[0,1,0],[-np.sin(xz_theta),0,np.cos(xz_theta)]])
   Rxy = np.array([[np.cos(xy_theta), -1*np.sin(xy_theta),0],[np.sin(xy_theta), np.cos(xy_theta),0],[0,0,1]])
   return np.matmul(Rxz, np.matmul(Rxy, vec))



def rotate_coords_simmso(coords):

   xz_theta = np.pi*4.18/180
   xy_theta = np.pi*5.1/180
   Rxz = np.array([[np.cos(xz_theta), 0, np.sin(xz_theta)],[0,1,0],[-np.sin(xz_theta),0,np.cos(xz_theta)]])
   Rxy = np.array([[np.cos(xy_theta), -1*np.sin(xy_theta),0],[np.sin(xy_theta), np.cos(xy_theta),0],[0,0,1]])
   return np.matmul(Rxz, np.matmul(Rxy, coords))

def get_cycloid_positions(R, theta, alt, N=100, zlim=None, tlim=None):
    R_0 = float(alt+mars_r)
    x0 = R_0*np.cos(theta)
    z0 = R_0*np.sin(theta)
    
    if tlim is None and zlim is None: tlim = np.pi/2
    elif zlim is not None: 
        tlim = np.arccos(1-(zlim*mars_r-z0)/R)
    
    t = np.linspace(0, tlim, N)

    
    x = (-1*R*(t-np.sin(t)) + x0)/mars_r
    z = (R*(1-np.cos(t)) + z0)/mars_r

    return (x, z)

def get_cycloid_velocity(R,theta,alt,v0,N=100, zlim=None, tlim=None):
    R_0 = alt+mars_r
    x0 = R_0*np.cos(theta)
    z0 = R_0*np.sin(theta)
    
    if tlim is None and zlim is None: tlim = np.pi/2
    elif zlim is not None: 
        tlim = np.arccos(1-(zlim*mars_r-z0)/R)
    
    t = np.linspace(0, tlim, N)

    vx = -1*v0*(1-np.cos(t))
    vz = v0*(np.sin(t))
    
    return (vx,vz)

