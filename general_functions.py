import numpy as np
import spiceypy as sp
import h5py
import matplotlib.pyplot as plt
sp.furnsh("maven_spice.txt")
label_lookup = {'H_p1_number_density':r'$n(H+)\;\mathrm{cm^{-3}}$',
          'O2_p1_number_density':u'$n(O_2+)\;\mathrm{cm^{-3}}$',
          'O_p1_number_density':u'$n(O+)\;\mathrm{cm^{-3}}$',
          'O_p2_number_density':u'$n(O++)\;\mathrm{cm^{-3}}$',
          'CO2_p1_number_density':u'$n(CO_2+)\;\mathrm{cm^{-3}}$',
          'H_number_density':u'$n(H)\;\mathrm{cm^{-3}}$',
          'He_number_density':u'$n(He)\;\mathrm{cm^{-3}}$',
          'electron_number_density':u'$n(e-)\;\mathrm{cm^{-3}}$',
          'number_density':u'$n\;\mathrm{cm^{-3}}$',
          'magnetic_field_radial':u'$B_r \mathrm{(nT)}$',
          'magnetic_field_x':u'$B_X \mathrm{(nT)}$',
          'magnetic_field_y':u'$B_Y \mathrm{(nT)}$',
          'magnetic_field_z':u'$B_Z \mathrm{(nT)}$',
          'magnetic_field_total':u'$|B| \mathrm{(nT)}$',
          'bats_min_LS270_SSL0':'BATSRUS: SSL=0',# (Solar min, LS=270)',
          'bats_min_LS270_SSL180':'BATSRUS: SSL=180',# (Solar min, LS=270)',
          'bats_min_LS270_SSL270':'BATSRUS: SSL=270',# (Solar min, LS=270)',
          'helio_1':'HELIOSARES: 1',# (Solar mod, LS=270)',
          'helio_2':'HELIOSARES: 2',# (Solar mod, LS=270)',
          'maven': "MAVEN"}
label_lookup['altitude'] = '$\mathrm{Altitude}$'
label_lookup['magnetic_field_total'] = '$\mathrm{|B|\;(nT)}$'
label_lookup['number_density'] = '$\mathrm{n\;(cm^{-3})}$'
label_lookup['batsrus_multi_species'] = 'BATSRUS Multi-Species'
label_lookup['batsrus_multi_fluid'] = 'BATSRUS Multi-Fluid'
label_lookup['heliosares'] = 'HELIOSARES'
label_lookup['x'] = 'x'
label_lookup['y'] = 'y'
label_lookup['z'] = 'z'
field_lims = {'magnetic_field_x':(-50,50),#(-33,33),
              'magnetic_field_y':(-50,50),#(-33,33),
              'magnetic_field_z':(-50,50),#(-33,33),
              'H_p1_number_density':(1e-1,3e4),
              'O2_p1_number_density':(1e0,3e5),
              'O_p1_number_density':(1e0, 2e4),
              'CO2_p1_number_density':(1e0,6e3),
              'H_number_density':(1e-2, 1e5), 
              'He_number_density':(1e-2, 1e5),
              'electron_number_density':(1e-2, 1e5),
              'magnetic_field_total':(0,120)}

log_fields2 = ['H_p1_number_density',
               'O2_p1_number_density',
               'O_p1_number_density',
               'CO2_p1_number_density',
               'altitude',
               'number_density']
def load_data(ds_name, field=None, fields=None, vec_field=False):
    ds = {}
    with h5py.File(ds_name, 'r') as f:
        if 'xmesh' in f.keys():
            ds['x'] = f['xmesh'][:]/3390
            ds['y'] = f['ymesh'][:]/3390
            ds['z'] = f['zmesh'][:]/3390
        else:
            ds['x'] = f['x'][:]
            ds['y'] = f['y'][:]
            ds['z'] = f['z'][:]
            
        if vec_field:
            ds[field+'_x'] = f[field+'_x'][:]
            ds[field+'_y'] = f[field+'_y'][:]
            ds[field+'_z'] = f[field+'_z'][:]
        elif fields is not None:
            for field in fields:
                ds[field] = f[field][:]
        elif field is not None:
            ds[field] = f[field][:]
            
    return ds

def get_datasets(fdir='/Volumes/triton/Data/ModelChallenge/SDC_Archive/', new_models=False, maven=True):
    ds_names = {}
    if new_models:
        fdir = '/Volumes/triton/Data/ModelChallenge/R2349/'
        ds_names['batsrus_multi_fluid'] =  fdir+'batsrus_3d_multi_fluid.h5'
        ds_names['batsrus_multi_species'] =  fdir+'batsrus_3d_multi_species.h5'
        ds_names['heliosares'] =  fdir+'helio_r2349.h5'
        
        ds_types = {'batsrus1':[key for key in ds_names.keys() if 'multi_fluid' in key],
                    'batsrus2':[key for key in ds_names.keys() if 'multi_species' in key],
                    'heliosares':[key for key in ds_names.keys() if 'helio' in key]}
    else:

        #BATSRUS
        """
        ds_names['bats_max_LS270-SSL0'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_PERmax-SSLONG0.h5'
        """   
        ds_names['bats_min_LS270_SSL0'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG0.h5'
        ds_names['bats_min_LS270_SSL180'] = \
            fdir+'BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG180.h5'
        ds_names['bats_min_LS270_SSL270'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_PERmin-SSLONG270.h5'        
                
        """
        ds_names['bats_max_LS270-SSL180'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_PERmax-SSLONG180.h5'
        
        ds_names['bats_max_LS270-SSL270'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_PERmax-SSLONG270.h5'
        

        ds_names['bats_max_LS90-SSL0'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_APHmax-SSLONG0.h5'
        ds_names['bats_min_LS90_SSL0'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_APHmin-SSLONG0.h5'
        ds_names['bats_max_LS90-SSL180'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_APHmax-SSLONG180.h5'
        ds_names['bats_min_LS90_SSL180'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_APHmin-SSLONG180.h5'
        ds_names['bats_max_LS90-SSL270'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_APHmax-SSLONG270.h5'
        ds_names['bats_min_LS90_SSL270'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_APHmin-SSLONG270.h5'

        ds_names['bats_max_LS180-SSL0'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmax-SSLONG0.h5'
        ds_names['bats_min_LS180_SSL0'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmin-SSLONG0.h5'
        ds_names['bats_max_LS180-SSL180'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmax-SSLONG180.h5'
        ds_names['bats_min_LS180_SSL180'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmin-SSLONG180.h5'
        ds_names['bats_max_LS180-SSL270'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmax-SSLONG270.h5'
        ds_names['bats_min_LS180_SSL270'] = \
                fdir+'BATSRUS/'+'3d__ful_4_n00060000_AEQNmin-SSLONG270.h5'
        """
        
        #HELIOSARES
        ds_names['helio_1'] = \
                fdir+'HELIOSARES/Hybrid/'+'helio_1.h5'
        
        ds_names['helio_2'] = \
                fdir+'HELIOSARES/Hybrid/'+'helio_2.h5'
            
        
        ds_types = {'batsrus1':[key for key in ds_names.keys() if 'bats' in key],
                    'heliosares':[key for key in ds_names.keys() if 'helio' in key]}

    
    #MAVEN
    if maven:
        ds_names['maven'] = \
        "/Users/hilaryegan/Projects/ModelChallenge/ModelProcessing/Output/test_orbit.csv"
        #'/Volumes/triton/Data/ModelChallenge/Maven/orbit2_{0:04d}.h5'

    return (ds_names, ds_types)

def geo_cart_coord_transform(geo_coords, use_r=False):
    alt = geo_coords[:, 0]
    lat = geo_coords[:, 1]
    lon = geo_coords[:, 2]
    if use_r:
        R = alt
    else:
        R = alt + 3388.25
    phi = -1*(lat-90)*np.pi/180.0
    theta = lon*np.pi/180.0

    x = R*np.cos(theta)*np.sin(phi)
    y = R*np.sin(theta)*np.sin(phi)
    z = R*np.cos(phi)

    return np.array([x,y,z]).T

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

def apply_grid_indx(ds, field, indx):
    #print ds[field].shape, indx.shape
    dat = np.zeros(indx.shape[1])
    for i in range(indx.shape[1]):
        dat[i] = ds[field][:][indx[0,i], indx[1,i], indx[2,i]]
    return dat

def get_ds_data(ds, field, indx, grid=True, normal=None, velocity_field=None, area=None):
    
    if grid: apply_indx = apply_grid_indx
    else: apply_indx = apply_flat_indx
    
    if type(indx)==str:
        if field in ds.keys():  return ds[field][:].flatten()
        else: return np.array([])

    if field in ds.keys():
        return apply_indx(ds, field, indx)#ds[field][:].flatten()[indx]
    elif '_total' in field and field.replace('_total', '_x') in ds.keys():
        x = apply_indx(ds, field.replace('_total', '_x'), indx)**2
        y = apply_indx(ds, field.replace('_total', '_y'), indx)**2
        z = apply_indx(ds, field.replace('_total', '_z'), indx)**2
        return np.sqrt(x+y+z)
    elif '_normal' in field and field.replace('_normal', '_x') in ds.keys():
        vx = apply_indx(ds, field.replace('_normal', '_x'), indx)
        vy = apply_indx(ds, field.replace('_normal', '_y'), indx)
        vz = apply_indx(ds, field.replace('_normal', '_z'), indx)
        v = np.array([vx,vy,vz])
        vn = np.sum(normal*v, axis=0)
        return vn
    elif '_flux' in field:
        vn = 1e5*get_ds_data(ds, velocity_field+'_normal', indx, grid, normal)
        dens = get_ds_data(ds, field.replace('flux', "number_density"), indx, grid)
        return vn*dens
    elif 'area' == field:
        return area
                                   
#    elif '_radial' in field and field.replace('_radial', '_x') in ds.keys():
#        return cart_geo_vec_transform(ds,field.replace('_radial', ''), indx)[0]
#    elif '_latitudinal'in field and field.replace('_latitudinal', '_x') in ds.keys():
#        return cart_geo_vec_transform(ds,field.replace('_latitudinal', ''), indx)[1]
#    elif 'longitudinal' in field and field.replace('_longitudinal', '_x') in ds.keys():
#        return cart_geo_vec_transform(ds,field.replace('_longitudinal', ''), indx)[2]
    else:
        print "Field {0} not found".format(field)
        return np.array([])
        #raise(ValueError)

def get_path_pts_adj(orb, Npts=50, units_mr=True, return_alt=False):
    fdir = '/Volumes/triton/Data/ModelChallenge/Maven/Traj/'
    with h5py.File(fdir+'traj_{0:04d}.h5'.format(orb), 'r') as f_h5:
        x = f_h5['x'][:]
        y = f_h5['y'][:]
        z = f_h5['z'][:]
        alt = f_h5['alt'][:]
    coords = np.array([x,y,z])
    if return_alt:
        return (coords, alt)
    else:
        return coords
        

def get_path_pts_reg(trange, geo=False, Npts=50, units_mr=False, sim_mars_r=3396.0,
                 adjust_spherical=True):
    if type(trange[0]) == str:
        et1, et2 = sp.str2et(trange[0]), sp.str2et(trange[1])
    else:
        et1, et2 = trange
    times = np.linspace(et1, et2, Npts)

    positions, lightTimes = sp.spkpos('Maven', times, 'MAVEN_MSO',
                                'NONE', 'MARS BARYCENTER')
    if units_mr:
        positions = positions/3390
    if not geo:
        return positions.T, times+946080000 +647812

    geo_coords = np.zeros_like(positions)

    for i in enumerate(positions):
	geo_coords[i,:] = sp.spiceypy.reclat(p)

	geo_coords[:,0] = (geo_coords[:,0]-3388.25)
	geo_coords[:,1] = (geo_coords[:,1]+np.pi)*180/np.pi
	geo_coords[:,2] = (geo_coords[:,2])*180/np.pi
	geo_coords = geo_coords[:,[0,2,1]].T

	return geo_coords, times+946080000+647812


def bin_coords(coords, dsf, grid=True):
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


def get_orbit_times(orbits, J200=False):

    trange_dat = np.loadtxt('/Users/hilaryegan/Projects/ModelChallenge/ModelProcessing/Output/orbit_times.csv',
			delimiter=',', unpack=True)
    t_peri= trange_dat[1, orbits]
    t_pre = trange_dat[1, orbits-1]
    t_post = trange_dat[1, orbits+1]

    tranges_J2000 = np.array([0.5*(t_peri+t_pre), 0.5*(t_peri+t_post)])
    tranges_utc = tranges_J2000 - (946080000+647812)

    if J200: return tranges_J2000
    else: return tranges_utc
