import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp
import h5py
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from general_functions import *
from flythrough_compare import *
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from itertools import product as iproduct
plt.style.use(['seaborn-poster', 'poster'])

label_lookup['H_p1_velocity_total'] = '$\mathrm{|v(H+)|\;(km/s)}$'
label_lookup['H_p1_velocity_normal'] = '$\mathrm{v(H+)\cdot \hat{r}\;(km/s)}$'
label_lookup['H_p1_flux'] = "$\mathrm{\Phi(H+)\;(\#/ (cm^{2} s) }$"
label_lookup['O2_p1_velocity_total'] = '$\mathrm{|v(O_2+)|\;(km/s)}$'
label_lookup['O2_p1_velocity_normal'] = '$\mathrm{v(O_2+)\cdot \hat{r}\;(km/s)}$'
label_lookup['O2_p1_flux'] = "$\mathrm{\Phi(O_2+)\;(\#/ (cm^{2} s) }$"
label_lookup['CO2_p1_velocity_total'] = '$\mathrm{|v(CO_2+)|\;(km/s)}$'
label_lookup['CO2_p1_velocity_normal'] = '$\mathrm{v(CO_2+)\cdot \hat{r}\;(km/s)}$'
label_lookup['CO2_p1_flux'] = "$\mathrm{\Phi(CO_2+)\;(\#/ (cm^{2} s) }$"
label_lookup['O_p1_velocity_total'] = '$\mathrm{|v(O+)|\;(km/s)}$'
label_lookup['O_p1_velocity_normal'] = '$\mathrm{v(O+)\cdot \hat{r}\;(km/s)}$'
label_lookup['O_p1_flux'] = "$\mathrm{\Phi(O+)\;(\#/ (cm^{2} s) }$"
label_lookup['area'] = '$\mathrm{dA}\;(cm^{2})$'
label_lookup['velocity_total'] = '$\mathrm{|v|\;(km/s)}$'
label_lookup['velocity_normal'] = '$\mathrm{v\cdot \hat{r}\;(km/s)}$'

diverging_fields = ['H_p1_flux', 'H_p1_velocity_normal']
log_fields = ['H_p1_number_density', 'H_p1_velocity_total']

def create_sphere_mesh(r):
    lon = np.arange(0,361, 5)
    lat = np.arange(-90,91, 5)
    phi = -1*(lat-90)*np.pi/180.0
    theta = lon*np.pi/180.0
    phi_v, theta_v = np.meshgrid(phi, theta)

    #Make face centers
    phi_f = 0.5*(phi_v[1:,1:]+phi_v[:-1,:-1])
    theta_f = 0.5*(theta_v[1:,1:]+theta_v[:-1,:-1])
    lat_f = -1*phi_f*180/np.pi+90
    lon_f = theta_f*180/np.pi

    x = (r*np.cos(theta_f)*np.sin(phi_f)).flatten()
    y = (r*np.sin(theta_f)*np.sin(phi_f)).flatten()
    z = (r*np.cos(phi_f)).flatten()

    coords_f = np.array([x,y,z])
    
    dphi = (phi_v[1:,1:]-phi_v[:-1,:-1])
    dtheta = (theta_v[1:,1:]-theta_v[:-1,:-1])
    area = np.abs((r*3390*1e5)**2*(np.sin(phi_f)*dtheta*dphi).flatten())

    rhat = coords_f/np.sqrt(np.sum(coords_f**2,axis=0))
    
    return ((lon_f, lat_f), coords_f, rhat, area)


def create_plot(field, xy, fdat,r, fname='Output/test.pdf'):
    
    if field in diverging_fields or 'flux' in field or 'normal' in field:
        cmap = 'RdBu'
        vmax = np.max(np.abs(fdat))
        vmin = -1*vmax
        print vmin, vmax
    else:
        cmap = 'viridis'
        vmin, vmax = np.min(fdat), np.max(fdat)
        
    if (field in log_fields or 'number_density'  in field) \
            and np.min(fdat)>0: norm = LogNorm(vmin=vmin, vmax=vmax)
    elif (field in log_fields or 'number_density'  in field):
        norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0)
    else: norm = Normalize(vmin=vmin, vmax=vmax)
        
    lon, lat = xy
    plt.pcolormesh(lon, lat, fdat.reshape(lon.shape), cmap=cmap,
                   norm=norm)
    plt.colorbar(label=label_lookup[field])
    plt.ylim(-90,90)
    plt.xlim(0,360)
    plt.xlabel('Longitude (0=Dayside, 180=Nightside)')
    plt.ylabel('Latitude')
    plt.title('R = {0} (RM)'.format(r))
    print 'Saving: {0}'.format(fname)
    plt.savefig(fname)
    plt.close()


def run_sphere_flux(ds_names, ds_types, r, fields, velocity_field=None):
    print velocity_field.format('fgj')
    xy, coords, rhat, area = create_sphere_mesh(r)
    indxs = get_path_idxs(coords, ds_names, ds_types)
    
    field_dat = {}
    for ds_type, keys in ds_types.items():
        for key in keys:
            field_dat[key] = {}
            dsf = ds_names[key]
            
            with h5py.File(dsf, 'r') as ds:
                for field in fields: 
                    ion_0, ion_1 = field.split('_')[:2]
                    ion = ion_0 + '_'+ion_1

                    if field == 'total_flux':
                        fluxes = np.array([v for k,v in field_dat[key].items() if 'flux' in k])
                        field_dat[key][field] = np.sum(fluxes, axis=0)

                    if field not in field_dat[key].keys():
                        field_dat[key][field] = get_ds_data(ds, field, indxs[ds_type], 
                                                            grid=ds_type=='heliosares', 
                                                            normal=rhat, velocity_field=velocity_field.format(ion))
                    
                    create_plot(field, xy, field_dat[key][field], r,
                                fname='Output/sphere_r{0}_{1}_{2}.pdf'.format(r,field,key))
    field_dat['area'] = area
    return field_dat

def main():
    
    radii=np.r_[[1.7]]#, np.arange(1.0, 3.0, 0.2)]
    ions = ['O2_p1', 'CO2_p1', 'O_p1']
    ds_type = 'batsrus_multi_species'
    
    if ds_type == 'batsrus_multi_fluid':
        ds_names={'batsrus_multi_fluid':
                    '/Volumes/triton/Data/ModelChallenge/R2349/batsrus_3d_multi_species.h5'}
        ds_types={'batsrus1': ['batsrus_multi_fluid']}
        #fields_suffix = ['number_density', 'velocity_total', 'velocity_normal', 'flux']
        fields_suffix = ['flux']
        fields = [ion+'_'+suff for ion, suff in iproduct(ions, fields_suffix)]
        velocity_field = '{0}_velocity'

    elif ds_type == 'batsrus_multi_species':
        ds_names={'batsrus_multi_species':
                '/Volumes/triton/Data/ModelChallenge/R2349/batsrus_3d_multi_fluid.h5'}
        ds_types={'batsrus1': ['batsrus_multi_species']}
    
        fields_suffix = ['number_density', 'flux']
        fields = [ion+'_'+suff for ion, suff in iproduct(ions, fields_suffix)]
        fields.append('velocity_total')
        fields.append('velocity_normal')
        velocity_field = 'velocity'

    elif ds_type == 'heliosares':
        ds_names={'heliosares':
                '/Volumes/triton/Data/ModelChallenge/R2349/helio_r2349.h5'}
        ds_types={'heliosares': ['heliosares']}
        ions = ['O_p1']
    
        fields_suffix = ['number_density', 'flux']
        fields = [ion+'_'+suff for ion, suff in iproduct(ions, fields_suffix)]
        velocity_field = '{0}_velocity'
    fields.append('total_flux')
    
    df = pd.DataFrame(columns=ions, index=radii)
    
    for r in radii:    
        field_dat = run_sphere_flux(ds_names, ds_types, r, fields, velocity_field=velocity_field)
        for ion in ions:
            total_ions = np.sum(field_dat['area']*field_dat[ds_type][ion+'_flux'])
            df.loc[r,ion] = total_ions
        
    df.to_csv('Output/sphere_flux_{0}.csv'.format(ds_type))
    
if __name__=='__main__':
    main()
                    
