import numpy as np
import netCDF4 as nc
import h5py
import spiceypy as sp
import glob

name_conversion = {'CO2pl_Density':'CO2_p1_number_density',
                   'CO2pl_Temperature':'CO2_p1_temperature',
                   'CO2pl_Ux':'CO2_p1_velocity_x',               
                   'CO2pl_Uy':'CO2_p1_velocity_y',               
                   'CO2pl_Uz':'CO2_p1_velocity_z',                
                   'Elew_Ex':'electric_field_x',                
                   'Elew_Ey':'electric_field_y',                 
                   'Elew_Ez':'electric_field_z',                 
                   'Hesw_Density':'He_number_density',            
                   'Hesw_Temperature':'He_temperature',        
                   'Hesw_Ux':'He_velocity_x',                 
                   'Hesw_Uy':'He_velocity_y',                 
                   'Hesw_Uz':'He_velocity_z',                 
                   'Hpl_Density':'H_p1_number_density',             
                   'Hpl_Temperature':'H_p1_temperature',         
                   'Hpl_Ux':'H_p1_velocity_x',                  
                   'Hpl_Uy':'H_p1_velocity_y',                  
                   'Hpl_Uz':'H_p1_velocity_z',                  
                   'Hsw_Density':'H_number_density',             
                   'Hsw_Temperature':'H_temperature',         
                   'Hsw_Ux':'H_velocity_x',                  
                   'Hsw_Uy':'H_velocity_y',                  
                   'Hsw_Uz':'H_velocity_z',                  
                   'Magw_Bx':'magnetic_field_x',                 
                   'Magw_By':'magnetic_field_y',                 
                   'Magw_Bz':'magnetic_field_z',                 
                   'O2pl_Density':'O2_p1_number_density',            
                   'O2pl_Temperature':'O2_p1_temperature',        
                   'O2pl_Ux':'O2_p1_velocity_x',                 
                   'O2pl_Uy':'O2_p1_velocity_y',                 
                   'O2pl_Uz':'O2_p1_velocity_z',                 
                   'Opl_Density':'O_p1_number_density',             
                   'Opl_Temperature':'O_p1_temperature',         
                   'Opl_Ux':'O_p1_velocity_x',                  
                   'Opl_Uy':'O_p1_velocity_y',                  
                   'Opl_Uz':'O_p1_velocity_z',                  
                   'Thew_Density':'electron_number_density',            
                   'Thew_Temperature':'electron_temperature',        
                   'Thew_Ux':'electron_velocity_x',                 
                   'Thew_Uy':'electron_velocity_y',                 
                   'Thew_Uz':'electron_velocity_z',
                   'x':'x',
                   'y':'y',
                   'z':'z'}

fdir = '/Volumes/triton/Data/ModelChallenge/R2349/Heliosares/'
h5name = '../helio_r2349.h5' 
fnames = glob.glob(fdir+'*.nc')

#dims = (410, 418, 258)
dims = (386, 386, 202)

with nc.Dataset(fnames[0], mode='r') as ds:
    # x = ds.variables['x'][:]
    # y = ds.variables['y'][:]
    # z = ds.variables['z'][:]
    x = ds.variables['X_axis'][:]
    y = ds.variables['Y_axis'][:]
    z = ds.variables['Z_axis'][:]

xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
xmesh, ymesh, zmesh = xmesh, ymesh, zmesh
if fdir[-2] == '2': 
    print 'reversing'
    zmesh == zmesh*-1

lat = -1*(np.arctan2(np.sqrt(xmesh**2+ymesh**2), zmesh)*180/np.pi)+90  #theta
lon = np.arctan2(ymesh, xmesh)*180/np.pi   #phi
alt = np.sqrt(xmesh**2+ymesh**2+zmesh**2)-3390

with h5py.File(fdir+h5name, 'w') as f:
    f.create_dataset('latitude', data=lat)
    f.create_dataset('altitude', data=alt)
    f.create_dataset('longitude', data=lon)
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('z', data=z)
    f.create_dataset('xmesh', data=xmesh)
    f.create_dataset('ymesh', data=ymesh)
    f.create_dataset('zmesh', data=zmesh)

with h5py.File(fdir+h5name, 'r+') as f:
    for fname in fnames:
        print fname
        with nc.Dataset(fname, mode='r') as ds:

            species = fname.split('/')[-1].split('_')[0]

            for k,v in ds.variables.items():
                if np.all(v.shape == dims):
                    if species+'_'+k in name_conversion:
                        key = name_conversion[species+'_'+k]
                    elif k in name_conversion:
                        key = name_conversion[k]
                    else:
                        continue
                    if species == 'Magw' and fdir[-2] == '2' and k!='Bz':
                        print 'mag'
                        f.create_dataset(key, data=-1*v[:].T)
                    else:
                        f.create_dataset(key, data=v[:].T)




