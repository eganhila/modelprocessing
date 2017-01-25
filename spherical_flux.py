"""
Calculates properties over a spherical shell.

Inputs:
    --field, -f: field suffix to calculate shell for. Can 
        also give "all" to get shells calculated for flux, 
        number density, and normal velocity
    --species, -s: What species to calculate shell for. Can
        also give "all" to get shells calculated for O2_p1,
        O_p1, and CO2_p1.
    --infile, -i: Input file.
    --radius, -r: Radius in units of Rm to calculate shell for.
        Can also give all to get evenly spaced shells from 1-3 Rm.  
    -output_csv, -o: Flag to output a csv file with all the data
        that has been calculated.
    -total, -t: Flag to add "total_flux" as a field. Only totals
        ions that have already been calculated.
    -ion_velocity, -v: Flag to use ion velocities instead of
        total velocities

"""
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp
import h5py
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.collections import LineCollection
from general_functions import *
from flythrough_compare import *
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from itertools import product as iproduct
from misc.labels import *
from misc.field_default_params import *
import sys
plt.style.use(['seaborn-poster', 'poster'])



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


def create_plot(field, xy, fdat,r, show=False, fname='Output/test.pdf'):
    
    # Check to see if the field diverges
    if field in field_lims:
        vmin, vmax = field_lims[field]
        linthresh = int(np.ceil(np.log10(vmax)))-4

    if sum([1 for k in diverging_field_keys if k in field]):
        cmap = 'RdBu'
        fdat = np.ma.filled(np.ma.masked_invalid( fdat),0)
        if field not in field_lims:
            vmax = np.max(np.abs(fdat))
            vmin = -1*vmax
            linthresh= 100*np.min(np.abs(fdat[fdat>0]))
    else:
        cmap = 'viridis'
        fdat = np.ma.masked_where(fdat==0, fdat)
        fdat = np.ma.masked_invalid(fdat)
        if field not in field_lims:
            vmin, vmax = np.min(fdat), np.max(fdat)

        
    symlog=False
    if sum([1 for k in symlog_field_keys if k in field]):
        norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=1e5)
	maxlog=int(np.ceil( np.log10(vmax) ))
	minlog=int(np.ceil( np.log10(-vmin) ))
	linlog=int(np.ceil(np.log10(linthresh)))
        symlog=True

	#generate logarithmic ticks 
	tick_locations=([-(10**x) for x in xrange(minlog,linlog-1,-1)]
			+[0.0]
			+[(10**x) for x in xrange(linlog,maxlog+1)] )
    elif sum([1 for k in log_field_keys if k in field]):
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else: norm = Normalize(vmin=vmin, vmax=vmax)
        
    lon, lat = xy
    plt.pcolormesh(lon, lat, fdat.reshape(lon.shape), cmap=cmap,
                   norm=norm, rasterized=True, vmin=vmin, vmax=vmax)
    if symlog:
        plt.colorbar(label=label_lookup[field],ticks=tick_locations,
                format=ticker.LogFormatterMathtext())
    else:
        plt.colorbar(label=label_lookup[field])
        
    plt.ylim(-90,90)
    plt.xlim(0,360)
    plt.xlabel('Longitude (0=Dayside, 180=Nightside)')
    plt.ylabel('Latitude')
    plt.title('R = {0} (RM)'.format(r))
    print 'Saving: {0}'.format(fname)
    if show:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close()


def run_sphere_flux(ds_names, ds_types, r, fields, velocity_field=None, make_plot=True):
    xy, coords, rhat, area = create_sphere_mesh(r)
    indxs = get_path_idxs(coords, ds_names, ds_types)
    data = get_all_data(ds_names, ds_types, indxs, 
                        [f for f in fields if f != 'total_flux'],
                        velocity_field=velocity_field, normal=rhat)

    if 'total_flux' in fields:
        flux_dat = {}
        for dsk in ds_names.keys(): 
            fluxes = np.array([v[dsk] for k,v in data.items() if 'flux' in k])
            flux_dat[dsk] = np.sum(fluxes, axis=0)
        data['total_flux'] = flux_dat
    
    if make_plot:
        for dsk in ds_names.keys(): 
            for field in fields: 
                create_plot(field, xy, data[field][dsk], r,
                            fname='Output/sphere_r{0}_{1}_{2}.pdf'.format(r,field,dsk))

    data['area'] = area
    return data

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:i:r:s:otv",
                ["field=","species=", "infile=","radius=", "output_csv",
                 "total", "ion_velocity"] )
    except getopt.GetoptError:
        print getopt.GetoptError
        print 'error'
        return

    out_csv, total, velocity_field = False, False, 'velocity'
    for opt, arg in opts:
        if opt in ("-f", "--field"):
            if arg == "all": 
                fields_suffix = ['flux', 'number_density', 'velocity_normal']
            else:
                fields_suffix = [arg]
        elif opt in ("-s", "--species"):
            if arg == 'all': ions = ['O2_p1',  'O_p1']#'CO2_p1',
            else: ions = [arg]
        elif opt in ("-i", "--infile"):
            dsk = arg.split('/')[-1].split('.')[0]
            ds_names = {dsk:arg}
            ds_types = {'heliosares':[dsk]}
        elif opt in ("-r", "--radius"):
            if arg == 'all': radii = np.arange(1.0, 3.0, 0.2)
            else: radii = [arg]
        elif opt in ("-o", "-output_csv"):
            out_csv = True
        elif opt in ("-t", "-total"):
            total = True
        elif opt in ("-v", "-ion_velocity"):
            velocity_field = '{0}_velocity'
    print velocity_field

    fields = [ion+'_'+suff for ion, suff in iproduct(ions, fields_suffix)]
    if total: fields.append('total_flux')

    if out_csv: df = pd.DataFrame(columns=ions, index=radii)
    
    for ds_type in ds_names.keys():
        for r in radii:    
            field_dat = run_sphere_flux(ds_names, ds_types, r, fields,
                                        velocity_field=velocity_field)
            print field_dat
            if out_csv:
                for ion in ions:
                    total_ions = np.sum(np.nan_to_num(field_dat['area']*\
                                field_dat[ion+'_flux'][ds_type]))
                    df.loc[r,ion] = total_ions

        if out_csv: df.to_csv('Output/sphere_flux_{0}.csv'.format(ds_type))
    
if __name__=='__main__':
    main(sys.argv[1:])
                    
