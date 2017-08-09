from general_functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm


def calculate_bins(field_x, field_y, ds, Nbins=50):
    all_bins = {}
    for ax, f in zip(['x', 'y'], [field_x, field_y]):

        if f in field_lims: fmin, fmax = field_lims[f] 
        else: 
            fmin = np.nanmin(ds[f])
            fmax = np.nanmax(ds[f])


        if f in log_field_keys: log='log' 
        elif f in symlog_field_keys: log='symlog'
        else: log='lin'


        if log == 'log':
            bins_e = np.logspace(np.log10(fmin), np.log10(fmax), Nbins) 
            log_bins = np.log10(bins_e)
            bins_c = 10**(0.5*(log_bins[1:]+log_bins[:-1]))
            all_bins[ax+'_log'] = 'log'
        elif log == 'symlog':
            pass
        elif log == 'lin':
            bins_e = np.linspace(fmin, fmax, Nbins)
            bins_c = 0.5*(bins_e[1:]+bins[:-1])
            all_bins[ax+'_log'] = 'lin'



        all_bins[ax+'_edges'] = bins_e
        all_bins[ax+'_centers'] = bins_c
        all_bins[ax+'_log'] = log
    all_bins['xfield'] = field_x
    all_bins['yfield'] = field_y

    return all_bins



def phaseplot(xname, yname, bins, result):
    vmin, vmax = np.min(result[result>0]), np.max(result)
    print vmin, vmax
    norm = LogNorm(vmax=vmax, vmin=vmin)

    im = plt.pcolormesh(bins["x_centers"], bins["y_centers"],
                   result, cmap='viridis', norm=norm, rasterized=True) 
    plt.colorbar(im)

    if bins['x_log']=='log':plt.semilogx()
    if bins['y_log']=='log':plt.semilogy()


    plt.ylabel(label_lookup[yname])
    plt.xlabel(label_lookup[xname])

    plt.savefig('Output/test.pdf')

def bin_data(data, bins, weight_field):

    if weight_field is not None: weights = data[weight_field]
    else: weights = None


    H, _, _ = np.histogram2d(data[bins['xfield']],  
                             data[bins['yfield']],  
                             [bins['x_edges'], bins['y_edges']],
                             weights=weights)
    return H



def create_phaseplot(ds_name, field_x, field_y, weight_field=None,  region=None):

    # Get data
    fields = [field_x, field_y]
    if weight_field is not None: fields.append(weight_field)
    ds = load_data(ds_name, fields=fields+['x', 'y', 'z']) 

    reg = np.logical_and(np.logical_and(ds['z']>1, ds['x']<1.0), np.abs(ds['y'])<1)
    ds = {k:v[reg] for k,v in ds.items()}


    #Calculate Bins
    bins = calculate_bins(field_x, field_y, ds)

    # Bin data
    result = bin_data(ds, bins, weight_field)

    # Create phaseplot and save it
    phaseplot(field_x, field_y, bins, result)



def main():

    create_phaseplot('/Volumes/triton/Data/ModelChallenge/R2349/batsrus_3d_multi_fluid_lowres.h5',
                     'J_cross_B_total', 'v_cross_B_total')

if __name__=='__main__':
    main()
