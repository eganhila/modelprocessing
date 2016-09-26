import yt
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from matplotlib.colors import LogNorm
from yt.frontends.netcdf.api import NetCDFDataset
import netCDF4 as nc
import spiceypy as sp
import h5py
import glob

dirs = {'heliosares':'/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/',
        'batsrus': '/Volumes/triton/Data/ModelChallenge/SDC_Archive/BATSRUS/'}
off_ax = {0: [1, 2], 1: [1, 0], 2:[2, 0]}
logax = {0:True, 1:False, 2:False}
labels = {0: "Altitude (km)",
          1: "Latitude",
          2: "Longitude",
          'O2pl_Density':'O$_2$+ Number Density (cm$^{-3}$)',
          'CO2pl_Density':'CO$_2$+ Number Density (cm$^{-3}$)',
          'Hesw_Density':'He Number Density (cm$^{-3}$)',
          'Hsw_Density':'H Number Density (cm$^{-3}$)',
          'Opl_Density':'O+ Number Density (cm$^{-3}$)', 
          'Thew_Density':'e- Number Density (cm$^{-3}$)',
          'r': 'Density (amu cm$^{-3}$)',
          'rhp':' H+ Density (amu cm$^{-3}$)',
          'rop2': 'O++ Density (amu cm$^{-3}$)',
          'rop': 'O$_2$+ Density (amu cm$^{-3}$)',
          'rco2': 'CO$_2$+ Density (amu cm$^{-3}$)'}



  

def load_data(fname, field):
    with h5py.File(fname, 'r') as f:

        fvals = f[field][:].flatten()
        if 'BATSRUS' in fname:
            alt = (f['altitude'][:].flatten()+3390-1)*3390
        else:
            alt = f['altitude'][:].flatten()
        lat = f['latitude'][:].flatten()
        lon = f['longitude'][:].flatten()
        geo_coords = np.array([alt, lat, lon]) 

    return (fvals, geo_coords)

    


def bin_data(fvals, geo_coords, test=False):

    lat_bins = np.linspace(-90, 90, 50)
    lon_bins = np.linspace(-180, 180, 50)
    alt_bins = np.logspace(2, np.log10(np.max(geo_coords[0])), 50)
    bins = np.array([alt_bins, lat_bins, lon_bins])
    
    if test:
        N_fvals = np.random.rand(51,51,51)
        sum_fvals = np.random.rand(51,51,51)
        return (N_fvals, sum_fvals, bins)

    alt_idx = np.digitize(geo_coords[0], alt_bins)
    lat_idx = np.digitize(geo_coords[1], lat_bins)
    lon_idx = np.digitize(geo_coords[2], lon_bins)

    N_fvals = np.zeros((51,51, 51))
    sum_fvals = np.zeros((51, 51, 51))

    for idx in range(len(fvals)):
	i, j, k = alt_idx[idx], lat_idx[idx], lon_idx[idx]
	N_fvals[i,j,k]+=1
	sum_fvals[i, j, k]+=fvals[idx]

    return (N_fvals, sum_fvals, bins)


def plot_data(field, binned_data, axis, fname):

    N_fvals, sum_fvals, bins = binned_data
    mean_fvals = np.sum(sum_fvals, axis=axis)/np.sum(N_fvals, axis=axis)
    mean_fvals = np.ma.masked_where(np.isnan(mean_fvals), mean_fvals)

    xax = off_ax[axis][0]
    yax = off_ax[axis][1]

    pcol = plt.pcolormesh(bins[xax], bins[yax],
                mean_fvals[:-1,:-1], cmap='viridis')
                #norm=LogNorm(mean_fvals[mean_fvals!=0].min(), mean_fvals.max()))

    if logax[xax]: plt.semilogx()
    if logax[yax]: plt.semilogy()

    plt.xlabel(labels[xax])
    plt.ylabel(labels[yax])
    plt.xlim(bins[xax, 0], bins[xax, -1])
    plt.ylim(bins[yax, 0], bins[yax, -1])

    pcol.set_edgecolor('face')
    plt.colorbar(label=labels[field])

    plt.savefig('Output/phase_{0}_{1}_{2}.pdf'.format(fname, field, axis))
    plt.close()


def main():

    fields = ['O2pl_Density', 'CO2pl_Density', 'Hesw_Density',
              'Hsw_Density', 'Opl_Density', 'Thew_Density']
    #fields = ['r', 'rhp', 'rop2', 'rop', 'rco2']

    model = 'heliosares'
    fnames = glob.glob(dirs[model]+'*.h5')
    for fname in fnames:
        print fname
        for field in fields:
            fvals, geo_coords = load_data(fname, field)


            bindat = bin_data(fvals, geo_coords)
            plot_data(field, bindat, 1, fname.split('/')[-1].split('.')[0]) 
            plot_data(field, bindat, 2, fname.split('/')[-1].split('.')[0]) 



if __name__ == '__main__':
    main()
