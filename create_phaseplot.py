import yt
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from matplotlib.colors import LogNorm
from yt.frontends.netcdf.api import NetCDFDataset
import netCDF4 as nc
import spiceypy as sp

off_ax = {0: [1, 2], 1: [1, 0], 2:[2, 0]}
logax = {0:True, 1:False, 2:False}
labels = {0: "Altitude (km)",
          1: "Latitude",
          2: "Longitude"}

def load_data():
    ds = NetCDFDataset('/Volumes/triton/Data/ModelChallenge/SDC_Archive/Heliosares/Hybrid/Run1/Hsw_18_06_14_t00600.nc', model='heliosares')
    ad = ds.all_data()
    ad['x']
    x, y, z = ad['x'].flatten(), ad['y'].flatten(), ad['z'].flatten()

    N = z.shape[0]

    lat, lon, alt = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        p_rec = [x[i], y[i], z[i]]
        p_lat = sp.spiceypy.reclat(p_rec)
        alt[i], lon[i], lat[i] = p_lat
            
    lat = lat*180/np.pi
    lon = lon*180/np.pi
    alt = alt - 3390

    geo_coords = np.array([alt, lat, lon])
    return (ds, geo_coords)


def bin_data(fvals, geo_coords):

    lat_bins = np.linspace(-90, 90, 50)
    lon_bins = np.linspace(-180, 180, 50)
    alt_bins = np.logspace(2, np.log10(np.max(geo_coords[0])), 50)
    bins = np.array(alt_bins, lat_bins, lon_bins)
    
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


def plot_data(field, binned_data, axis):

    N_fvals, sum_fvals, bins = binned_data
    mean_fvals = np.sum(sum_fvals, axis=axis)/np.sum(N_fvals, axis=axis)
    mean_fvals = np.ma.masked_where(np.isnan(mean_fvals), mean_fvals)

    xax = off_ax[axis][0]
    yax = off_ax[axis][1]

    pcol = plt.pcolormesh(bins[xax], bins[yax],
                mean_fvals[:-1,:-1], cmap='viridis')

    if logax[xax]: plt.semilogx()
    if logax[yax]: plt.semilogy()

    plt.xlabel(labels[xax])
    plt.ylabel(labels[yax])
    plt.xlim(bins[xax, 0], bins[xax, -1])
    plt.ylim(bins[yax, 0], bins[yax, -1])

    pcol.set_edgecolor('face')
    plt.colorbar(label=field)

    plt.savefig('Output/phase_helioR1_{0}_{1}.pdf'.format(field, axis))


def main():

    ds, geo_coords = load_data()

    ad = ds.all_data()
    for field in ds.derived_field_list:
        if "number_density" in field[1]:
            fvals = ad[field]

        bindat = bin_data(fvals, geo_coords)
        plot_data(field[1], bindat, 1) 
        plot_data(field[1], bindat, 2) 

        return


if __name__ == '__main__':
    main()
