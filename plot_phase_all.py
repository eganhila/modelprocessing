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
from plot_phase_indv import *

src_off_ax = {0: [1, 2], 1:[0,2], 2:[0, 1]}
src_labels = {0: 'LS', 1: 'EUV', 2:"SSL"}
src_vals = {0:[270, 180, 0], 1:['Max', "Min"], 2:[0, 180, 270]}

def plot_axdat(ax, bindat, idxs, shape, dataxis, src_axis):

    N_fvals, sum_fvals, bins = bindat
    mean_fvals = np.sum(sum_fvals, axis=dataxis)/np.sum(N_fvals, axis=dataxis)
    mean_fvals = np.ma.masked_where(np.isnan(mean_fvals), mean_fvals)
    xax, yax = off_ax[dataxis]

    pcol = ax.pcolormesh(bins[xax], bins[yax],
                mean_fvals[:-1,:-1], cmap='viridis')
    pcol.set_edgecolor('face')
    
    ax.set_xlim([bins[xax].min(), bins[xax].max()])
    ax.set_ylim([bins[yax].min(), bins[yax].max()])
    ax.set_yscale('log')

    srcx, srcy = src_off_ax[src_axis]

    if idxs[0] != shape[0]-1: 
        ax.set_xticklabels([])
    if idxs[1] != 0:
        ax.set_yticklabels([])
    if idxs[1]==0:
        ax.set_ylabel(labels[yax])# {0}'.format(ls))
    if idxs[0]== shape[0]-1:
        ax.set_xlabel(labels[xax])# {0}'.format(ls))
    if idxs[0]==0:
        ax.set_title('{0}: {1}'.format(src_labels[srcy], src_vals[srcy][idxs[1]]))
    if idxs[1]==shape[1]-1:
        ax2 = ax.twinx()
        ax2.set_ylabel('{0}: {1}'.format(src_labels[srcx], 
            src_vals[srcx][idxs[0]]),
            rotation=270, labelpad=30)
        ax2.set_yticks([])


def plot_all(field, fnames, src_axis, src_index, plt_index):
    if src_axis == 0: fnames=fnames[src_index, :, :]
    if src_axis == 1: fnames=fnames[:, src_index, :]
    if src_axis == 2: fnames=fnames[:, :, src_index]

    f, plt_axes = plt.subplots(fnames.shape[0], fnames.shape[1])

    for i in range(fnames.shape[0]):
        for j in range(fnames.shape[1]):
            fname = fnames[i, j]
            pltax = plt_axes[i, j]

            fvals, geo_coords = load_data(fname, field)
            bindat = bin_data(fvals, geo_coords)
            plot_axdat(pltax, bindat, (i, j), fnames.shape, plt_index, src_axis)

    plt.savefig('Output/batrus_{0}_{1}_{2}.pdf'.format(src_labels[src_axis],
        src_vals[src_axis][src_index], field))


def main():
    model = 'batsrus'

    fnames = np.array(glob.glob(dirs[model]+'*.h5'))
    fnames = fnames.reshape(3, 2, 3)
    #rnames = np.array([n.split('/')[-1].split('.')[0] for n in fnames])
    #rnames = rnames.reshape(3, 2, 3)

    # Axis 0: LS
    # Axis 1: EUV Max/Min
    # Axis 2: Subsolar longitude

    fields = ['r', 'rhp', 'rop2', 'rop', 'rco2']

    for field in fields:
        for lon_i in range(3):
            plot_all(field, fnames, src_axis=2, src_index=lon_i, plt_index=1)
    


if __name__ == "__main__":
    main()
