import yt 
import numpy as np
import matplotlib.pyplot as plt
from yt.frontends.netcdf.api import NetCDFDataset
import spiceypy as sp
from data_load import load_heliosares
plt.style.use('ggplot')

from test_ytpath import *


def nan_converter(x):
    if x == 'NaN':
        return np.nan
    else:
        return float(x)


def load_all():

    datasets = {}

    # Load MGITM
    fdir = '/Volumes/triton/Data/ModelChallenge/MGITM/'
    fname = 'mgitm_ls180_f070_150615.nc'
    datasets['MGITM'] = NetCDFDataset(fdir+fname, model='mgitm')


    # Load Heliosares
    fdir = '/Volumes/triton/Data/ModelChallenge/Heliosares/test/'
    fname = 'Hsw_18_06_14_t00600.nc'
    #datasets['heliosares'] = NetCDFDataset(fdir+fname, model='heliosares')


    # Load MAVEN
    fdir = '/users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'
    fname = 'orbit_2349_density.csv'
    #datasets['maven'] = np.loadtxt(fdir+fname,delimiter=',', unpack=True,
    #                 converters={i:nan_converter for i in range(3)})

    # Load GCM
    fdir = '/Volumes/triton/Data/ModelChallenge/Heliosares/'
    fname = 'Heliosares_Ionos_Ls90_SolMean1_11_02_13.nc' 
    datasets['gcm'] = NetCDFDataset(filename=fdir+fname, model='gcm')


    # Load gcm 
    fdir = '/Volumes/triton/Data/ModelChallenge/Heliosares/'
    fname = 'Heliosares_Ionos_Ls90_SolMean1_11_02_13.nc'
    datasets['gcm'] = NetCDFDataset(fdir+fname, model='gcm')
    print datasets['gcm'].field_list

    return datasets

def create_paths(datasets, fields, quick_test=False):

    # Dictionary containing all our flythroughs
    fts = {}

    # Create highres/spherical and lowres/cartesian coords
    if quick_test: steps = (100, 50)
    else: steps = (1000, 100)
    hr_coords, hr_times = get_maven_path(geo=True, step=steps[0])
    lr_coords, lr_times = get_maven_path(geo=False, step=steps[1])

    for key, ds in datasets.items():

        # Case 1: data is already in flythrough, aka MAVEN data
        if key == 'maven':
            ft = (ds[0], {'H_p1_number_density':ds[1], 
                          'O_p1_number_density':ds[2], 
                          'O2_p1_number_density':ds[3]})
            fts[key] = ft

        # Case 2: Data is in not cartesian coordinates so need to
        #         sample by points instead of by rays (for some reason)
        elif ds.geometry != 'cartesian':
            t, dat, arr_coords, idx =  get_path_arrays(ds, 
                    hr_coords, hr_times, fields, use_pts=True)
            #t = np.linspace(1450110480.600119, 1450126772.468823, dat[fields[0]].shape[0])
            fts[key] = (t, dat)

        # Case 3: Dataset is in cartesian coordinates, sample normally
        else:
            t, dat, arr_coords, idx =  get_path_arrays(ds, 
                    lr_coords, lr_times, fields, use_pts=False)
            #dat  = {k:v/1E6 for k,v in dat.items()}
            #t = np.linspace(1450110480.600119, 1450126772.468823, dat[fields[0]].shape[0])
            fts[key] = (t, dat)

    return fts


def plot_all(fts, field, ax):

    for key, val in fts.items():
        t, dat = val
        ax.plot(t, dat[field], label=key)

    ax.set_yscale('log')
    ax.set_ylabel(field)
    #ax.set_xlim(7000+1.45011e9, 10000+1.45011e9)
    ax.set_ylim(10, 1e5)

def main():

    species = ['O_p1_number_density', 'O2_p1_number_density']
    [('netcdf', u'Merid_vel'), ('netcdf', u'Temp_elec'), ('netcdf', u'Temp_ion'), ('netcdf', u'Temperature'), ('netcdf', u'Vert_vel'), ('netcdf', u'Zonal_vel'), ('netcdf', u'co2'), ('netcdf', u'co2plus'), ('netcdf', u'elec'), ('netcdf', u'h'), ('netcdf', u'h2'), ('netcdf', u'o')

    datasets = load_all()
    flythroughs = create_paths(datasets, species, quick_test=False)


    f, axes = plt.subplots(2)

    for ax, field in zip(axes.flatten(), species):

        plot_all(flythroughs, field, ax)

    plt.legend()
    plt.savefig('Output/flythrough.pdf')

if __name__ == "__main__":
    main()
