import yt
from yt.frontends.netcdf.api import NetCDFDataset
import numpy as np

def load_mgitm():
    fdir = '/Volumes/triton/Data/ModelChallenge/MGITM/'
    fname = 'MGITM_LS180_F070_150615.nc'

    ds = yt.load(fdir+fname, model='mgitm')

    return ds

def load_heliosares():
    fdir = '/Volumes/triton/Data/ModelChallenge/Heliosares/test/'
    fname = 'Hsw_18_06_14_t00600.nc' 

    ds = yt.load(fdir+fname, model='heliosares')

    return ds

def load_gcm():
    fdir = '/Volumes/triton/Data/ModelChallenge/Heliosares/'
    fname = 'Heliosares_Ionos_Ls90_SolMean1_11_02_13.nc' 

    ds = yt.load(fdir+fname, model='gcm')


    return ds



def main(ds_type):

    if ds_type == 'mgitm': ds = load_mgitm()
    elif ds_type == 'heliosares': ds = load_heliosares()
    elif ds_type == 'gcm': ds = load_gcm()
    else:
        print 'Test type {0} unrecognized'.format(ds_type)
        return

    ad = ds.all_data()
    #print np.max(ad['Opl_Density'])
    #print ad['O_p1_number_density']
    print ad['oplus']
    slc = yt.SlicePlot(ds, 'latitude', 'oplus')
    #slc = yt.SlicePlot(ds, 'x', 'Opl_Density')
    slc.save('Output/')


#print ad['oplus']
#print ds.derived_field_list

#print ad['O_p1_number_density']
#print ad['latitude']
#print ad['latitude'][:]
#print ad['longitude']
#print ad['altitude']

#slc = yt.SlicePlot(ds, 2, 'ion_temperature')
#slc.save('Output/')

if __name__ == '__main__':
    import sys, getopt

    opts, args = getopt.getopt(sys.argv[1:], "t:", ["test="])
    test_type = None
    for opt, arg in opts:
        if opt in ['-t', '--test']:
            test_type = arg
    
    main(test_type)    
