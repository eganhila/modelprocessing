import yt
from yt.units.yt_array import YTArray
import netCDF4 as nc
import numpy as np
aliases ={'o2pl':'O2_p1', 'opl':'O_p1'}
def load_heliosares(fdir, fname=None,  subset=None):
    if fname is None: fname = 'Hsw_18_06_14_t00600.nc'
    print fdir+fname

    nc_data = nc.Dataset(fdir+fname)

    # get data bounding box and dimension
    coord_vars = ('z', 'y', 'x')
    bbox = np.zeros((3,2))
    dims = np.zeros(3, dtype=int)

    for i, dim in enumerate(coord_vars):
        if dim == 'z':
            bbox[i, 0] = nc_data[dim][0]
            bbox[i, 1] = nc_data[dim][-1]
        else:
            bbox[i, 0] = nc_data[dim][-1]
            bbox[i, 1] = nc_data[dim][0]
        dims[i] = nc_data[dim].shape[0]

    nc_data.close()
    data = {}
    if subset is None:
        file_species = ['co2pl', 'hesw',  'hsw',   'o2pl', 'thew',
                     'elew', 'hpl', 'magw', 'opl']
    else:
        file_species = subset

    for s in file_species:
        print s
        nc_data = nc.Dataset(fdir+s+'_18_06_14_t00600.nc')

        for key, var in nc_data.variables.items():
            if np.all(var.shape == dims):
                data[(aliases[s],key)] = var[:, ::-1, ::-1]

        nc_data.close()

    ds = yt.load_uniform_grid(data, domain_dimensions=dims, bbox=bbox,
                    unit_system="mks", periodicity=(False, False, False),
                    geometry=('cartesian', ('z', 'y', 'x')))
    ds.fluid_types = ['gas', 'deposit', 'index', 'stream']+ ['O_p1', 'O2_p1']#file_species
    print ds.field_list

    return ds


def load_mgitm(fdir, fname=None,  subset=None):
    if fname is None: fname = 'hsw_18_06_14_t00600.nc'

    nc_data = nc.Dataset(fdir+fname)

    # get data bounding box and dimension
    coord_vars = ('altitude', 'Latitude', 'Longitude')
    bbox = np.zeros((3,2))
    dims = np.zeros(3, dtype=int)

    for i, dim in enumerate(coord_vars):
        bbox[i, 0] = nc_data[dim][0]
        bbox[i, 1] = nc_data[dim][-1]
        dims[i] = nc_data[dim].shape[0]

    bbox[1] = [-90, 90]
    bbox[2] = [0, 360]




    u'coordinate_system', u'ls', u'longsubsol', u'dec', u'mars_radius', u'altitude_from', u'Longitude', u'Latitude', u'altitude', u'o2plus', u'oplus', u'co2plus', u'n_e', u'co2', u'co', u'n2', u'o2', u'o', u'Zonal_vel', u'Merid_vel', u'Vert_vel', u'Temp_tn', u'Temp_ti', u'Temp_te'

    nc_dat = nc_data.variables

    for key, var in nc_data.variables.items():
        if np.all(var.shape == dims):
            data[key] = var[:, :, :]

    print nc_data.variables.keys()

    data['radius'] = data['altitude'] + nc_data.variables['mars_radius'] 


    ds = yt.load_uniform_grid(data, domain_dimensions=dims, bbox=bbox,
                    unit_system="mks", periodicity=(False, False, True),
                    length_unit=(1, 'km'),  
                    geometry=('geographic', ('altitude', 'latitude', 'longitude')))

    for key, var in nc_data.variables.items():
        if np.sum(var.shape) == 0:
            ds.parameters[key] = var[:]


    

    ds.fluid_types = ['gas', 'deposit', 'index', 'stream']

    nc_data.close()
    return ds


def test_heliosares():
    fdir = '/Volumes/triton/Data/ModelChallenge/Heliosares/test/'
    ds = load_heliosares(fdir)
    print ds.field_list

def test_mgitm():
    fdir = '/Volumes/triton/Data/ModelChallenge/MGITM/'
    fname = 'MGITM_LS180_F070_150615.nc'
    ds = load_mgitm(fdir, fname=fname)
    print ds.field_list

    slc = yt.SlicePlot(ds, 'latitude', 'oplus', width=500)
    slc.set_zlim('oplus', 1E5, 1E9)
    slc.save('Output/')

    slc = yt.SlicePlot(ds, 'longitude', 'oplus', center=[-1*ds.parameters['mars_radius'],0,90], width=500)
    #slc.set_zlim('oplus', 1E5, 1E9)
    slc.save('Output/')

if __name__ == '__main__':
    import sys, getopt

    opts, args = getopt.getopt(sys.argv[1:], "t:", ["test="])
    test_type = None
    for opt, arg in opts:
        if opt in ['-t', '--test']:
            test_type = arg
    
    if test_type == 'heliosares':
        test_heliosares()
    elif test_type == 'mgitm':
        test_mgitm()
    elif test_type is None:
        pass
    else:
        print 'Test "{0}" is unrecognized'.format(test_type)

    



