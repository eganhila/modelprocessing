import yt
from load_heliosares import load_mgitm
import matplotlib.pyplot as plt
import numpy as np

mars_r = 3388.25

#   def get_slice_coords(normal, center, dx):
#
#       dx = 100
#       x = np.arange(-5000, 5000, dx)
#       z = np.arange(-5000, 5000, dx)
#
#       x_mesh, z_mesh = np.meshgrid(x, z)
#       x_mesh = x_mesh.flatten()
#       z_mesh = z_mesh.flatten()
#
#       alt = np.sqrt(x_mesh**2+z_mesh**2)-mars_r
#       lat = -1*(np.arccos(z_mesh/np.sqrt(x_mesh**2+z_mesh**2))-np.pi)*180/np.pi
#       lon = np.zeros_like(lat)
#
#
#
#       return np.array([x_mesh, z_mesh]), np.array([alt, lat, lon]) 
#
#   def get_slice_data(ds, coords, field):
#
#       vals = np.zeros(coords.shape[1])
#       ad = ds.all_data()
#       print ad['latitude'], ad['longitude'], ad['altitude'] 
#       for i, coord in enumerate(coords.T):
#           val = ds.find_field_values_at_point(field, coord)
#           if sum(val.shape) != 0:
#               vals[i] = val
#       return vals
#
#   def plot_slc(coords, data):
#
#       plt.pcolormesh(coords[0].reshape((100,-1)), coords[1].reshape((100,-1)), data.reshape((100,-1)))
#       plt.savefig('Output/test_slice.pdf')


def slice_lon(ds, field, lon=0, return_slc=False):

   slc_1 = ds.slice('longitude',lon)
   frb_1 = np.array(slc_1.to_frb(500, 1024, center=[0,0,0])[field])

   slc_2 = ds.slice('longitude', 180+lon)
   frb_2 = np.array(slc_2.to_frb(500, 1024, center=[0,0,0])[field])


   frb = np.zeros_like(frb_1)
   frb[:, :frb.shape[1]/2] = frb_1[:, frb.shape[1]/2:][:,::-1]
   frb[:, frb.shape[1]/2:] = frb_2[:, frb.shape[1]/2:]

   if return_slc:
       return frb
   
   plt.imshow(frb, cmap='viridis', extent=[-500,500,-500,500])
   plt.savefig('Output/testim_{0:03d}.png'.format(lon))


def slice_lat0(ds, field, return_slc=False):
   slc = ds.slice('latitude', 0)
   frb = np.array(slc.to_frb(500, 1024, center=[0,0,0])[field])

   if return_slc:
       return frb

   plt.imshow(frb, extent=[-500,500,-500,500], cmap='viridis')
   plt.savefig('Output/test.pdf')




def main():

    fdir = '/Volumes/triton/Data/ModelChallenge/MGITM/'
    fname = 'MGITM_LS180_F070_150615.nc'

    ds = load_mgitm(fdir=fdir, fname=fname)

    for lon in range(0,180,5):
        slice_lon(ds, 'oplus', lon)


if __name__ == "__main__":
    main()
