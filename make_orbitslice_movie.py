import numpy as np
from sliceplot import *
from general_functions import *

def make_plots(infile, field, orbit, regrid_data, vec_field):
    dsname = infile.split('/')[-1].replace('.h5', '') 
    ds = load_data(infile, field=field, vec_field=vec_field)
    trange = get_orbit_times(np.array([orbit]))
    pts, times = get_path_pts(trange, Npts=300)


    for i in range(pts.shape[1]):

	center = pts[:,i]/3390
        plot = setup_sliceplot()
    
	for ax in [0,1,2]:
	    slc = slice_data(ds, ax, field, regrid_data, vec_field=vec_field,center=center)
	    plot_data(plot, slc, ax, vec_field, field, zlim=(1e-3, 1e5))
	finalize_sliceplot(plot, orbit=orbit, center=center, show_center=True, 
			   fname='Output/orbit_slices/slice_{0}_orb371_{1:03d}.png'.format(dsname, i))
    plt.close()






def main(argv):

    try:
        opts, args = getopt.getopt(argv,"f:i:o:",["field=","infile=", "orbit="])
    except getopt.GetoptError:
        return
    
    infile, field, orbit, center, test = None, None, None, None, False
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--orbit"):
            orbit = int(arg)
    
    if 'velocity' in field or 'magnetic_field' in field: vec_field = True
    else: vec_field = False
    if 'Heliosares' in infile: regrid_data = False
    else: regrid_data = True

    make_plots(infile, field, orbit, regrid_data, vec_field)


if __name__ == '__main__':
        main(sys.argv[1:])
