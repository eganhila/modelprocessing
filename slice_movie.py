import numpy as np
import matplotlib.pyplot as plt
import h5py
from general_functions import import *
from sliceplot import import *

def load_data():
    pass

def magv_setup_plot():
    plot = {}
    f, axes = plt.subplots(1,3)
    plot['figure'] = f
    plot['axes'] = axes
    
    return plot

def magv_finalize_plot(plot, z,orbit=None, fname='Output/test.pdf'):
    plot['figure'].set_size_inches(10,5)
    ax_labels = ['X','Y']
    mars_frac = np.real(np.sqrt(1-z**2))
    for ax_i, ax in enumerate(plot['axes']):
        ax.set_aspect('equal')
        circle = plt.Circle((0, 0), 1*mars_frac, color='k')
        ax.add_artist(circle)
        if orbit is not None: add_orbit(ax,ax_i, orbit)
        ax.set_xlabel('$\mathrm{'+ax_labels[0]+'} \;(R_M)$')
        ax.set_ylabel('$\mathrm{'+ax_labels[1]+'} \;(R_M)$')
    
    plot['axes'][0].set_title('$B_X$')
    plot['axes'][1].set_title('$B_Y$')
    plot['axes'][2].set_title('$\mathrm{Z=' +'{0:0.02}'.format(z)+'}\;(R_M)$')
    plt.tight_layout()
    print 'Saving: {0}'.format(fname)
    plt.savefig(fname)
    plt.close()


def make_plots_onax(infile):

    for idx in range(0, ds['x'].shape[2],1):

        plot = magv_setup_plot()
        
        for ax_i, v in enumerate(['x', 'y']):
            field = 'magnetic_field_{0}'.format(v)
            
            slc = slice_data(ds, 2, field, regrid_data=False, vec_field=False, idx=idx)
            plot_data(plot, slc, ax_i, vec_field=False, field=field,
                      logscale=False, zlim=(-30,30), cbar=False)
            
        ax_i = 2
        field = 'magnetic_field'
            
        slc = slice_data(ds, 2, field, regrid_data=False, vec_field=True, idx=idx)
        plot_data(plot, slc, ax_i, vec_field=True, field=field)
        
        z =ds['z'][0,0,idx]
        magv_finalize_plot(plot, orbit=orbit, z=z,
                         fname='Output/zi_slices/{0}_zi-{1:03d}.pdf'.format(oprefix, idx))
    

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"i:",["infile="])
    except getopt.GetoptError:
        return
    
    infile, field, orbit, test = None, None, None, False
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
    
    
    if 'Heliosares' in infile: make_plots_onax(infile, oprefix)
    else: make_plot_offax(infile, oprefix)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])

