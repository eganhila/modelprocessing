import numpy as np
import matplotlib.pyplot as plt
import h5py
from general_functions import *
from sliceplot import *
from helio_pts import *
plt.style.use('seaborn-poster')

field_lims = {'magnetic_field':((-30,30),(-30,30)),
              'velocity':((-500,500), (-500,500)),
              'electron_velocity':((-500,500), (-500,500))}


def zi_finalize_plot(plot, z,orbit=None, fname='Output/test.pdf'):
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
    plt.savefig(fname, dpi=300)
    plt.close()


def make_plots_onax(infile, oprefix, vec_prefix='magnetic_field'):
    ds = load_data(infile, fields=['{0}_x'.format(vec_prefix),
                   '{0}_y'.format(vec_prefix)], vec_field=False)

    for idx in range(100, ds['x'].shape[2]-100,1):
        plot = {}
        f, axes = plt.subplots(1,3)
        plot['figure'] = f
        plot['axes'] = axes

        
        for ax_i, v in enumerate(['x', 'y']):
            field = '{0}_{1}'.format(vec_prefix, v)
            
            slc = slice_data(ds, 2, field, regrid_data=False, vec_field=False, idx=idx)
            plot_data(plot, slc, ax_i, vec_field=False, field=field,
		      logscale=False, zlim=field_lims[vec_prefix][ax_i], cbar=False, diverge_cmap=True)
            
        ax_i = 2
            
        slc = slice_data(ds, 2, vec_prefix, regrid_data=False, vec_field=True, idx=idx)
        plot_data(plot, slc, ax_i, vec_field=True, field=vec_prefix)
        
        z =ds['z'][0,0,idx]
        zi_finalize_plot(plot,
                         z=z,
                         fname='Output/zi_slices/{0}_zi-{1:03d}.png'.format(oprefix, idx-100))
    
def make_plots_offax(infile, oprefix, vec_prefix='magnetic_field'):
    ds = load_data(infile, fields=['{0}_x'.format(vec_prefix),
                   '{0}_y'.format(vec_prefix)], vec_field=False)
    
    for idx, z in enumerate(helio_idxpts[100:-100]):
        plot = {}
        f, axes = plt.subplots(1,3)
        plot['figure'] = f
        plot['axes'] = axes

        slc = slice_data(ds, 2, field='{0}_x'.format(vec_prefix), regrid_data=True, vec_field=False,
                         center=[0,0,z], extra_fields = ['{0}_y'.format(vec_prefix)])
        
        for ax_i, v in enumerate(['x', 'y']):
            field = '{0}_{1}'.format(vec_prefix, v)
            slc_temp = (slc[0], slc[1], slc[2][field])
            plot_data(plot, slc_temp, ax_i, vec_field=False, field=field,
		      logscale=False, zlim=field_lims[vec_prefix][ax_i], cbar=False, diverge_cmap=True)
            
        ax_i = 2
        slc_temp = (slc[0], slc[1], [slc[2]['{0}_x'.format(vec_prefix)],
                                     slc[2]['{0}_y'.format(vec_prefix)]])
        
        plot_data(plot, slc_temp, ax_i, vec_field=True, field=vec_prefix)
        
        #z =ds['z'][0,0,idx]
        zi_finalize_plot(plot,
                         z=z,
                         fname='Output/zi_slices/{0}_zi-{1:03d}.png'.format(oprefix, idx))

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"i:f",["infile=", "field="])
    except getopt.GetoptError:
        return
    
    infile, field, orbit, test = None, None, None, False
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        if opt in ("-f","--field"):
            field = arg
    
    oprefix = 'slice_magv_helio_1' 
    if 'Heliosares' in infile: make_plots_onax(infile, oprefix, vec_prefix=field)
    else: make_plots_offax(infile, oprefix, vec_prefix=field)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])

