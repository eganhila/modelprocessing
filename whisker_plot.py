import numpy as np
import matplotlib.pyplot as plt
import h5py
from general_functions import *
from matplotlib.collections import LineCollection
import sys
import getopt

def setup_whiskerplot():
    plot = {}
    
    f, axes = plt.subplots(3,1)
    
    plot['figure'] = f
    plot['axes'] = axes[::-1]
    
    return plot





def plot_whisker_data(plot, data, coords):

    norm = 1.0/(np.max(data))
    off_ax = [[1,2],[0,2],[0,1]]

    for ax_i in range(3):
        vec_x = data[off_ax[ax_i][0]]
        vec_y = data[off_ax[ax_i][1]]

        p0_x = coords[off_ax[ax_i][0]]
        p0_y = coords[off_ax[ax_i][1]]

        p1_x = p0_x+vec_x*norm
        p1_y = p0_y+vec_y*norm
        
        lines = []
        for pi in range(vec_x.shape[0]):
            lines.append([(p0_x[pi], p0_y[pi]),(p1_x[pi], p1_y[pi])])
            
        lc = LineCollection(lines, cmap=plt.get_cmap('inferno'), linewidth=1)

        lc.set_array(np.linspace(0,1,vec_x.shape[0]))
        plot['axes'][ax_i].add_collection(lc)
        







def finalize_whiskerplot(plot, fname='Output/whisker.pdf'):
    plot['figure'].set_size_inches(5,10)
    ax_labels = [['Y','Z'],['X','Z'],['X','Y']]
    ax_title_lab = ["X", "Y", "Z"]
    
    for ax_i, ax in enumerate(plot['axes']):
        ax.set_aspect('equal')
        mars_frac = 1
        ax.set_title('$\mathrm{'+ax_title_lab[ax_i]+'= 0}\;(R_M)$')

        circle = plt.Circle((0, 0), mars_frac, color='k')
        ax.add_artist(circle)
        circle = plt.Circle((0, 0), 1, color='k', alpha=0.1, zorder=0)
        ax.add_artist(circle)

        ax.set_xlabel('$\mathrm{'+ax_labels[ax_i][0]+'} \;(R_M)$')
        ax.set_ylabel('$\mathrm{'+ax_labels[ax_i][1]+'} \;(R_M)$')

        plot['axes'][ax_i].set_xlim(-4,4)
        plot['axes'][ax_i].set_ylim(-4,4)
        off_ax = [[1,2],[0,2],[0,1]]

    plt.tight_layout()
    print 'Saving: {0}'.format(fname)
    plt.savefig(fname)


def get_whisker_data(infile, field, orbit, ds_type):
    fields = [field+c for c in ['_x', '_y', '_z']]

    coords, indx = get_orbit_coords(int(orbit), Npts=150, return_idx=True)
    if ds_type != 'maven': 
        indx = bin_coords(coords, infile, ds_type=='heliosares')


    dat = get_all_data({'ds':infile}, {ds_type:['ds']}, {ds_type:indx}, fields)

    return (coords, dat)


def make_plot(infile, field, orbit, ds_type):

    coords, dat = get_whisker_data(infile, field, orbit, ds_type)
    dat = np.array([dat[field+c]['ds'] for c in ['_x', '_y', '_z']])
    plot = setup_whiskerplot()

    plot_whisker_data(plot, dat, coords)
    if ds_type == 'maven': dst = 'maven'
    else: dst = infile.split('/')[-1].split('.')[0]
    finalize_whiskerplot(plot, fname='Output/whisker_{0}_{1}_{2}.pdf'.format(dst, field, orbit))






def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:i:o:t:",["field=","infile=", "orbit=", 'type='])
    except getopt.GetoptError:
        print getopt.GetoptError()
        print 'error'
        return
    
    infile, field, orbit, ds_type = None, None, None, None
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--orbit"):
            orbit = int(arg)
        elif opt in ("-t", "--type"):
            maven = True

    make_plot(infile, field, orbit, ds_type)


if __name__ == '__main__':
    main(sys.argv[1:])
