import numpy as np
import matplotlib.pyplot as plt
import h5py
from general_functions import *
from matplotlib.collections import LineCollection
import sys
import getopt

reg_ax_lims = {'plume':[[-2,2],[-1,3],[-1,3]]}

def setup_whiskerplot():
    plot = {}
    
    f, axes = plt.subplots(3,1)
    
    plot['figure'] = f
    plot['axes'] = axes[::-1]
    
    return plot


def plot_whisker_data(plot, data, coords, showz=False):

    norm = 2.0/(np.nanmax(data))
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
        
        if showz:
            lc = LineCollection(lines, cmap=plt.get_cmap('RdBu'),
                                norm=plt.Normalize(-1.0/norm, 1.0/norm))
            zdat = data[ax_i]
        else:
            lc = LineCollection(lines, cmap=plt.get_cmap('inferno'),
                                norm=plt.Normalize(0,1))
            zdat = np.linspace(0,1,vec_x.shape[0])
        
        lc.set_array(zdat)
        plot['axes'][ax_i].add_collection(lc)
        
        plot['axes'][ax_i].set_xlim(-4,4)
        plot['axes'][ax_i].set_ylim(-4,4)



def finalize_whiskerplot(plot, fname='Output/whisker.pdf', show=False, showp=None, region=None):
    plot['figure'].set_size_inches(5,10)
    ax_labels = [['Y','Z'],['X','Z'],['X','Y']]
    ax_title_lab = ["X", "Y", "Z"]
    off_ax = [[1,2],[0,2],[0,1]]
    
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

        if region in reg_ax_lims:
            xlim = reg_ax_lims[region][off_ax[ax_i][0]]
            ylim = reg_ax_lims[region][off_ax[ax_i][1]]
        else:
            xlim, ylim = (-4,4), (-4,4)
        plot['axes'][ax_i].set_xlim(xlim)
        plot['axes'][ax_i].set_ylim(ylim)
        off_ax = [[1,2],[0,2],[0,1]]

        if showp is not None:
            ax.scatter([showp[off_ax[ax_i][0]]], [showp[off_ax[ax_i][1]]],
                       marker='x', color='grey', zorder=20)

    plt.tight_layout()
    print 'Saving: {0}'.format(fname)
    if show:
        plt.show()
    else:
        plt.savefig(fname)


def get_whisker_data(infile, field, orbit, ds_type):
    fields = [field+c for c in ['_x', '_y', '_z']]

    if ds_type == 'maven': 
        coords = get_all_data({'ds':infile}, {ds_type:['ds']},
                              {ds_type:[]}, ['x','y','z'])
        coords = np.array([coords['x']['ds'], coords['y']['ds'], coords['z']['ds']])/3390
        time = get_all_data({'ds':infile}, {ds_type:['ds']},
                              {ds_type:[]}, ['time'])['time']['ds']
        time = (time-time[0])/(time[-1]-time[0])
        indx = []

    else:
        coords, time  = get_orbit_coords(int(orbit), Npts=25, return_time=True)
        indx = bin_coords(coords, infile, ('helio' in ds_type) or ('rhybrid' in ds_type) )

    dat = get_all_data({'ds':infile}, {ds_type:['ds']}, {ds_type:indx}, fields)

    return (coords, dat, time)


def make_plot(infile, field, orbit, ds_type, showz, showp, region=None):

    coords, dat, time = get_whisker_data(infile, field, orbit, ds_type)
    dat = np.array([dat[field+c]['ds'] for c in ['_x', '_y', '_z']])


    if region is not None: 
        if region == 'plume': tlim=(0.15, 0.45)

        t_i = next(x[0] for x in enumerate(time) if x[1] > tlim[0])
        t_f = next(x[0] for x in enumerate(time) if x[1] > tlim[1])
        coords = coords[:, t_i:t_f]
        dat = dat[:, t_i:t_f]


    plot = setup_whiskerplot()

    plot_whisker_data(plot, dat, coords, showz=showz)
    if ds_type == 'maven': dst = 'maven'
    else: dst = infile.split('/')[-1].split('.')[0]
    finalize_whiskerplot(plot, 
            fname='Output/whisker_{0}_{1}_{2}_{3}.pdf'.format(dst, field, orbit, showz),
            showp=showp, region=region)






def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:i:o:t:p:zr:",
                                    ["field=","infile=", "orbit=", 'type=', 'showz', 'showp=', "region="])
    except getopt.GetoptError:
        print getopt.GetoptError()
        print 'error'
        return
    
    infile, field, orbit, ds_type,showz,showp, region = None, None, None, None, False,None, None
    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            infile = arg
        elif opt in ("-f", "--field"):
            field = arg
        elif opt in ("-o", "--orbit"):
            orbit = int(arg)
        elif opt in ("-t", "--type"):
            ds_type = arg
        elif opt in ("-z", "--showz"):
            showz=True
        elif opt in ("-p", "--showp"):
            import ast
            showp = np.array(ast.literal_eval(arg))
        elif opt in ("-r", "--region"):
            region = arg

    if infile is None and ds_type is not None:
        ds_names, ds_types = get_datasets(R2349=True) 
        infile = ds_names[ds_type] 


    make_plot(infile, field, orbit, ds_type, showz, showp, region=region)


if __name__ == '__main__':
    main(sys.argv[1:])
