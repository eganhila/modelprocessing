import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp
from general_functions import *

def setup_rayplot(fields, ds_names, title=None):
    f, axes = plt.subplots(len(fields),1)
    
    #f, axes = plt.subplots(len(fields), 1)
    colors = {'maven':'k', 
              'bats_min_LS270_SSL0':'CornflowerBlue',
              'bats_min_LS270_SSL180':'DodgerBlue',
              'bats_min_LS270_SSL270':'LightSkyBlue',
              'batsrus_3dmhd':'CornflowerBlue',
              'helio_1':'LimeGreen',
              'helio_2':'ForestGreen',
              '2349_1RM_225km':'Black',
              'T0_1RM_225km':'DarkBlue',
              'T1_1RM_225km':'DodgerBlue'}
    
    plot = {}
    plot['axes'] = {field:ax for field, ax in zip(fields, axes)}
    plot['kwargs'] = {ds:{ 'label':ds.replace('_', ' '), 'color':colors[ds], 'lw':1.5}
                        for ds in ds_names}
    plot['figure'] = f
    plot['ax_arr'] = axes
    plot['N_axes'] = len(fields)
    plot['title'] = title
    return plot


def finalize_rayplot(plot, fname=None, show=False, zeroline=False):
    for f, ax in plot['axes'].items():
        if f in label_lookup: label = label_lookup[f]
        else: label = f
        ax.set_ylabel(label)
        if zeroline:
            ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyle=':', alpha=0.4)
    for i in range(plot['N_axes']):
        ax = plot['ax_arr'][i]

        ax.set_xlim(50, 5e3)
        ax.set_yscale('log')
        ax.set_xscale('log')


        if i == plot['N_axes']-1:
            ax.set_xlabel(u'$\textrm{Altitude\; (km)}$')
        else:
            ax.set_xticks([])
        #ax.set_ylabel(ax.get_ylabel(), labelpad=30*(i%2))    
    
    
    plot['ax_arr'][-1].legend()#(bbox_to_anchor=(1.4, 1))
    plot['ax_arr'][-1].set_zorder(1)
    plot['ax_arr'][0].set_title(plot['title'])
    plot['figure'].set_size_inches(10,10)

    if show:
        plt.show()
    if fname is None:
        plt.savefig('Output/test.pdf')
    else:
        print 'Saving: {0}'.format(fname)
        plt.savefig(fname)

def get_rayresults(ds_names, ds_types, fields, ax_i):
    results = {}

    ax = ['x','y','z'][ax_i]
    off_ax_i = [[1,2],[0,2],[0,1]]
    off_ax = [['y','z'],['x','z'],['x','y']]

    for dsk, dsn in ds_names.items():

        ds = load_data(dsn, fields=fields)
        data = {}

        if 'batsrus' not in dsk: 
	    shape = ds['x'].shape
            idx1 = shape[off_ax_i[ax_i][0]]/2
            idx2 = shape[off_ax_i[ax_i][1]]/2
            
            if ax_i == 0:
                for field in fields: data[field] = ds[field][:,idx1, idx2]
                rval = ds['x'][:,idx1, idx2]
            elif ax_i == 1:
                for field in fields: data[field] = ds[field][idx1,:, idx2]
                rval = ds['y'][idx1, :, idx2]
            elif ax_i == 2:
                for field in fields: data[field] = ds[field][idx1, idx2, :]
                rval = ds['z'][idx1, idx2,:]

        else:
            idx1 = np.abs(ds[off_ax[ax_i][0]]) == np.min(np.abs(ds[off_ax[ax_i][0]]))
            idx2 = np.abs(ds[off_ax[ax_i][1]]) == np.min(np.abs(ds[off_ax[ax_i][1]]))
            idx = np.logical_and(idx1, idx2)
            
            rval = ds[ax][idx]
            for field in fields: data[field] = ds[field][idx]

        results[dsk] = (rval, data)

    return results


def plot_rayresults(rval, data, ax, normal, **kwargs):

    sort = rval.argsort()
    rval = rval[sort]
    data = data[sort]

    if normal == '+':
        x = (rval[rval>1]-1)*3390
        y = data[rval>1]
    else:
        x = (np.abs(rval[rval<-1])-1)*3390
        y = data[rval<-1]
        
    if np.sum(y<=0)==len(y): y=-1*y
    ax.plot(x, y, **kwargs)
    



def make_rayplot(fields, ax_i, normal, save=False):

    ds_names, ds_types = get_datasets(load_key='exo_comparisonB', maven=False)
    plot = setup_rayplot(fields, ds_names,  title=['x','y','z'][ax_i]+normal)

    results = get_rayresults(ds_names, ds_types, fields, ax_i)



    for dsk, result in results.items():
        for field in fields:
            rval, data = result 
            plot_rayresults(rval,data[field],  plot['axes'][field], 
                            normal, **plot['kwargs'][dsk])
        if save:
            import pandas as pd
            df = pd.DataFrame.from_dict(result[1])
            df['radius'] = result[0]
            df.set_index('radius')
            df.to_csv('Output/{0}/{1}.csv'.format(dsk, save))

    finalize_rayplot(plot, fname='Output/ray_{0}{1}.pdf'.format(normal, ['x','y','z'][ax_i]))



def main():
    fields =['H_p1_number_density_normalized',
          'H_p1_flux_normalized',
          'H_p1_velocity_x_normalized',
          'H_p1_velocity_total_normalized',
          'H_p1_kinetic_energy_density_normalized',
          'magnetic_field_total_normalized',
          'O_p1_number_density_normalized',
          'O2_p1_number_density_normalized',
          ]  

    ax_i = 0

    #for ax_i in range(3):
    #    for norm in ['-', '+']:
    make_rayplot(fields, ax_i, normal='+',
                 save='subsolar_ray')

if __name__ == '__main__':
    main()
