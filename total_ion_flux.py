import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from general_functions import *
import matplotlib.lines as mlines
from misc.labels import *
plt.style.use(['seaborn-talk', 'poster'])

df = pd.read_csv('Output/sphere_flux_batsrus_3d_multi_fluid.csv')
df = df.sort_values('Unnamed: 0')
df_mfluid = df.set_index('Unnamed: 0')

df = pd.read_csv('Output/sphere_flux_batsrus_3d_multi_species.csv')
df = df.sort_values('Unnamed: 0')
df_mspecies = df.set_index('Unnamed: 0')

df = pd.read_csv('Output/sphere_flux_batsrus_3d_pe.csv')
df = df.sort_values('Unnamed: 0')
df_pe = df.set_index('Unnamed: 0')

df = pd.read_csv('Output/sphere_flux_heliosares_multi.csv')
df = df.sort_values('Unnamed: 0')
df_helio = df.set_index('Unnamed: 0')

df = pd.read_csv('Output/sphere_flux_rhybrid.csv')
df = df.sort_values('Unnamed: 0')
df_rhybrid = df.set_index('Unnamed: 0')

dfs = {'fluid':df_mfluid, 'species':df_mspecies, 'helio':df_helio, 'rhybrid':df_rhybrid, 'pe':df_pe}
colors = {'fluid':'DodgerBlue', 'species':'LightSkyBlue', 'helio':'MediumVioletRed', 'rhybrid':'Orchid', 'pe':'LightSeaGreen'}
'$\mathrm{Ion\;Flux}$'

f, axes = plt.subplots(2,1)
for ax, ion in zip(axes, [ 'O2_p1', 'O_p1']):
    for dsk in ['fluid', 'species', 'helio', 'rhybrid', 'pe']:
        ax.plot(dfs[dsk][ion].index[1:], dfs[dsk][ion].values[1:], 'o-',  color=colors[dsk],label=dsk)


    ax.set_yscale('log')
    ax.set_ylim(1e22, 1e25)

    ax.set_ylabel(label_lookup[ion+'_flux'])

axes[1].set_xlabel('$\mathrm{Radius\;(R_M)}$')
axes[0].set_xticks([])

plt.legend(loc='upper right')
plt.tight_layout()
f.set_size_inches(10,8)


plt.savefig('Output/total_ion_flux.pdf')


