
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from general_functions import *
import matplotlib.lines as mlines
from misc.labels import *
import matplotlib as mpl
mpl.rcParams['text.latex.unicode'] = False
plt.style.use(['seaborn-talk', 'poster'])

df = pd.read_csv('Output/sphere_flux_batsrus_3d_multi_fluid_lowres.csv')
df = df.sort_values('Unnamed: 0')
df_mfluid = df.set_index('Unnamed: 0')
print df_mfluid

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
colors = {'fluid':'DodgerBlue', 'species':'Navy', 'helio':'MediumVioletRed', 'rhybrid':'Orchid', 'pe':'LightSeaGreen'}
labels = {'fluid':'BATSRUS-MF', 'pe':'BATSRUS-MF+Pe', 'rhybrid':'RHybrid', 'helio':"HELIOSARES", 'species':'BATSRUS-MS'}

keys = ['species', 'fluid', 'pe', 'helio', 'rhybrid'] 
O_p1 = [dfs[key].loc[1.8, 'O_p1_'] for key in keys] 
O2_p1 = [dfs[key].loc[1.8, 'O2_p1_'] for key in keys] 

N = len(keys)

fig, ax = plt.subplots()
rects = ax.bar(np.arange(N), O_p1,color=[colors[k] for k in keys])
ax.bar(np.arange(N)+N+1, O2_p1,color=[colors[k] for k in keys])
#plt.semilogy()

ax.set_xticks([2,8])
ax.set_xticklabels(('$O+$', '$O_2+$'))
plt.legend(rects, [labels[k] for k in keys], loc='upper left')
plt.yscale('log')
plt.ylim(1e23,1e25)
plt.ylabel('Ion Flux [\#/s]')

plt.savefig('Output/escape.pdf')
