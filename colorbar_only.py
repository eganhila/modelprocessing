'''
Make a linear colorbar as a separate figure.

vmin, vmax = colorbar limits
cmap = colormap to be used
orientation = 'vertical' or 'horizontal'
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
from misc.field_default_params import *
from misc.labels import *
import cmocean
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
plt.style.use('seaborn-poster')


field = 'O_p1_number_density'
cmap = 'viridis'
orientation = 'horizontal'
log = 'log'

vmin, vmax = field_lims_slices[field] 
linthresh = None
 


if log == 'linear':
    norm = Normalize(vmin=vmin, vmax=vmax)
elif log == 'log':
    norm = LogNorm(vmax=vmax, vmin=vmin)
elif log == 'symlog':
    norm = SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh)
if field in label_lookup.keys(): label = label_lookup[field]
else: label = field

ax = plt.subplot(111)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation=orientation)
cb.set_label(label)

f= plt.gcf()
if orientation == 'vertical': f.set_size_inches(1.25,10)
else: f.set_size_inches(10,1.25)

plt.tight_layout()
plt.savefig('Output/colorbar.pdf')
