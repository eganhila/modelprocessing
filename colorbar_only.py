'''
Make a linear colorbar as a separate figure.

vmin, vmax = colorbar limits
cmap = colormap to be used
orientation = 'vertical' or 'horizontal'
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean

vmin, vmax = -30,30
cmap = cmocean.cm.delta
orientation = 'vertical'


ax = plt.subplot(111)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation=orientation)
cb.set_label('$\mathrm{B\; (nT)}$')

f= plt.gcf()
if orientation == 'vertical': f.set_size_inches(1.25,6)
else: f.set_size_inches(6,1.25)

plt.tight_layout()
plt.savefig('Output/colorbar.pdf')
