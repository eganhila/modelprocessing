'''
Make a colorbar as a separate figure.
'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean

ax = plt.subplot(111)
# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = cmocean.cm.delta
norm = mpl.colors.Normalize(vmin=-30, vmax=30)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb.set_label('$\mathrm{B\; (nT)}$')

f= plt.gcf()
#f.set_size_inches(6,1.25)
f.set_size_inches(1.25,6)

plt.tight_layout()

plt.savefig('Output/colorbar.pdf')
