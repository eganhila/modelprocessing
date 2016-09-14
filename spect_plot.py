import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec as GS
import numpy as np
import matplotlib.figure as Figure
from matplotlib.colors import LogNorm
import datetime

class SpectPlot(object):

    def __init__(self):

        self._update_vars = []
        self.active_vars = []
        self.data = {}
        self.ydat = {}
        self.t = {}
        self.ndims = {}
        self.axes = {}
        self.caxes = {}
        self.N_axes = 0 
        self.display_names = {}
        self.limit = None
        self.cmap = None
        self.log = {}

        self.N_add_axes = 0
        self.N_del_axes = 0

        self.figure = figure(1)

    def _update_plot(self):

        # Create or delete axes if necessary
        if self.N_add_axes - self.N_del_axes != 0:
            self.N_axes += self.N_add_axes
            self.N_axes -= self.N_del_axes
            self.N_add_axes = 0
            self.N_del_axes = 0

            gs = GS(self.N_axes, 2, width_ratios=[1, 0.05], wspace=0.01, hspace=0.05) 

            for i, var in enumerate(self.active_vars):
                self.axes[var] = plt.subplot(gs[i,0])
                if self.ndims[var] == 2:
                    self.caxes[var] = plt.subplot(gs[i,1])


            self._update_vars = self.active_vars

        # Replot necessary axes
        for var in self._update_vars:
            self._update_ax(var)

        self._update_vars = []

    def _update_ax(self, var):
        
        if self.ndims[var] == 1:
            self.plot1d(var)
        elif self.ndims[var] == 2:
            self.plot2d(var)

        ax = self.axes[var]
        ax.set_ylabel(self.display_names[var])

        if var != self.active_vars[-1]:
            ax.set_xticklabels([])

        if self.limit is not None:
            ax.set_xlim(self.limit)

    def plot1d(self, var):

        ax = self.axes[var]
        dat = self.data[var]
        t = self.t[var]

        ax.plot(t, dat, rasterized=True)
        if self.log[var]:
            ax.set_yscale('log')


    def plot2d(self, var):
        ax = self.axes[var]
        dat = self.data[var]
        y = self.ydat[var]
        t = self.t[var]

        im = ax.pcolormesh(t, y, dat.T, cmap=self.cmap, rasterized=True,
                 norm=LogNorm(vmin=dat[dat>0].min(), vmax=dat.max()))
        ax.axis([t.min(), t.max(), y.min(), y.max()])
        if self.log[var]:
            ax.set_yscale('log')


        self.figure.colorbar(im, cax=self.caxes[var], orientation='vertical')
        


    def add_data(self, name, t=None, y=None, z=None, display_name=None,
            log=True,activate_data=True):

        if z is not None:
            self.data[name] = z
            self.ydat[name] = y
            self.ndims[name] = 2
        else:
            self.data[name] = y
            self.ydat[name] = None
            self.ndims[name] = 1

        self.t[name] = t
        self.log[name] = log


        if activate_data and name not in self.active_vars:
            self.active_vars.append(name)
            self._update_vars.append(name)
            self.N_add_axes += 1

        if display_name is None:
            display_name = name
        self.display_names[name] = display_name



    def tlimit(self, lim0, lim1):
        self.limit = (np.datetime64(lim0),
                      np.datetime64(lim1))
        self._update_vars = self.active_vars

    def save(self, fname):
        self._update_plot()
        self.figure.autofmt_xdate()
        plt.savefig(fname)






