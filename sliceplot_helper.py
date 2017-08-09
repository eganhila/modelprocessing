"""
Helper functions for sliceplots
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge

def add_mars(ax_i, **kwargs):
    """
    Function to overplot mars over a given axis,
    assuming normally ordered axes
    """
    if ax_i == 2: add_mars_xy(**kwargs)
    if ax_i == 1: add_mars_xz(**kwargs)
    if ax_i == 0: add_mars_yz(**kwargs)
        
def add_mars_xy(ax=None,alpha=1):
    if ax is None: ax = plt.gca()
    center = (0,0)
    radius = 1
    theta1, theta2 = 90, 270
    
    for radius in [1, 0.66, 0.3]:
        w1 = Wedge(center, radius, theta1, theta2, fc='k', lw=1, ec='k',alpha=alpha,width=0.33)
        w2 = Wedge(center, radius, theta2, theta1, fc='white', ec='k',lw=1,alpha=alpha)
        for wedge in [w1, w2]:
            ax.add_artist(wedge)     

def add_mars_xz(ax=None,alpha=1):
    if ax is None: ax = plt.gca()
    center = (0,0)
    radius = 1
    theta1, theta2 = 90, 270
    
    w1 = Wedge(center, radius, theta1, theta2, fc='k', lw=1, ec='k',alpha=alpha)
    w2 = Wedge(center, radius, theta2, theta1, fc='white', ec='k', lw=1,alpha=alpha)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
        
    for theta in [-np.pi/5,np.pi/5, 0]:
        x,y = [-np.cos(theta),np.cos(theta)],[np.sin(theta), np.sin(theta)]
        ax.plot(x,y,color='k',lw=1,alpha=alpha)
        
def add_mars_yz(ax=None,alpha=1):
    if ax is None: ax = plt.gca()
    circle = plt.Circle((0, 0), 1, color='w', ec='k', lw=1,alpha=alpha)
    ax.add_artist(circle)
    
    for theta in [-np.pi/5,np.pi/5, 0]:
        x,y = [-np.cos(theta),np.cos(theta)],[np.sin(theta), np.sin(theta)]
        ax.plot(x,y,color='k',lw=1,alpha=alpha)


def orbit_intersect_plane(coords, center, ax_i):
    """
    Plot where an orbit intersects the slice plane
    """
    if center is None: center = [0,0,0]
    # test to see if there is any intersection
    
    ax_c = center[ax_i]
    pt_argmin  = np.argsort(np.abs(coords[ax_i, :-1]-ax_c))
    
    p1_idx = pt_argmin[0]
    if p1_idx == coords.shape[1]: p1_idx = p1_idx
    p1 = coords[:,p1_idx]
    
    p1_into_plane = ((coords[ax_i, p1_idx] - coords[ax_i, p1_idx+1])>0)
    
    i = 1
    while True:
        p2_idx = pt_argmin[i]
        p2_into_plane = ((coords[ax_i, p2_idx] - coords[ax_i, p2_idx+1])>0)
        if p2_into_plane == p1_into_plane: 
            i+=1
        else: break
            
    return (p1, coords[:,p2_idx])



def add_orbit(ax, ax_i, orbit, center=None, show_intersect=False,
              show_center=False, lw=5, tlimit=None):
    """
    Overplot an orbit on top of a sliceplot
    """
    off_ax = [[1,2],[0,2],[0,1]]
    coords, time = get_orbit_coords(orbit, Npts=250, return_time=True)

    if tlimit is not None:
        i0 = next(x[0] for x in enumerate(time) if x[1] > tlimit[0])
        i1 = next(x[0] for x in enumerate(time) if x[1] > tlimit[1])
        coords = coords[:, i0:i1]


    ltimes = np.linspace(0,1,coords.shape[1])
    #ltimes[np.logical_and(coords[ax_i]<0, np.sqrt(np.sum(coords[off_ax[ax_i]],axis=1))<1)] = 0
    
    x,y = coords[off_ax[ax_i][0]], coords[off_ax[ax_i][1]]
    
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('inferno'),
                        norm=plt.Normalize(ltimes.min(), ltimes.max()))
    lc.set_array(ltimes)
    lc.set_linewidth(lw)
    
    ax.add_collection(lc)
    
    if show_intersect:
        p1, p2 = orbit_intersect_plane(coords, center, ax_i)
        ax.scatter([p1[off_ax[ax_i][0]], p2[off_ax[ax_i][0]]],
                   [p1[off_ax[ax_i][1]], p2[off_ax[ax_i][1]]],
                   marker='x', color='grey', zorder=20)


def add_boundaries(ax, ax_i, center):

    lkwargs = {'color':'whitesmoke', 'lw':1}
    #Conics via Trotignon 2006

    #shock
    conic_args = {'x':0.600, 'ecc':1.026, 'L':2.081}
    add_conic(ax, ax_i, center, conic_args, lkwargs)

    #MPB x>0
    conic_args = {'x':0.64, 'ecc':0.770, 'L':1.080}
    add_conic(ax, ax_i, center, conic_args, lkwargs, lim_x='pos')

    #MPB x<0
    conic_args = {'x':1.600, 'ecc':1.009, 'L':0.528}
    add_conic(ax, ax_i, center, conic_args, lkwargs, lim_x='neg')



def add_conic(ax, ax_i, center, conic_args, lkwargs, lim_x=None):
    x0, ecc, L = conic_args['x'], conic_args['ecc'], conic_args['L']

    if ax_i == 0:
        x = center[0]
        r = np.sqrt((ecc**2 - 1)*(x - x0)**2 - 2*ecc*L*(x-x0)+L**2)

        phi = np.linspace(0,2*np.pi,100)
        y = r*np.sin(phi)
        z = r*np.cos(phi)
        ax.plot(y,z, **lkwargs)

    else:
        if lim_x == 'pos': x = np.linspace(0, 5, 250)
        elif lim_x == 'neg': x = np.linspace(-5,0,250)
        else: x = np.linspace(-5,5,500)
        
        r2 = (ecc**2 - 1)*(x - x0)**2 - 2*ecc*L*(x-x0)+L**2
        if ax_i == 2: z = center[2]
        else: z = center[1]

        y = np.sqrt(r2-z**2)

        ax.plot(x, y,  **lkwargs)
        ax.plot(x,-1*y, **lkwargs)






