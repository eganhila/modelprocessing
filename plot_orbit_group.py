import numpy as np
import matplotlib.pyplot as plt
from general_functions import *
from sliceplot import *


def main():

    orbits = np.array([353, 360, 363, 364, 364, 365, 366, 367,
                       367, 368, 369, 370, 371, 375, 376, 376,
                       380, 381, 381, 382, 386, 386, 387, 390, 391])

    plot = setup_sliceplot()

    for ax_i, ax in enumerate(plot['axes']):
        for orbit in orbits:
            add_orbit(ax, ax_i, orbit,lw=0.8)

        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)

    finalize_sliceplot(plot)

if __name__=='__main__':
    main()
