from modelprocessing.spherical_flux import create_sphere_mesh
from modelprocessing.general_functions import yt_load
import numpy as np
from yt.visualization.api import Streamlines

def single_topology(stream, bbox):
    stream = stream[np.all(stream != 0.0, axis=1)]
    
    top = -1
    
    radius = np.sqrt(np.sum(stream**2,axis=1))
    if radius.min()<0.9: top = 0  #through center -> closed
    elif radius[-1] < 1.0: top = 0 #through boundary -> closed
    elif stream[-1,0] >=  bbox[0,1]: top = 1 #out of bounds -> open
    elif stream[-1,1] >=  bbox[1,1]: top = 1
    elif stream[-1,2] >=  bbox[2,1]: top = 1
    elif stream[-1,0] <=  bbox[0,0]: top = 1 #out of bounds -> open
    elif stream[-1,1] <=  bbox[1,0]: top = 1
    elif stream[-1,2] <=  bbox[2,0]: top = 1

    return top

class TopologyMap():
    # topology: 0 = closed, 1 = open, 2 = draped, -1 = unprocessed
    # ends: 0 = radius < planet_r, 1 = radius >= box_size, -1 = unprocessed
    
    def __init__(self, fname):
        self.planet_r = 3390
        self.fname = fname
        self.positions_set = False

    def set_custom_positions(self, positions):
        self.init_coords = positions
        self.N_streamlines = positions.shape[1]
        self.positions_set = True
    
        
    def _create_basemap_positions(self, altitude, d_angle=5.0):
        r = (altitude+self.planet_r)/self.planet_r
        flonlat, coords_f, rhat, area = create_sphere_mesh(r, d_angle=d_angle)
        lon_f, lat_f = flonlat
        self.init_coords = coords_f
        self.map_shape = lat_f.shape
        self.map_x = lon_f
        self.map_y = lat_f
        self.map_area = area
        self.N_streamlines = coords_f.shape[-1]
        self.positions_set = True
    
    def _prepare_data_structures(self):
        self.topology = -1*np.ones(self.N_streamlines)
        self.in_ends = -1*np.ones(self.N_streamlines)
        self.out_ends = -1*np.ones(self.N_streamlines)
        
    def _integrate(self):
        #inbound
        streamlines = Streamlines(self.ds, self.init_coords.T, 'magnetic_field_x', 
                          'magnetic_field_y', 'magnetic_field_z',
                          length=self.ds.quan(10, 'code_length'), 
                          get_magnitude=True,direction=1)
        streamlines.integrate_through_volume()
        self.streamlines_in = streamlines
        
        #outbound
        streamlines = Streamlines(self.ds, self.init_coords.T, 'magnetic_field_x', 
                  'magnetic_field_y', 'magnetic_field_z',
                  length=self.ds.quan(10, 'code_length'), 
                  get_magnitude=True,direction=-1)
        streamlines.integrate_through_volume()
        self.streamlines_out = streamlines
        
    def _check_topology(self):
        bbox = np.array([self.ds.domain_left_edge.d,self.ds.domain_left_edge.d])
        for i, stream in enumerate(self.streamlines_in.streamlines):
            self.in_ends[i] = single_topology(stream, bbox)
    
        for i, stream in enumerate(self.streamlines_out.streamlines):
            self.out_ends[i] = single_topology(stream, bbox)
    
        s_closed = np.logical_and(self.in_ends==0, self.out_ends==0)
        s_draped = np.logical_and(self.in_ends==1, self.out_ends==1)
        s_open = np.logical_or(
                    np.logical_and(self.in_ends==0, self.out_ends==1),
                    np.logical_and(self.in_ends==1, self.out_ends==0))
        s_err = np.logical_or(self.in_ends==-1, self.out_ends==-1)

        self.topology[s_closed] = 0
        self.topology[s_open] = 1
        self.topology[s_draped] = 2
        self.topology[s_err] = -1
    
    def run(self, altitude=300, d_angle=10):
        self.ds = yt_load(self.fname,fields="all")
        if not self.positions_set:
            self._create_basemap_positions(altitude, d_angle=d_angle)
        self._prepare_data_structures()
        self._integrate()
        self._check_topology()
        
def plot_topmap(self, fname=None, show=False, line_lat=None):

    lon, lat = self.map_x, self.map_y
    plt.pcolormesh(lon, lat, self.topology.reshape(lon.shape), 
                   cmap=plt.cm.get_cmap('viridis', 3),
                   rasterized=True, vmin=0, vmax=2)

    tick_locs = [0,1,2]
    tick_labels = ["closed", "open", "draped"]
    cbar = plt.colorbar(label="Topology",ticks=tick_locs)
    cbar.ax.set_yticklabels(tick_labels)

    plt.gca().set_aspect('equal')
    
    if line_lat is not None:
        plt.plot([-90,270], [line_lat,line_lat],ls='--', 
                 color='white', alpha=0.4)
        plt.plot([-90,270], [-1*line_lat,-1*line_lat],ls='--', 
                 color='white', alpha=0.4)

    plt.ylim(-90,90)
    plt.xlim(-90,270)
    plt.xticks([-90,-30,30,90,150,210,270])
    plt.yticks([-90,-45,0,45,90])
    plt.xlabel('Longitude (0=Dayside, 180=Nightside)')
    plt.ylabel('Latitude')
    #plt.title('R = {0} (RM)'.format(r))

    if fname is None: fname = "Output/test.pdf"
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        
def plot_streamline(topmap, s_i):
    plot = setup_sliceplot()
    color = ['orange','blue','pink'][int(topmap.topology[s_i])]

    s_in = topmap.streamlines_in.streamlines[s_i]
    radius = np.sqrt(np.sum(s_in**2,axis=1))
    stop = np.argmax(radius<1)
    s_in = s_in[:stop]

    plot['axes'][0].plot(s_in[:,0], s_in[:,1], color=color)
    plot['axes'][1].plot(s_in[:,1], s_in[:,2], color=color)
    plot['axes'][2].plot(s_in[:,0], s_in[:,2], color=color)

    s_out = topmap.streamlines_out.streamlines[s_i]
    radius = np.sqrt(np.sum(s_out**2,axis=1))
    stop = np.argmax(radius<1)
    s_out = s_out[:stop]

    plot['axes'][0].plot(s_out[:,0], s_out[:,1], color=color)
    plot['axes'][1].plot(s_out[:,1], s_out[:,2], color=color)
    plot['axes'][2].plot(s_out[:,0], s_out[:,2], color=color)


    center = s_out[0].d
    field = "O2_p1_number_density"
    stream_field = "magnetic_field"


    for ax in [0,1,2]:
        slc = slice_data(ds, ax, field, regrid_data=False, center=center)
        plot_data(plot['axes'][ax], slc, ax, field)

        slc = slice_data(ds, ax, stream_field, regrid_data=False, vec_field=True, center=center)
        plot_data(plot['axes'][ax], slc, ax, stream_field, stream_field=True)

    finalize_sliceplot(plot,show=True, center=center)
