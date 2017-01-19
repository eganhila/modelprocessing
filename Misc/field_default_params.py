field_lims = {'magnetic_field_x':(-50,50),#(-33,33),
              'magnetic_field_y':(-50,50),#(-33,33),
              'magnetic_field_z':(-50,50),#(-33,33),
              'H_p1_number_density':(7e-1,7e4),
              'O2_p1_number_density':(1e0,6e4),
              'O_p1_number_density':(4e0, 2e4),
              'CO2_p1_number_density':(1e0,6e3),
              'H_number_density':(1e-2, 1e5), 
              'He_number_density':(1e-2, 1e5),
              'electron_number_density':(1e-2, 1e5),
              'magnetic_field_total':(0,120),
              'altitude':(5E1, 1e4)}

log_fields2 = ['H_p1_number_density',
               'O2_p1_number_density',
               'O_p1_number_density',
               'CO2_p1_number_density',
               'altitude',
               'number_density']
diverging_field_keys = ['flux', 'velocity_normal']
symlog_field_keys = ['flux', 'velocity_normal']
log_field_keys = ['number_density']
