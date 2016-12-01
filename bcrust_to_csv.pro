; First calculate the Bcrust along the orbit
orbit = 2349
mvn_model_bcrust_load, bcrust, orbit=orbit
get_data, 'mvn_mod_bcrust_mso', time, bcrust, val


t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
trange = [time_string(t0_unix), time_string(t1_unix)]
; Then try and calculate Bcrust at a given time along orbit

; First we have to get the positions of the trajectory
; Must be in IAU_MARS planetocentric coordinates


get_mvn_eph, time, eph 
pos = [eph.X_PC, eph.Y_PC, eph.Z_PC]
pos = reform(pos,size(eph.x_pc, /N_elements), 3)

mvn_model_bcrust, time[0], pos=pos, data=data

mk=mvn_spice_kernels(/all, /load, trange=trange)

iau_bcrust = data.pc
bss=spice_vector_rotate(transpose(iau_bcrust), trange, 'IAU_MARS',
                        'MAVEN_MSO')

mag = transpose(bss)

