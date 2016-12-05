; First calculate the Bcrust along the orbit
orbit = 2349
t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2

trange = [time_string(t0_unix), time_string(t1_unix)]
time = indgen(t1_unix-t0_unix-20, start=t0_unix+10, /l64)
;   mvn_model_bcrust, time, data=data, /arkani, nmax=60
;
;   ;mvn_model_bcrust, trange, data=bcrust
;   ;get_data, 'mvn_mod_bcrust_mso', time, bcrust, val
;   bcrust= transpose(data.ss)
;
;   dat = [time-time[0], bcrust[*, 0], bcrust[*, 1], bcrust[*, 2]]
;   dat = transpose(reform(dat, size(time, /N_Elements), 4))
;   print, time[0]
;
;   write_csv, 'Output/test_bcrust.csv', dat
mvn_kp_read, trange, insitu_0
mvn_kp_resample, insitu_0, time, insitu

pos = [insitu.spacecraft.GEO_x, insitu.spacecraft.GEO_y, insitu.spacecraft.GEO_z] 
pos = reform(pos, size(time, /N_elements),3)
;pos = transpose(pos)

mk = mvn_spice_kernels(/all, /load, trange=trange)
mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=data, pos=pos, /arkani, nmax=60

bss = transpose(spice_vector_rotate(data.pc, time, 'IAU_MARS', 'MAVEN_MSO'))

dat = [time-time[0], bss[*, 0], bss[*, 1], bss[*, 2]]

dat = transpose(reform(dat, size(time, /N_Elements), 4))
print, time[0]

write_csv, 'Output/test_bcrust_0.csv', dat


