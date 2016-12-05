; First calculate the Bcrust along the orbit
;orbit = 2349
;t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
;t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
;t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
;
;trange = [time_string(t0_unix), time_string(t1_unix)]
;time = indgen(t1_unix-t0_unix-20, start=t0_unix+10, /l64)
;t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
;
;trange = [time_string(t0_unix), time_string(t1_unix)]
;time = indgen(t1_unix-t0_unix-20, start=t0_unix+10, /l64)
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
;   mvn_kp_read, trange, insitu_0
;   mvn_kp_resample, insitu_0, time, insitu
;
;   pos = [insitu.spacecraft.GEO_x, insitu.spacecraft.GEO_y, insitu.spacecraft.GEO_z] 
;   pos = reform(pos, size(time, /N_elements),3)
;   ;pos = transpose(pos)
;
;   mk = mvn_spice_kernels(/all, /load, trange=trange)
;   mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=data, pos=pos, /arkani, nmax=60
;
;   bss = transpose(spice_vector_rotate(data.pc, time, 'IAU_MARS', 'MAVEN_MSO'))
;
;   dat = [time-time[0], bss[*, 0], bss[*, 1], bss[*, 2]]
;
;   dat = transpose(reform(dat, size(time, /N_Elements), 4))
;   print, time[0]

;write_csv, 'Output/test_bcrust_0.csv', dat

strucr = read_csv("Output/coords_geo_real.csv")
GEO_real = [strucr.field1, strucr.field2, strucr.field3]
GEO_real = transpose(reform(GEO_real, size(strucr.field1, /N_elements), 3))

strucf = read_csv("Output/coords_geo_frozen.csv")
GEO_frozen = [strucf.field1, strucf.field2, strucf.field3]
GEO_frozen = transpose(reform(GEO_frozen, size(strucf.field1, /N_elements), 3))



orbit=2349
mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=B_frozen, pos=GEO_frozen, /arkani, nmax=60
mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=B_real, pos=GEO_real, /arkani, nmax=60

bf = transpose(B_frozen.pc)
br = transpose(B_real.pc)

dat = [ bf[*, 0], bf[*, 1], bf[*, 2], br[*,0], br[*,1], br[*,2]]

dat = transpose(reform(dat, size(bf[*,0], /N_Elements), 6))

write_csv, 'Output/test_bcrust_2.csv', dat
