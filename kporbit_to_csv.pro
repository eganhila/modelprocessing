orbit = 2349

t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
trange = [time_string(t0_unix), time_string(t1_unix)]

mvn_kp_read, trange, insitu, /static, /swia, /swea, /mag, /ngims,/lpw, /new_files

msox = insitu.spacecraft.MSO_X
msoy = insitu.spacecraft.MSO_Y
msoz = insitu.spacecraft.MSO_Z
alt = insitu.spacecraft.altitude

o2_p1 = insitu.ngims.O2plus_density
o_p1 = insitu.ngims.Oplus_density
co2_p1 = insitu.ngims.CO2plus_density
h_p1 = insitu.swia.Hplus_density
e = insitu.lpw.electron_density
magx = insitu.mag.mso_x
magy = insitu.mag.mso_y
magz = insitu.mag.mso_z


dat = [msox, msoy,msoz, alt, o2_p1, o_p1, co2_p1, h_p1, e, magx, magy, magz]
dat_t = transpose(reform(dat,size(msox, /N_elements), 12))

write_csv, 'Output/test_orbit.csv', dat_t
