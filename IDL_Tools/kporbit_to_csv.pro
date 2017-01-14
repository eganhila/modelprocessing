; This routine makes a csv with data for an orbit

orbit = 2349

t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
trange = [time_string(t0_unix), time_string(t1_unix)]

mvn_kp_read, trange, insitu_0, /static, /swia, /swea, /mag, /ngims,/lpw ;, /new_files

time = indgen(t1_unix-t0_unix-20, start=t0_unix+10, /l64)
mvn_kp_resample, insitu_0, time, insitu

msox = insitu.spacecraft.MSO_X
msoy = insitu.spacecraft.MSO_Y
msoz = insitu.spacecraft.MSO_Z
alt = insitu.spacecraft.altitude
time = insitu.time

o2_p1 = insitu.ngims.O2plus_density
o_p1 = insitu.ngims.Oplus_density
co2_p1 = insitu.ngims.CO2plus_density
h_p1 = insitu.swia.Hplus_density
e = insitu.lpw.electron_density
magx = insitu.mag.mso_x
magy = insitu.mag.mso_y
magz = insitu.mag.mso_z


dat = [time, msox, msoy,msoz, alt, o2_p1, o_p1, co2_p1, h_p1, e, magx, magy, magz]
dat_t = transpose(reform(dat,size(msox, /N_elements), 13))
header = ["time", "x", "y", "z", "altitude", "O2_p1_number_density", "O_p1_number_density", "CO2_p1_number_density","H_p1_number_density","electron_number_density","magnetic_field_x","magnetic_field_y", "magnetic_field_z"]

write_csv, 'test_orbit.csv', dat_t, header=header
