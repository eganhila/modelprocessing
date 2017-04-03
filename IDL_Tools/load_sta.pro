orbit=2349
t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
trange = [time_string(t0_unix), time_string(t1_unix)]
timespan, trange

mvn_sta_l2_load, sta_apid=['c6']

