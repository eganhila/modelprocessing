pro LS_to_csv
orbits = indgen(2495)
orbit_times = time_string(mvn_orbit_num(orbnum=orbits))
mvn_eph_subsol_pos, orbit_times, /ls
get_data, 'mvn_eph_ls', time, dat, val
all_dat = [orbits, dat]
all_dat = transpose(REFORM(all_dat, 2495, 2))
out_dir = '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'
write_csv, out_dir+'solar_longitude.csv', all_dat
end