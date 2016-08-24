pro get_upstream

load_upstream

orbits = indgen(2926)
close_times = mvn_orbit_num(orbnum=orbits)


max_gap = 4.7d*1.25d*3600d
options, '*sw', 'datagap', /default, max_gap

tplot_names

bsw = data_cut('bsw', close_times)
psw = data_cut('psw', close_times)
npsw = data_cut('npsw', close_times)
nasw = data_cut('nasw', close_times)
vpsw = data_cut('vpsw', close_times)
vvecsw = data_cut('vvecsw', close_times)
tpmagsw = data_cut('tpmagsw', close_times)
in_sw = psw eq psw

all_dat = [orbits, in_sw, bsw[*,0], bsw[*,1],bsw[*,2], bsw[*,3], psw, npsw, nasw, vpsw, tpmagsw, vvecsw[*,0], vvecsw[*,1], vvecsw[*,2]]
all_dat = transpose(REFORM(all_dat, 2926, 14))
out_dir = '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'
write_csv, out_dir+'upstream.csv', all_dat


end