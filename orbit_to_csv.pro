pro orbit_to_csv
struc = READ_CSV("/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/orbits_aphelion_moderate.dat")
orbits = struc.field1[0]

;orbits = [2349]
foreach orbit, orbits do begin 
  t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
  t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
  trange = [time_string(t0_unix), time_string(t1_unix)]
  
  timespan, trange
  mvn_sta_l2_load,  sta_apid=['c6']
  mvn_sta_l2_tplot
  mvn_sta_l2_gen_kp
  ;mvn_swe_load_l2, /all
  ;mvn_swe_kp
  
  tplot_names
  
  get_data, 'mvn_sta_H+_raw_density', time, hp, val
  get_data, 'mvn_sta_He++_raw_density', time, hpp, val
  get_data, 'mvn_sta_O+_raw_density', time, op, val
  get_data, 'mvn_sta_O2+_raw_density', time, opp, val
  
  all_dat = [time, hp, op, opp] ; hpp is empty for some unknown reason
  all_dat = transpose(REFORM(all_dat, 4074, 4))
  out_dir = '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'
  write_csv, out_dir+'orbit_'+orbit+'_density.csv', all_dat
endforeach

end
