pro orbitstime_to_csv


  orbits = indgen(2926)
  close_times = mvn_orbit_num(orbnum=orbits)
  all_dat = [orbits, close_times]
  all_dat = transpose(REFORM(all_dat, 2926, 2))
  out_dir = '/Users/hilaryegan/Projects/ModelChallenge/ModelProcessing/Output/'
  write_csv, out_dir+'orbit_times.csv', all_dat

end
