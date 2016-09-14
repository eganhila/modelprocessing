pro upstream_to_csv

load_upstream

get_data , 'bsw', time, bsw, val
get_data , 'psw', time, psw, val
get_data , 'npsw', time, npsw, val
get_data , 'nasw', time, nasw, val
get_data , 'vpsw', time, vpsw, val
get_data , 'vvecsw', time, vvecsw, val
get_data , 'tpmagsw', time, tpmagsw, val

all_dat = [time, bsw[*,0], bsw[*,1],bsw[*,2], bsw[*,3], psw, npsw, nasw, vpsw, tpmagsw, vvecsw[*,0], vvecsw[*,1], vvecsw[*,2]]
all_dat = transpose(REFORM(all_dat, 1359, 13))
out_dir = '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'
write_csv, out_dir+'upstream.csv', all_dat

end
