pro euv_to_csv

restore, '/Users/hilaryegan/Library/IDL/euv_orbit_l2_v5r4.sav'

all_dat = [euv_ma[0,*], euv_ma[2, *]]
all_dat = transpose(REFORM(all_dat[2415, 2]))
out_dir = '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'
write_csv, out_dir+'euv.csv', all_dat
end