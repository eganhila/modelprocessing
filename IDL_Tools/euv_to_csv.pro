pro euv_to_csv

restore, '/Users/hilaryegan/Library/IDL/euv_orbit_l2_v5r4.sav'
out_dir = '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/'


all_dat = [transpose(euvma_orbit[0,*]), transpose(euvma_orbit[2, *])]
all_dat = transpose(REFORM(all_dat, 2415, 2))
write_csv, out_dir+'euv_A.csv', all_dat

all_dat = [transpose(euvmb_orbit[0,*]), transpose(euvmb_orbit[2, *])]
all_dat = transpose(REFORM(all_dat, 2414, 2))
write_csv, out_dir+'euv_B.csv', all_dat

all_dat = [transpose(euvmc_orbit[0,*]), transpose(euvmc_orbit[2, *])]
all_dat = transpose(REFORM(all_dat, 2415, 2))
write_csv, out_dir+'euv_C.csv', all_dat

end