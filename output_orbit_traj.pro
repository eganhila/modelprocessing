pro output_orbit_traj

    Rm0=3396.0
    out_dir = '/Volumes/triton/Data/ModelChallenge/Maven/Traj/'

    ;struc = READ_CSV("/Volumes/triton/Data/maven/orbit_plots/final_orbits/orbitN.dat")
    ;orbits = struc.field1;[0]
    orbits = make_array(1895, start=1105, /index) 

    ;orbits = [2349]
    foreach orbit, orbits do begin 

        t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
        t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
        trange = [time_string(t0_unix), time_string(t1_unix)]

        mvn_kp_read, trange ,insitu, /insitu_only, /text_files, /new_files

        time=insitu.time
        mso_x = insitu.spacecraft.mso_x
        mso_y = insitu.spacecraft.mso_y
        mso_z = insitu.spacecraft.mso_z
        alt = insitu.spacecraft.altitude
        sza = insitu.spacecraft.sza

        
        alt1 = min(alt)
        R = sqrt(mso_x*mso_x+mso_y*mso_y+mso_z*mso_z)
        alt2 = min(R)-Rm0
        Rm = Rm0+ alt2-alt1   ; adjust the Mars radius according to periapsis alttiude

        sim_x = mso_x/Rm
        sim_y = mso_y/Rm
        sim_z = mso_z/Rm

        all_dat = [sim_x, sim_y, sim_z, alt] 
        all_dat = transpose(REFORM(all_dat, size(sim_x, /N_ELEMENTS), 4))
        
        write_csv, out_dir+'orbit_'+string(orbit)+'_trajectory.csv', all_dat

    endforeach

        
end 
