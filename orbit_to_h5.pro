pro orbit_to_h5

struc = READ_CSV("/Volumes/triton/Data/maven/orbit_plots/final_orbits/orbitN.dat")
orbits = struc.field1[*]
print, orbits

fname = '/Volumes/triton/Data/ModelChallenge/Maven/LS_270.h5' 
fid = H5F_CREATE(fname)

foreach orbit, orbits do begin 
    print,  '---------------------: ', orbit


    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    
    timespan, trange
    mvn_sta_l2_load,  sta_apid=['c6']
    mvn_sta_l2_tplot
    mvn_sta_l2_gen_kp
    mvn_mag_load, trange=trange
    mvn_spice_load, trange=trange
    spice_vector_rotate_tplot, 'mvn_B_1sec', 'MAVEN_MSO', check = 'MAVEN_SC_BUS'
    ;mvn_swe_load_l2, /all
    ;mvn_swe_kp
    
    fields = ['mvn_sta_H+_raw_density','mvn_sta_O+_raw_density', 'mvn_sta_O2+_raw_density', 'mvn_B_1sec_MAVEN_MSO', 'mvn_sta_H+_vcx_MAVEN_MSO', 'mvn_sta_H+_vcy_MAVEN_MSO', 'mvn_sta_H+_vcz_MAVEN_MSO','mvn_sta_O2+_V-Vsc_MAVEN_MSO_vx','mvn_sta_O2+_V-Vsc_MAVEN_MSO_vy','mvn_sta_O2+_V-Vsc_MAVEN_MSO_vz','mvn_sta_H+_flux', 'mvn_sta_O2+_flux','mvn_sta_O+_flux' ]
    field_names = ['H_p1_number_density', 'O_p1_number_density', 'O2_p1_number_density', 'magnetic_field', 'H_p1_velocity_x','H_p1_velocity_y','H_p1_velocity_z','O2_p1_velocity_x' ,'O2_p1_velocity_y','O2_p1_velocity_z', 'H_p1_flux', 'O2_p1_flux', 'O_p1_flux']

    gid = H5G_CREATE(fid, string(orbit)) 
    for i=0, 12 do begin
        print,  i
        get_data, fields[i], time, dat, val

        dt_id = H5T_IDL_CREATE(dat)
        dsp_id = H5S_CREATE_SIMPLE(size(dat,/DIMENSIONS))
        ds_id = H5D_CREATE(gid,field_names[i], dt_id, dsp_id)
        H5D_WRITE, ds_id, dat

        H5D_CLOSE, ds_id
        H5S_CLOSE, dsp_id
        H5T_CLOSE, dt_id
    endfor
    H5G_CLOSE, gid

endforeach


H5F_CLOSE, fid

end
