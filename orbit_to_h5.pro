pro orbit_to_h5

struc = READ_CSV("/Volumes/triton/Data/maven/orbit_plots/final_orbits/orbitN.dat")
orbits = struc.field1[5:*]
 
fdir = '/Users/hilaryegan/Temp/';LS_270.h5' 


;foreach orbit, orbits do begin 
for orbit= 2409, 2500 do begin
    fid = H5F_CREATE(fdir+'orbit_'+string(format='(I04)', orbit)+'.h5')
    print,  '---------------------: ', orbit


    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    
    timespan, trange
    mvn_spice_load
    mvn_sta_l2_load,  sta_apid=['c6']
    mvn_sta_l2_tplot
    mvn_sta_l2_gen_kp
    timespan, trange
    mvn_mag_load
    spice_vector_rotate_tplot, 'mvn_B_1sec', 'MAVEN_MSO', check = 'MAVEN_SC_BUS'
    ;mvn_swe_load_l2, /all
    ;mvn_swe_kp
    tlimit, trange[0], trange[1]
    
    fields = ['mvn_sta_H+_raw_density','mvn_sta_O+_raw_density', 'mvn_sta_O2+_raw_density', 'mvn_B_1sec_MAVEN_MSO', 'mvn_sta_H+_vcx_MAVEN_MSO', 'mvn_sta_H+_vcy_MAVEN_MSO', 'mvn_sta_H+_vcz_MAVEN_MSO','mvn_sta_O2+_V-Vsc_MAVEN_MSO_vx','mvn_sta_O2+_V-Vsc_MAVEN_MSO_vy','mvn_sta_O2+_V-Vsc_MAVEN_MSO_vz','mvn_sta_H+_flux', 'mvn_sta_O2+_flux','mvn_sta_O+_flux' ]
    field_names = ['H_p1_number_density', 'O_p1_number_density', 'O2_p1_number_density', 'magnetic_field', 'H_p1_velocity_x','H_p1_velocity_y','H_p1_velocity_z','O2_p1_velocity_x' ,'O2_p1_velocity_y','O2_p1_velocity_z', 'H_p1_flux', 'O2_p1_flux', 'O_p1_flux']

    for i=0, 12 do begin
        print,  i
        get_data, fields[i], time, dat, val

        dt_id = H5T_IDL_CREATE(dat)
        dsp_id = H5S_CREATE_SIMPLE(size(dat,/DIMENSIONS))
        ds_id = H5D_CREATE(fid,field_names[i], dt_id, dsp_id)
        H5D_WRITE, ds_id, dat
        H5D_CLOSE, ds_id
        H5S_CLOSE, dsp_id
        H5T_CLOSE, dt_id
     

        
    endfor
    get_data, 'mvn_B_1sec_MAVEN_MSO', time, dat, val
    dt_id = H5T_IDL_CREATE(time)
    dsp_id = H5S_CREATE_SIMPLE(size(time,/DIMENSIONS))
    ds_id = H5D_CREATE(fid,'mag_time', dt_id, dsp_id)
    H5D_WRITE, ds_id, time
    H5D_CLOSE, ds_id
    H5S_CLOSE, dsp_id
    H5T_CLOSE, dt_id
    
    
    H5F_CLOSE, fid
;endforeach
endfor



end
