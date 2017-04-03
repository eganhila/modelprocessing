pro tplot_features, orbit

    load_tplot_data, orbit
    plot_tplot_data

end

pro load_tplot_data, orbit

    ; Define time
    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange

    ; Load spice kernels
    mvn_spice_load, trange=trange


    ; Get mag data and rotate it
    mvn_mag_load, trange=trange
    spice_vector_rotate_tplot, 'mvn_B_1sec', 'MAVEN_MSO', check = 'MAVEN_SC_BUS'

    ; Get swia data
    swia_en_bin, orbit, 'temp.csv'
    restore, 'temp.sav'
    store_data,'swia_number_density_en_bin',data={x:time[t_start_i:t_start_i+n_t-1], y:number_dens,ylog:1}
    store_data,'swia_velocity_x_en_bin',data={x:time, y:reform(velocity[*,0,*])}
    store_data,'swia_velocity_y_en_bin',data={x:time, y:reform(velocity[*,1,*])}
    store_data,'swia_velocity_z_en_bin',data={x:time, y:reform(velocity[*,2,*])}
    mvn_swia_load_l2_data,/loadall, /tplot

    ; Get static data
    mvn_sta_l2_load, trange=trange, sta_apid=['c6', 'd0']
    mvn_sta_l2_tplot

    sta_mass_bin,orbit, 'temp.csv', 'temp.csv' 
    restore, 'temp.sav'
    store_data,'static_number_density_mass_bin',data={x:mvn_d0_dat.time, y:number_dens,ylog:1}
    store_data,'static_velocity_x_mass_bin',data={x:mvn_d0_dat.time, y:reform(velocity[*,0,*])}
    store_data,'static_velocity_y_mass_bin',data={x:mvn_d0_dat.time, y:reform(velocity[*,1,*])}
    store_data,'static_velocity_z_mass_bin',data={x:mvn_d0_dat.time, y:reform(velocity[*,2,*])}
    

    sta_en_bin,orbit, 'temp.csv', 'temp.csv' 
    restore, 'temp.sav'
    store_data,'static_number_density_en_bin',data={x:mvn_d0_dat.time, y:number_dens,ylog:1}
    store_data,'static_velocity_x_en_bin',data={x:mvn_d0_dat.time, y:reform(velocity[*,0,*])}
    store_data,'static_velocity_y_en_bin',data={x:mvn_d0_dat.time, y:reform(velocity[*,1,*])}
    store_data,'static_velocity_z_en_bin',data={x:mvn_d0_dat.time, y:reform(velocity[*,2,*])}



    tplot_names

    save, /variables,/comm, filename='temp.sav'
end

pro plot_tplot_data


    ylim, 'mvn_B_1sec_MAVEN_MSO', -25,25
    fields = ['mvn_B_1sec_MAVEN_MSO', 'mvn_sta_c6_E', 'mvn_sta_d0_E', 'mvn_sta_c6_M', 'mvn_sta_d0_M', 'mvn_swics_en_counts',  'static_number_density_mass_bin', 'static_number_density_en_bin', 'swia_number_density_en_bin', 'static_velocity_x_en_bin', 'static_velocity_y_en_bin','static_velocity_z_en_bin','static_velocity_x_mass_bin','static_velocity_y_mass_bin','static_velocity_z_mass_bin'] 
    tplot, fields,  var_label=['alt','sza']


end


