pro swia_en_bin, orbit, outname

    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange

    mvn_swia_load_l2_data,/loadcoarse, trange=trange

    elo = [100, 1000, 4000 ]
    ehi = [1000, 4000, 20000 ]
    n_m = n_elements(elo)

    get_data , 'mvn_swics_en_eflux', time, temp, val
    n_t = n_elements(time)
    number_dens = fltarr( n_t, n_m)*!values.f_nan
    velocity = fltarr(n_t,3, n_m)*!values.f_nan
    tlimit, trange


    for h=0, n_m-1 do begin
        mvn_swia_part_moments,erange=[elo[h], ehi[h]]
        get_data , 'mvn_swics_density', time, temp, val


        print, n_t, n_elements(temp)
        number_dens[*,h] = temp

        ;spice_vector_rotate_tplot, 'mvn_swics_velocity', 'MAVEN_MSO'
        mvn_swia_inst2mso
        tplot_names
        get_data , 'mvn_swics_velocity_mso', time, temp, val
        velocity[*,*,h] = temp

    endfor

    all_vdat = [time, number_dens[*,0], number_dens[*,1], number_dens[*,2], velocity[*,0,0],  velocity[*,1,0], velocity[*,2,0], velocity[*,0,1],  velocity[*,1,1], velocity[*,2,1], velocity[*,0,2],  velocity[*,1,2], velocity[*,2,2]]
    print, n_elements(all_vdat), n_t
    all_vdat = transpose(reform(all_vdat, n_t, 13))
    
    write_csv, outname, all_vdat
    SAVE, /VARIABLES,/COMM,  FILENAME = 'temp.sav'
end
