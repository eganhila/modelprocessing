
pro sta_mass_bin,orbit, outname1, outname2 

    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange

    mvn_sta_l2_load, sta_apid=['d0']
    common mvn_d0,mvn_d0_ind,mvn_d0_dat

    mvn_sta_l2_load, sta_apid=['c6']
    common mvn_c6,mvn_c6_ind,mvn_c6_dat

    mlo = [ 9., 22., 38. ]
    mhi = [ 22., 38., 100. ]
    n_m = n_elements(mlo)
    
    num_c6 = n_elements(mvn_c6_dat.time)
    num_d0 = n_elements(mvn_d0_dat.time)
    ;number_dens = fltarr(num_c6, n_m)*!values.f_nan
    number_dens = fltarr(num_d0, n_m)*!values.f_nan
    velocity = fltarr(num_d0, 3, n_m)*!values.f_nan

    mvn_spice_load, trange=trange

   ;;;; Loop through timestamps
   ;for i = 0L, num_c6-1L do begin
   ;        
   ;    ;; Loop through mass bins
   ;    dat = mvn_sta_get_c6(mvn_c6_dat.time[i])
   ;    ;dat = mvn_sta_get_d0(data.time[i])

   ;    for h=0,n_m-1 do begin
   ;        number_dens[i,h] = n_4d(dat, mass=[mlo[h], mhi[h]])
   ;        ;vtemp = v_4d(dat, mass=[mlo[h], mhi[h]])
   ;        ;vtemp2 = spice_vector_rotate(vtemp, data.time[i], 'MAVEN_STATIC', "MAVEN_MSO", /verbose)
   ;        ;velocity[i,*,h]  = vtemp2
   ;    endfor
   ;endfor

    ;;; Loop through timestamps
    for i = 0L, num_d0-1L do begin
            
        ;; Loop through mass bins
        dat = mvn_sta_get_d0(mvn_d0_dat.time[i])
        ;dat = mvn_sta_get_d0(data.time[i])

        for h=0,n_m-1 do begin
            number_dens[i,h] = n_4d(dat, mass=[mlo[h], mhi[h]])
            vtemp = v_4d(dat, mass=[mlo[h], mhi[h]])
            vtemp2 = spice_vector_rotate(vtemp, mvn_d0_dat.time[i], 'MAVEN_STATIC', "MAVEN_MSO")
            velocity[i,*,h]  = vtemp2
        endfor
    endfor

    SAVE, /VARIABLES,/COMM,  FILENAME = 'temp.sav'
    
    ;all_ndat = [mvn_c6_dat.time, number_dens[*,0], number_dens[*,1], number_dens[*,2]]
    all_vdat = [mvn_d0_dat.time, number_dens[*,0], number_dens[*,1], number_dens[*,2], velocity[*,0,0],  velocity[*,1,0], velocity[*,2,0], velocity[*,0,1],  velocity[*,1,1], velocity[*,2,1], velocity[*,0,2],  velocity[*,1,2], velocity[*,2,2]]

    ;all_ndat = transpose(reform(all_ndat, num_c6, 4))
    all_vdat = transpose(reform(all_vdat, num_d0, 13))
  ;  write_csv, outname1, all_ndat
    write_csv, outname2, all_vdat

end 

pro sta_en_bin,orbit, outname1, outname2 

    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange

    mvn_sta_l2_load, sta_apid=['d0']
    common mvn_d0,mvn_d0_ind,mvn_d0_dat

    elo = [100, 1000, 4000 ]
    ehi = [1000, 4000, 20000 ]
    n_m = n_elements(elo)
    
    num_d0 = n_elements(mvn_d0_dat.time)
    number_dens = fltarr(num_d0, n_m)*!values.f_nan
    velocity = fltarr(num_d0, 3, n_m)*!values.f_nan

    mvn_spice_load, trange=trange
    print,  size(number_dens)

    ;;; Loop through timestamps
    for i = 0L, num_d0-1L do begin
            
        ;; Loop through mass bins
        dat = mvn_sta_get_d0(mvn_d0_dat.time[i])

        for h=0,n_m-1 do begin
            number_dens[i,h] = n_4d(dat, energy=[elo[h], ehi[h]], mass=[0,100])
            vtemp = v_4d(dat, energy=[elo[h], ehi[h]], mass=[0,100])
            vtemp2 = spice_vector_rotate(vtemp, mvn_d0_dat.time[i], 'MAVEN_STATIC', "MAVEN_MSO")
            velocity[i,*,h]  = vtemp2
        endfor
    endfor

    all_vdat = [mvn_d0_dat.time, number_dens[*,0], number_dens[*,1], number_dens[*,2], velocity[*,0,0],  velocity[*,1,0], velocity[*,2,0], velocity[*,0,1],  velocity[*,1,1], velocity[*,2,1], velocity[*,0,2],  velocity[*,1,2], velocity[*,2,2]]

    all_vdat = transpose(reform(all_vdat, num_d0, 13))
    write_csv, outname2, all_vdat



    
    
end 
