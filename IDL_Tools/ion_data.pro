
pro swia_en_bin, orbit, outname

    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange

    mvn_swia_load_l2_data,/loadcoarse, trange=trange, /tplot
    mvn_spice_load, trange=trange

    elo = [100, 1000, 4000 ]
    ehi = [1000, 4000, 20000 ]
    n_m = n_elements(elo)

    tplot_names
    get_data , 'mvn_swics_en_counts', time, temp, val
    n_t = 0
    t_start_i = -1
    
    for i=0, n_elements(time)-1 do begin
        if t_start_i eq -1 then begin
            if time[i] ge t0_unix then begin 
                t_start_i = i
            endif
        endif else begin
            if time[i] lt t1_unix then begin
                n_t = n_t +1
            endif
        endelse
    endfor


    number_dens = fltarr( n_t, n_m)*!values.f_nan
    velocity = fltarr(n_t,3, n_m)*!values.f_nan
    tlimit, trange


    for h=0, n_m-1 do begin
        mvn_swia_part_moments,erange=[elo[h], ehi[h]]
        tplot_names
        get_data , 'mvn_swics_density', time, temp, val

        number_dens[*,h] = temp[t_start_i: t_start_i+n_t-1]

        get_data , 'mvn_swics_velocity', time, vtemp, val

        for i=0, n_t-1 do begin
            velocity[i,*,h] = spice_vector_rotate(transpose(vtemp[i+t_start_i,*]), time[i+t_start_i], 'MAVEN_SWIA', "MAVEN_MSO")
        endfor

    endfor

    all_vdat = [time[t_start_i:t_start_i+n_t-1], number_dens[*,0], number_dens[*,1], number_dens[*,2], velocity[*,0,0],  velocity[*,1,0], velocity[*,2,0], velocity[*,0,1],  velocity[*,1,1], velocity[*,2,1], velocity[*,0,2],  velocity[*,1,2], velocity[*,2,2]]
    all_vdat = transpose(reform(all_vdat, n_t, 13))
    
    write_csv, outname, all_vdat
    SAVE, /VARIABLES,/COMM,  FILENAME = 'temp.sav'
end

function load_qual_lims
    lim_struc = read_csv('/Volumes/triton/Data/ModelChallenge/Maven/plume_elim.csv')    
    qual_lims = fltarr(291, 2)
    qual_lims[*,0] = lim_struc.field3
    qual_lims[*,1] = lim_struc.field4
    return,  qual_lims
end

function orbit_time, orbit
    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange
    return,  trange 
end

pro sta_plume_moments,orbit, outname, hard_limit=hl

    trange = orbit_time(orbit)
    mvn_kp_read, trange, insitu_0, /mag

    mvn_sta_l2_load, sta_apid=['d0']
    common mvn_d0,mvn_d0_ind,mvn_d0_dat

    mvn_kp_resample, insitu_0, mvn_d0_dat.time, insitu
    msox = insitu.spacecraft.MSO_X/3390
    msoy = insitu.spacecraft.MSO_Y/3390
    msoz = insitu.spacecraft.MSO_Z/3390

    num_d0 = n_elements(mvn_d0_dat.time)
    number_dens = fltarr(num_d0, 2)*!values.f_nan
    velocity = fltarr(num_d0, 3, 2)*!values.f_nan

    mvn_spice_load, trange=trange

    qualitative_lims = load_qual_lims()


    ;;; Loop through timestamps
    for i = 0L, num_d0-1L do begin
            
        ;; Loop through mass bins
        dat = mvn_sta_get_d0(mvn_d0_dat.time[i])
        if keyword_set(hl) then elim =[2200,30000] $ 
        else elim = qualitative_lims[i, *]

        number_dens[i,0] = n_4d(dat, energy=elim, mass=[9,22])
        vtemp = v_4d(dat, energy=elim, mass=[9,22])
        vtemp2 = spice_vector_rotate(vtemp, mvn_d0_dat.time[i], 'MAVEN_STATIC', "MAVEN_MSO")
        velocity[i,*, 0]  = vtemp2

        number_dens[i,1] = n_4d(dat, energy=elim, mass=[22,38])
        vtemp = v_4d(dat, energy=elim, mass=[22,38])
        vtemp2 = spice_vector_rotate(vtemp, mvn_d0_dat.time[i], 'MAVEN_STATIC', "MAVEN_MSO")
        velocity[i,*, 1]  = vtemp2
    endfor


    all_dat = [mvn_d0_dat.time, msox, msoy,msoz,number_dens[*,0], velocity[*,0,0],  velocity[*,1,0], velocity[*,2,0], number_dens[*,1], velocity[*,0,1],  velocity[*,1,1], velocity[*,2,1]]

    all_dat = transpose(reform(all_dat, num_d0, 12))
    write_csv, outname, all_dat
    SAVE, /VARIABLES,/COMM,  FILENAME = 'temp.sav'
    
end 

pro sta_Edistr, orbit

    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    timespan, trange

    mvn_sta_l2_load, sta_apid=['d0']
    common mvn_d0,mvn_d0_ind,mvn_d0_dat
    num_d0 = n_elements(mvn_d0_dat.time)


    for i = 0L, num_d0-1L do begin

        dat = mvn_sta_get_d0(mvn_d0_dat.time[i])

        window, 2
        contour4d, dat,   /fill, /mass, /twt
    
        saveimage, '/Volumes/triton/Data/maven/energy_spectrum_'+string(i)+'.gif'
    endfor


end


pro sta_veldistr, orbit, hard_limit=hl

    trange = orbit_time(orbit)
    mvn_sta_l2_load, sta_apid=['d0', 'c6']
    common mvn_d0,mvn_d0_ind,mvn_d0_dat
    num_d0 = n_elements(mvn_d0_dat.time)
    mvn_spice_load, trange=trange

    qualitative_lims = load_qual_lims()

    for i = 0L, num_d0-1L do begin

        if keyword_set(hl) then elim =[2200,30000] $ 
        else elim = qualitative_lims[i, *]

        mvn_sta_slice2d_snap, mvn_d0_dat.time[i], angle=[-90,90], rotation='xz', /keepwin, /bline, /mso, mass=[9,22],m_int=16, /noolines,/vsc, xrange=[-500,500],erange=elim
        saveimage, '/Volumes/triton/Data/maven/velocity_distr_xz_O+_highE'+string(i)+'.gif'
        wdelete
        
        mvn_sta_slice2d_snap, mvn_d0_dat.time[i], angle=[-90,90],rotation='xy',/keepwin, /bline, /mso, mass=[9,22],m_int=16, /noolines, /vsc, xrange=[-500,500],erange=elim
        saveimage, '/Volumes/triton/Data/maven/velocity_distr_xy_O+_highE'+string(i)+'.gif'
        wdelete
        
        mvn_sta_slice2d_snap, mvn_d0_dat.time[i], angle=[-90,90],rotation='yz',/keepwin, /bline, /mso, mass=[9,22],m_int=16, /noolines, /vsc, xrange=[-500,500],erange=elim
        saveimage, '/Volumes/triton/Data/maven/velocity_distr_yz_O+_highE'+string(i)+'.gif'
        wdelete

        mvn_sta_slice2d_snap, mvn_d0_dat.time[i], angle=[-90,90],rotation='xz', /keepwin, /bline, /mso, mass=[22,38],m_int=32, /noolines, /vsc, xrange=[-500,500],erange=elim
        saveimage, '/Volumes/triton/Data/maven/velocity_distr_xz_O2+_highE'+string(i)+'.gif'
        wdelete
        
        mvn_sta_slice2d_snap, mvn_d0_dat.time[i], angle=[-90,90],rotation='xy',/keepwin, /bline, /mso, mass=[22,38],m_int=32, /noolines, /vsc, xrange=[-500,500],erange=elim
        saveimage, '/Volumes/triton/Data/maven/velocity_distr_xy_O2+_highE'+string(i)+'.gif'
        wdelete
        
        mvn_sta_slice2d_snap, mvn_d0_dat.time[i], angle=[-90,90],rotation='yz',/keepwin, /bline, /mso, mass=[22,38],m_int=32, /noolines, /vsc, xrange=[-500,500],erange=elim
        saveimage, '/Volumes/triton/Data/maven/velocity_distr_yz_O2+_highE'+string(i)+'.gif'
        wdelete
    endfor


end
