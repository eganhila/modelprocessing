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


pro make_orbit, orbit, outname

    ; First set trange
    trange = orbit_time(orbit)

    ; Get the kp data
    mvn_kp_read, trange, insitu_0, /mag, /static, /ngims

    ; Load the static d0 data
    mvn_sta_l2_load, sta_apid=['d0']
    common mvn_d0,mvn_d0_ind,mvn_d0_dat

    ; Load spice kernels
    mvn_spice_load, trange=trange

    ; Load Elimits picked out for the plume
    qualitative_lims = load_qual_lims()

    ; Resample the kp data at that cadence, since I only
    ; know how to resample kp, and the d0 rate is fine
    mvn_kp_resample, insitu_0, mvn_d0_dat.time, insitu

    ; Setup ion data data structures, currently just O2+ and O+
    ; Any blank data will be NaN
    mlo = [9, 22]
    mhi = [22, 38]
    n_m = n_elements(mlo)
    num_d0 = n_elements(mvn_d0_dat.time)
    number_dens = fltarr(num_d0, n_m)*!values.f_nan
    velocity = fltarr(num_d0, 3, n_m)*!values.f_nan

    ; Loop through the timestamps and pick out either a plume moment
    ; or a kp data point or leave blank
    number_dens[*,0] =  insitu.ngims.Oplus_density
    number_dens[*,1] =  insitu.ngims.O2plus_density

    for i=0L, num_d0-1L do begin
        

        ; Check for plume moment
        if qualitative_lims[i,0] ne 0 then begin

            ; Loop through species
            for m_i=0, n_m-1 do begin
                dat = mvn_sta_get_d0(mvn_d0_dat.time[i])
                elim = qualitative_lims[i,*]
                mlim = [mlo[m_i], mhi[m_i]]
                number_dens[i,m_i] = n_4d(dat, energy=elim, mass=mlim)
                vtemp = v_4d(dat, energy=elim, mass=mlim)
                vtemp2 = spice_vector_rotate(vtemp, mvn_d0_dat.time[i],$
                                             'MAVEN_STATIC', "MAVEN_MSO")
                velocity[i,*, m_i]  = vtemp2
            endfor
        endif

    endfor

    time = mvn_d0_dat.time
    alt = insitu.spacecraft.altitude
    msox = insitu.spacecraft.MSO_X
    msoy = insitu.spacecraft.MSO_Y
    msoz = insitu.spacecraft.MSO_Z
    magx = insitu.mag.mso_x/3390
    magy = insitu.mag.mso_y/3390
    magz = insitu.mag.mso_z/3390

    dat = [time, alt, msox,msoy,msoz,number_dens[*,0],velocity[*,0,0], velocity[*,1,0], velocity[*,2,0],number_dens[*,1],velocity[*,0,1], velocity[*,1,1], velocity[*,2,1],magx,magy,magz]
    header = ['time','altitude',  'x','y','z', 'O_p1_number_density', 'O_p1_velocity_x', 'O_p1_velocity_y', 'O_p1_velocity_z', 'O2_p1_number_density', 'O2_p1_velocity_x', 'O2_p1_velocity_y', 'O2_p1_velocity_z', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']

    dat_t = transpose(reform(dat,size(msox, /N_elements), 16))
    write_csv, outname, dat_t, header=header



    


end
