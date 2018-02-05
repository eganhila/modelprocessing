pro process_all_orbits

struc = READ_CSV("/Volumes/triton/Data/maven/orbit_plots/final_orbits/orbitN.dat")
orbits = struc.field1[5]
orbits = 2349 
foreach orbit, [orbits] do begin ;,  2344] do begin
;for orbit= 0,2925 do begin
  
    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]
    print, trange

    mvn_swia_load_l2_data, tplot=1, trange=trange, /loadspec, /eflux, /loadmom
    mvn_swe_load_l2, trange, /spec
    mvn_swe_sumplot, /loadonly
    mvn_sta_l2_load, trange=trange, sta_apid=['c6']
    mvn_sta_l2_tplot
    mvn_mag_load, trange=trange
    mvn_spice_load, trange=trange
    spice_vector_rotate_tplot, 'mvn_B_1sec', 'MAVEN_MSO', check = 'MAVEN_SC_BUS'
    mvn_load_eph_brain, trange


    crust_trange = [time_string(t0_unix+(t1_unix-t0_unix)/3.0),time_string(t0_unix+(t1_unix-t0_unix)*2/3.0)]
    mvn_model_bcrust_load, crust_trange, /tplot, nmax=60, /cain_2011, resolution='10sec'
    
    ylim, 'mvn_B_1sec_MAVEN_MSO', -15,15
    tplot, var_label=['alt', 'sza']
    
    WINDOW, 0, XSIZE=1200, YSIZE=1100
   
    tlimit, trange
    trange = trange
    ;tplot_options,'charsize',1.5
    tplot_options,'xmargin',[20,20]
    tplot_options,'ymargin',[10,3]
    ;options, 'mvn_B_1sec_MAVEN_MSO','labflag',-1
    options, 'mvn_B_1sec_MAVEN_MSO', 'colors', [0, 200, 400]
    options, 'mvn_mod_bcrust_mso_c11', 'colors', [0,200,400]
 
    ;

    tplot, ['mvn_sun', 'swe_a4', 'mvn_swis_en_eflux', 'mvn_sta_c6_E', 'mvn_sta_c6_M', 'mvn_B_1sec_MAVEN_MSO', 'mvn_mod_bcrust_mso_c11','mvn_swim_velocity_mso']


    orbit_str = string( format='(I04)', orbit)
    date_str = strmid(time_string(t0_unix), 0, 10)
    date_arr = strsplit(date_str, '-', /extract)

    makepng, '/Volumes/triton/Data/maven/orbit_plots/'+date_arr[0]+'/'+date_arr[1]+'/dat_'+date_arr[2]+'_'+orbit_str
endforeach
;endfor

end

