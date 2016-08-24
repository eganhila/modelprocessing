pro create_all_trajs
for orbit= 633,2925 do begin
    t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
    t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
    trange = [time_string(t0_unix), time_string(t1_unix)]


    maven_orbit_tplot,timecrop=trange, /loadonly 
    maven_orbit_snap, times=trange, /keep


    orbit_str = string( format='(I04)', orbit)
    date_str = strmid(time_string(t0_unix), 0, 10)
    date_arr = strsplit(date_str, '-', /extract)

    makepng, '/Volumes/triton/Data/maven/orbit_plots/'+date_arr[0]+'/'+date_arr[1]+'/traj_'+date_arr[2]+'_'+orbit_str
endfor

end

