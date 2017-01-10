pro plot_orbit

; Set time
;trange = ['2014-12-17/10:00:00', '2014-12-17/14:32:00']
;trange = ['2014-12-23/08:00:00', '2014-12-23/12:32:00']
;trange = ['2016-01-09/05:00:00', '2016-01-09/09:32:00']
;trange = ['2016-01-20/15:00:00', '2016-01-22/19:32:00']
trange = ['2015-12-14/16:30:00', '2015-12-14/21:00:00']

maven_orbit_tplot, timecrop=trange
maven_orbit_snap, keep=1, times=trange

end
