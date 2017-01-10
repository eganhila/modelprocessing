pro plot_maven_orbits
;t0 = '2014-12-17'
;trange = ['2014-12-17/10:00:00', '2014-12-17/14:32:00']
;t0 = '2014-12-23'
;trange = ['2014-12-23/08:00:00', '2014-12-23/12:32:00']
;t0 = '2016-01-09'
;trange = ['2016-01-09/05:00:00', '2016-01-09/09:32:00']
;t0 = '2016-01-20'
;trange = ['2016-01-20/18:00:00', '2016-01-20/23:30:00']
t0 = '2015-12-14'
trange = ['2015-12-14/16:30:00', '2015-12-14/21:00:00']

filename = "~/Projects/ModelChallenge/Output/"+t0+".tplot"
tplot_restore, filenames=filename

window, 0, xsize=700, ysize=1600
tplot_options,'ymargin',[8,4] 
tplot_options,'xmargin',[30,20] 

tplot, ['alt', 'swe_a4', 'mvn_swis_en_counts', 'mvn_sta_c6_E', 'mvn_sta_c6_M', 'mvn_B_full']
tlimit, trange[0], trange[1]

end
