; Set time
;t0 = '2014-12-17'
;trange = ['2014-12-17/10:00:00', '2014-12-17/14:32:00']
;t0 = '2014-12-23'
;trange = ['2014-12-23/08:00:00', '2014-12-23/12:32:00']
;t0 = '2016-01-09'
;trange = ['2016-01-09/05:00:00', '2016-01-09/09:32:00']
;t0 = '2016-01-20'
;trange = ['2016-01-20/17:00:00', '2016-01-20/22:30:00']
t0 = '2015-12-14'
trange = ['2015-12-14/16:30:00', '2015-12-14/21:00:00']

; Load data and get it into tplot
;mvn_swe_load_l2, trange, spec=1, sumplot=1  ;sumplot to get vars into tplot
;mvn_swia_load_l2_data, tplot=1, loadspec=1, trange=trange 
;mvn_sta_l2_load, trange=trange
;mvn_sta_l2_tplot
mvn_mag_load, trange=trange;, tplot=1

;Rotate magnetic field
mvn_spice_load, trange=trange
spice_vector_rotate_tplot, 'mvn_B_1sec', 'MAVEN_MSO', check = 'MAVEN_SC_BUS'

; Make the plot of the variables we want using tplot
vars =  ['alt', 'swe_a4', 'mvn_swis_en_counts', 'mvn_sta_c6_E', 'mvn_sta_c6_M', 'mvn_B_full']
filename = "Output/"+t0
