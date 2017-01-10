pro monthly_drivers
mnths =  ['01', '02', '03', '04', '05', '06', '07','08','09','10','11','12', '01']
foreach yr, ['2014', '2015', '2016'] do begin
  for i=0,11 do begin
    tlimit, [yr+'-'+mnths[i]+'-01', yr+'-'+mnths[i+1]+'-01']
    makepng, '/Volumes/triton/Data/maven/orbit_plots/monthly_drivers/drivers_'+yr+'-'+mnths[i]
  endfor
endforeach

tlimit, ['2014-12-01','2015-01-01']
makepng, '/Volumes/triton/Data/maven/orbit_plots/monthly_drivers/drivers_2014-12'
tlimit, ['2015-12-01','2016-01-01']
makepng, '/Volumes/triton/Data/maven/orbit_plots/monthly_drivers/drivers_2015-12'
end