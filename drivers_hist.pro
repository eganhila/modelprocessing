pro drivers_hist
drivers_1d = ['psw',  'bsw', 'npsw', 'nasw', 'vpsw', 'tpmagsw']
drivers_1d = []
foreach driver, drivers_1d do begin
   get_data, driver, t, data, vals
   hist = histogram(data, locations=xbins, nbins=50)
   p = plot(xbins, hist, /histogram, xtitle=driver)
   p.save, "/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/drivershist_"+driver+".png"
   p.close
endforeach

drivers_3d = ['bsw']

foreach driver, drivers_3d do begin
  get_data, driver, t, data, vals
 
  hist0 = histogram(data[*,0], locations=xbins0, nbins=50)
  p = plot(xbins0, hist0, /histogram, xtitle=driver,color='blue', name='x')
  hist1 = histogram(data[*,1], locations=xbins1, nbins=50)
  p = plot(xbins1, hist1, /histogram, xtitle=driver, /overplot, color='green', name='y')
  hist2 = histogram(data[*,2], locations=xbins2, nbins=50)
  p = plot(xbins2, hist2, /histogram, xtitle=driver, /overplot, color='red', name='z')
  leg=legend(shadow=0)
  
  
  p.save, "/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/drivershist_"+driver+".png"
  ;p.close
endforeach


end