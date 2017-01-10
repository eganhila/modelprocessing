pro save_sat, starttime=starttime, endtime=endtime

;write satellite file based on kp data, the time range is set by startime and endtime

  starttime ='2014-12-07/00:00:00'
  endtime ='2014-12-07/23:59:00'
  mvn_kp_read, [starttime,endtime] ,insitu, /insitu_only, /text_files

  dir0 ='/Users/yingjuan/simulations/MAVEN/'
  YY = strmid(starttime,0,4)
  MN = strmid(starttime,5,2)
  MN2 = strmid(endtime,5,2)

  instu='sat'
  DY = strmid(starttime,8,2)
  DY2 = strmid(endtime,8,2)
  outfile=dir0+instu+YY+'_'+MN+'_'+DY+'_'+MN2+'_'+DY2+'_kp.dat'
  print, 'save ', instu, ' data to ', outfile

  time=insitu.time
  Rm0=3396.0
  x = insitu.spacecraft.mso_x
  y = insitu.spacecraft.mso_y
  z = insitu.spacecraft.mso_z
  
  alt = insitu.spacecraft.altitude
  sza = insitu.spacecraft.sza
  alt1 = min(alt)
  print, 'alt, min max=', min(alt), max(alt)
  print, 'sza, min,max=',min(sza), max(sza)

  R = sqrt(x*x+y*y+z*z)
  alt2 = min(R)-Rm0
  print, 'min alt2=', alt2

  Rm = Rm0+ alt2-alt1   ; adjust the Mars radius according to periapsis alttiude
  print, 'Rm =' , Rm
  print, 'min alt3=', ((min(R)/Rm)-1.0)*Rm0   ;check if the minimum altutde using the adjusted Mars radius is similar to KP

  x = x/Rm
  y = y/Rm
  z = z/Rm
    
  openw,21,outfile
  printf,21,'#START'
  NT= N_ELEMENTS(time)
  for i=0, NT-2 do begin
    stime=time_string(time(i))
    MN = strmid(starttime,5,2)
    DY = strmid(starttime,8,2)
    hour = strmid(stime,11,2)
    min = strmid(stime,14,2)
    sec = strmid(stime,17,2)
    printf,21,FORMAT='(A4,A3,4(I3),A5,3F9.5)',YY, MN, DY, hour, min,sec, ' 000 ', x(i),y(i),z(i)
  endfor
  close,21
    
end 


