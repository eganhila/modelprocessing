 pro get_shadow
 
 orbits = indgen(2926)
 close_times = mvn_orbit_num(orbnum=orbits)
 get_mvn_eph, close_times, eph
 in_shadow = fltarr(2926)
 
 for i=1, 2925 do begin

 x=eph[i].X_SS
 y=eph[i].Y_SS
 z=eph[i].Z_SS
 
 if (sqrt(y*y+z*z) lt 3390) then begin
  if (x lt 0) then in_shadow[i] = 1
 endif
 
 endfor

write_csv, '/Users/hilaryegan/Projects/ModelChallenge/MavenProcessing/Output/in_shadow.csv', orbits, in_shadow

 end