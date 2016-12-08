function legendre_schmidt_all, nmax, x
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; legendre_schmidt_all.pro                              ;
;                                                       ;
; Function returns the Schmidt-normalized associated    ;
; legendre polynomials:                                 ;
;  P(n,m,x) = Cnm * (1-x^2)^(m/2) * d^m/dx^m P(n,x)     ;
; Where:                                                ;
;  Cnm = 1 if m=0                                       ;
;  Cnm = sqrt( 2 * (n-m)! / (n+m)! )                    ;
; for all n,m combinations where n=0-nmax and m=0-n     ;
;                                                       ;
; Inputs:                                               ;
;  nmax -   The maximum degree of the Assoc. Leg. Poly. ;
;  x    -   A number (should be between -1 and 1)       ;
;                                                       ;
; Output:                                               ;
;  P    -   A matrix with dimension [nmax+1,nmax+1]     ;
;           that contains P(n,m,x), stored in element   ;
;           [n,m], and 0 everywhere else                ;
;           Double precision is used                    ;
;                                                       ;
; Keywords:                                             ;
;                                                       ;
; Uses the recursion relation:                          ;
;  sqrt( n^2 - m^2) * P(n,m,x) =                        ;
;     x * (2*n-1) * P(n-1,m,x) -                        ;
;     sqrt( (n+m-1)*(n-m-1) ) * P(n-2,m,x)              ;
; Where:                                                ;
;  P(m,m) = Cnm * (1-x^2)^(m/2) * (2*m-1)!!             ;
;           !! = product of all odd integers <= 2m-1    ;
;  P(m-1,m) = 0                                         ;
;                                                       ;
; Assumes the user doesn't feed the routine nonsense    ;
;                                                       ;
; Dave Brain                                            ;
; October 5, 2001                                       ;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

   P = dblarr(nmax+1,nmax+1)

   P[0,0] = 1d

   IF nmax GT 0 THEN BEGIN
      twoago = 0d0
      FOR i = 1, nmax DO BEGIN
         P[i,0] = ( x * (2d0*i - 1d0) * P[i-1,0] - $
                   (i - 1d0) * twoago ) / (i)
         twoago = P[i-1,0]
      ENDFOR
   ENDIF

   Cm = sqrt(2D0)
   FOR m = 1d0, nmax DO BEGIN

      Cm = Cm / sqrt(2d0*m*(2d0*m-1d0))

      P[m,m] = (1d0 - x^2)^(0.5d0 * m) * Cm

      FOR i = 1d0, m-1 DO P[m,m] = (2d0*i + 1d0) * P[m,m]

      IF nmax GT m THEN BEGIN
         twoago = 0d0
         FOR i = m+1d0, nmax DO BEGIN
            P[i,m] = ( x * (2d0*i - 1d0) * P[i-1,m] - $
                       sqrt( (i+m-1d0) * (i-m-1d0) ) * twoago ) / $
                     sqrt( ( i*i - m*m ) )
            twoago = P[i-1,m]
         ENDFOR
      ENDIF

   ENDFOR; m = 1D, nmax

   return, P

end
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
function sph_b, g, h, a_over_r, sct, scp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; sph_b.pro                                     ;
;                                               ;
; Routine to calculate vector magnetic field    ;
;  at a given location (a_over_r,sct,scp) in    ;
;  spherical coordinates from a spherical       ;
;  harmonic model                               ;
;                                               ;
; Inputs:                                       ;
;  g, h are the coefficients, in square arrays  ;
;   with dimensions [nmax+1,nmax+1].  Coeffs    ;
;   are stored according to [n,m].              ;
;  a_over_r is the vaue of a/r in the spherical ;
;   harmonic expansion, or the mean planetary   ;
;   radius divided by the radius at which you   ;
;   are calculating the field                   ;
;  sct, scp are the colatitude and east         ;
;   longitude at which you are calculating      ;
;   the field, IN RADIANS                       ;
;                                               ;
; Output:                                       ;
;  [Br, Bt, Bp] at scr, sct, scp                ;
;                                               ;
; Dave Brain                                    ;
; October 8, 2001 - sct, scp to radians         ;
;                   a,r to a_over_r             ;
; October 4, 2001                               ;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

   ; Determine nmax
      nmax = n_elements(g[0,*]) - 1
      cntarr = dindgen(nmax+1)

   ; Compute R(r) and dR(r) at each n
   ;  Only compute parts inside the summation over n
   ;  R(r) = [a/r]^(n+1)
   ;  dR(r) = (n+1)*[a/r]^(n+1)  ( Factors omitted that can move
   ;                               outside of summation - see
   ;                               pg 34 in Thesis Book 2 )
      R = (a_over_r)^(cntarr+1)
      dR = R*(cntarr+1)


   ; Compute Phi(phi) and dPhi(phi) at each m,n combo
   ;  Phi(phi) = gnm * cos(m*phi) + hnm * sin(m*phi)
   ;  dPhi(phi) = m * [-gnm * sin(m*phi) + hnm * cos(m*phi)]
      cos_m_phi = cos( cntarr * scp )
      sin_m_phi = sin( cntarr * scp )

      Phi  = g*0d
      dPhi = Phi

      FOR n = 1, nmax DO BEGIN
         Phi[n,*]  = cos_m_phi * g[n,*] + sin_m_phi * h[n,*]
         dPhi[n,*] = ( cos_m_phi * h[n,*] - sin_m_phi * g[n,*] ) * cntarr
      ENDFOR; n = 1, nmax


   ; Compute Theta and dTheta at each m,n combo
   ;  Theta(theta) = P(n,m,x)  the Schmidt normalized associated legendre poly.
   ;  dTheta(theta) = m * cos(theta) / sin(theta) * P(n,m,x) -
   ;                  C(n,m) / C(n,m+1) * P(n,m+1,x)
   ;                  Where C(n,m) = 1 if m=0
   ;                               = ( 2 * (n-m)! / (n+m)! ) ^ (1/2)
   ;                  Cool math tricks are involved
      cos_theta = cos(sct)
      sin_theta = sin(sct)

      Theta = legendre_schmidt_all(nmax,cos_theta)
      reftime1 = systime(1)
      dTheta = g*0d

      dTheta[1,*] = cntarr * cos_theta / sin_theta * Theta[1,*]
      dTheta[1,0] = dTheta[1,0] - Theta[1,1]

      FOR n = 2, nmax DO BEGIN
         dTheta[n,*] = cntarr * cos_theta / sin_theta * Theta[n,*]
         dTheta[n,0] = dTheta[n,0] - $
                       sqrt( (n * (n+1)) * 0.5d ) * Theta[n,1]
         dTheta[n,1:n] = dTheta[n,1:n] - $
                         sqrt( (n-cntarr[1:n]) * (n+cntarr[1:n]+1) ) * $
                          [ [ Theta[n,2:n] ], [ 0d ] ]
      ENDFOR; n = 1, nmax


   ; Put it all together

   ; Br = a/r Sum(n=1,nmax) { (n+1) * R(r) *
   ;      Sum(m=0,n) { Theta(theta) * Phi(phi) } }
      br = total( Theta*Phi, 2 )      ; Sum over m for each n
      br = total( br * dR ) * a_over_r ; (0th element contributes 0)

   ; Btheta = B_SN
   ; Btheta = a*sin(theta)/r Sum(n=1,nmax) { R(r) *
   ;          Sum(m=0,n) { dTheta(theta) * Phi(phi) } }
      bt = total( dTheta*Phi, 2 )      ; Sum over m for each n
      bt = -1.d * total( bt * R ) * a_over_r ; (0th element contributes 0)

   ; Bphi = B_EW
   ; Bphi = -a/r/sin(theta) Sum(n=1,nmax) { R(r) *
   ;        Sum(m=0,n) { Theta(theta) * DPhi(phi) } }
      bp = total( Theta*dPhi, 2 )      ; Sum over m for each n
      bp = -1.d * total( bp * R ) * a_over_r / sin_theta ; ( 0th element
                                                         ;   contributes 0 )


   ; Return the vector field
      return, [br, bt, bp]

end

pro bcrust
    model_file = '/Users/hilaryegan/Library/IDL/maven_sw/projects/maven/models/martiancrustmodels.sav'
    restore, model_file

    g = ga ; expansion coefficients
    h = ha ; expansion coefficients

    struc = read_csv("Output/coords_geo_sphere.csv")
    Rmars = 3390

    r = struc.field1
    theta = struc.field2; 
    phi = struc.field3
    delta = struc.field4
    Rdr = Rmars/r ; 

    N = size(R, /N_elements)
    bbx_0 = make_array(N, /double, value=0)
    bby_0 = make_array(N, /double, value=0)
    bbz_0 = make_array(N, /double, value=0)
    bbx_1 = make_array(N, /double, value=0)
    bby_1 = make_array(N, /double, value=0)
    bbz_1 = make_array(N, /double, value=0)

    for i=0,N-1 do begin 
        print, i
        bb_0 = sph_b(g, h, Rdr[i], theta[i], phi[i])

        bbx_0[i] = bb_0[0]
        bby_0[i] = bb_0[1]
        bbz_0[i] = bb_0[2]

        bb_1 = sph_b(g, h, Rdr[i], theta[i], phi[i]+delta)

        bbx_1[i] = bb_1[0]
        bby_1[i] = bb_1[1]
        bbz_1[i] = bb_1[2]
    endfor

    dat = [bbx_0, bby_0, bbz_0, bbx_1, bby_1, bbz_1]
    dat = transpose(reform(dat, N, 6))
    write_csv, 'Output/test_bcrust_3.csv', dat

end



; First calculate the Bcrust along the orbit
;orbit = 2349
;t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
;t0_unix = (mvn_orbit_num(orbnum=orbit-1)+mvn_orbit_num(orbnum=orbit))/2
;t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
;
;trange = [time_string(t0_unix), time_string(t1_unix)]
;time = indgen(t1_unix-t0_unix-20, start=t0_unix+10, /l64)
;t1_unix = (mvn_orbit_num(orbnum=orbit+1)+mvn_orbit_num(orbnum=orbit))/2
;
;trange = [time_string(t0_unix), time_string(t1_unix)]
;time = indgen(t1_unix-t0_unix-20, start=t0_unix+10, /l64)
;   mvn_model_bcrust, time, data=data, /arkani, nmax=60
;
;   ;mvn_model_bcrust, trange, data=bcrust
;   ;get_data, 'mvn_mod_bcrust_mso', time, bcrust, val
;   bcrust= transpose(data.ss)
;
;   dat = [time-time[0], bcrust[*, 0], bcrust[*, 1], bcrust[*, 2]]
;   dat = transpose(reform(dat, size(time, /N_Elements), 4))
;   print, time[0]
;
;   write_csv, 'Output/test_bcrust.csv', dat
;   mvn_kp_read, trange, insitu_0
;   mvn_kp_resample, insitu_0, time, insitu
;
;   pos = [insitu.spacecraft.GEO_x, insitu.spacecraft.GEO_y, insitu.spacecraft.GEO_z] 
;   pos = reform(pos, size(time, /N_elements),3)
;   ;pos = transpose(pos)
;
;   mk = mvn_spice_kernels(/all, /load, trange=trange)
;   mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=data, pos=pos, /arkani, nmax=60
;
;   bss = transpose(spice_vector_rotate(data.pc, time, 'IAU_MARS', 'MAVEN_MSO'))
;
;   dat = [time-time[0], bss[*, 0], bss[*, 1], bss[*, 2]]
;
;   dat = transpose(reform(dat, size(time, /N_Elements), 4))
;   print, time[0]

;write_csv, 'Output/test_bcrust_0.csv', dat

;   strucr = read_csv("Output/coords_geo_real.csv")
;   GEO_real = [strucr.field1, strucr.field2, strucr.field3]
;   GEO_real = transpose(reform(GEO_real, size(strucr.field1, /N_elements), 3))
;
;   strucf = read_csv("Output/coords_geo_frozen.csv")
;   GEO_frozen = [strucf.field1, strucf.field2, strucf.field3]
;   GEO_frozen = transpose(reform(GEO_frozen, size(strucf.field1, /N_elements), 3))
;
;
;
;   orbit=2349
;   mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=B_frozen, pos=GEO_frozen, /arkani, nmax=60
;   mvn_model_bcrust, time_string(mvn_orbit_num(orbnum=orbit)), data=B_real, pos=GEO_real, /arkani, nmax=60
;
;   bf = transpose(B_frozen.pc)
;   br = transpose(B_real.pc)
;
;   dat = [ bf[*, 0], bf[*, 1], bf[*, 2], br[*,0], br[*,1], br[*,2]]
;
;   dat = transpose(reform(dat, size(bf[*,0], /N_Elements), 6))
;
;   write_csv, 'Output/test_bcrust_2.csv', dat
