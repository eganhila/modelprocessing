! !===================================================================
! subroutine user_get_b0(X1,Y1,Z1,B1)
!   use ModMain
!   use ModPhysics 
!   use ModNumConst
!   implicit none
! 
!   real, intent(in) :: X1,Y1,Z1
!   real, intent(out), dimension(3) :: B1
!   
!   real :: R0, theta, phi, rr, X0, Y0, Z0, delta
!   real, dimension(3) :: bb, B0, B2
!   real :: sinp, cosp, sint, cost
!   real :: X2, Y2, Z2
!   !-------------------------------------------------------------------------
!   call timing_start('user_get_b0')
!
!   if(USEMSO)then
!       uv = sqrt (u*u+v*v)
!       cost1 = u/ uv
!       sint1 = v/ uv
!rotate around Z axis to make the axis in the XZ plane
!      x0 = x1*cost1 + y1*sint1
!      y0 = -x1*sint1 + y1*cost1
!rotate around Y axis
!      X2 = X0*w-Z1*uv
!      Y2 = Y0
!      Z2 = X0*uv+Z1*w
!rotate back around Z axis so that the subsolar point is along the x axis
!       uvw=sqrt(w*w*u*u+v*v)
!       cost2=w*u/uvw
!       sint2 = v/uvw
!      x0=x2*cost2-y2*sint2
!      z0=z2
!      y0=x2*sint2+y2*cost2
!
!   else
!      X0 = X1*cos(thetilt)-Z1*sin(thetilt)
!      Y0 = Y1
!      Z0 = X1*sin(thetilt)+Z1*cos(thetilt)      
!   end if
!
!   R0 = sqrt(X0*X0 + Y0*Y0 + Z0*Z0)
!   rr = max(R0, 1.00E-6)
!   if(abs(X0).lt.1e-6) then
!      if(Y0.lt.0) then
!         phi=-cPi/2.
!      else
!         phi=cPi/2.
!      endif
!   else 
!      if(X0.gt.0) then		
!         phi=atan(Y0/X0)
!      else 
!         phi=cPi+atan(Y0/X0)
!      endif
!   endif
!   
!
!   !RotPeriodSi=0.0 if not use rotation
!   !delta=rot-Time_Simulation*VRad  !(Vrad=cTwoPi/RotPeriodSi)
!   delta = rot
!
!   If(RotPeriodSi > 0.0) then
!      delta=rot-Time_Simulation/RotPeriodSi*cTwoPi
!   end If
!
!    write(*,*)'RotPeriodSi=',RotPeriodSi
!
!   theta=acos(Z0/rr)
!
!   call MarsB0(R0,theta, phi+delta, bb)
!
!   sint=sin(theta)
!   cost=cos(theta)
!   sinp=sin(phi)
!   cosp=cos(phi)
!
!   B0(1) = bb(1)*sint*cosp+bb(2)*cost*cosp-bb(3)*sinp
!   B0(2) = bb(1)*sint*sinp+bb(2)*cost*sinp+bb(3)*cosp 
!   B0(3) = bb(1)*cost-bb(2)*sint
!  
!   if(USEMSO)then  !rotate around X axis
!      B1(1) = B0(1)*cost2+B0(2)*sint2 
!      B1(2) = -B0(1)*sint2+B0(2)*cost2 
!      B1(3) = B0(3)
!
!      B2(1)=w*B1(1)+uv*B1(3)  
!      B2(2)=B1(2)
!      B2(3)=-uv*B1(1)+w*B1(3)
!      
!      B1(1) = B2(1)*cost1-B2(2)*sint1 
!      B1(2) = B2(1)*sint1+B2(2)*cost1 
!      B1(3) = B2(3)
!
!   else
!      B1(1) = B0(1)*cos(thetilt)+B0(3)*sin(thetilt)
!      B1(2) = B0(2)
!      B1(3) = -B0(1)*sin(thetilt)+B0(3)*cos(thetilt)
!   end if
!
!   ! Normalize the crustal magnetic field
!   B1(1)=B1(1)*Io2No_V(UnitB_)
!   B1(2)=B1(2)*Io2No_V(UnitB_)
!   B1(3)=B1(3)*Io2No_V(UnitB_)
!
!   call timing_stop('user_get_b0')
! end subroutine user_get_b0

  !===========================================================================
  subroutine MarsB0(r,theta, phi, bb)
    implicit none  
    
    integer, parameter:: nMax=62
    real, intent(in) :: r, theta, phi
    real, dimension(1:3),intent(out) :: bb
    !real :: Rlgndr, dRlgndr
    integer :: NN, n, m, im, NNm, i
    real :: dRnm, signsx, Rmm, mars_eps
    real :: xtcos,xtsin,xtabs, xx
    real, dimension(0:nMax-1) :: xpcos, xpsin
    real :: a, arr, arrn, arrm, somx2, fact, temp
    real,dimension(0:nMax,0:nMax) :: Rnm
    real,dimension(0:nMax,0:nMax) :: cmars, dmars
    real,dimension(0:nMax), save  :: Factor1_I, Factor2_I, Factor3_I
    real,dimension(0:nMax,0:nMax), save :: Factor1_II, Factor2_II, Factor3_II
    logical :: DoSetFactor = .true.

    !-------------------------------------------------------------------------

    NNm=60
    open(15,file='Output/marsmgsp.txt')
    do i=0,NNm 
    read(15,*)n,(cmars(n-1,m),m=0,n-1),(dmars(n-1,m),m=0,n-1)
    end do
    close(15)


    if(DoSetFactor)then
       DoSetFactor = .false.
       do m = 0, nMax
          Factor1_I(m) = sqrt((2.*m+2.)*(2.*m+1.))
          Factor2_I(m) = sqrt(4.*m+6.)
          Factor3_I(m) = sqrt(2.*m+5.)
          do n = m, nMax
             if(n>m+2)then
                temp= sqrt((n-1.-m)*(n+m+1.)/(2.*n+1.)/(n-m-2.))
                Factor1_II(n,m) = sqrt((2.*n-1.)/(n-m-2.))/temp
                Factor2_II(n,m) = sqrt((n+m)/(2.*n-3.))/temp
             end if
             Factor3_II(n,m) = sqrt((n-m)*(n+m+1.))
          end do
       end do
    end if

    a=1.035336
    arr=a/r
    
       
    mars_eps=1e-3
       
    if(r.lt.1.0) then 
       NN=0       
    else 
       NN=NNm-1
    endif

    xtcos=cos(theta)
    xtsin=sin(theta)
                            
    do im=0,NN
       xpcos(im)=cos(im*phi)
       xpsin(im)=sin(im*phi)
    end do
    
    bb(1)=0.0
    bb(2)=0.0
    bb(3)=0.0
    !	    somx2=sqrt((1.-xtcos)*(1.+xtcos))
    somx2=abs(xtsin)
    signsx=sign(1., xtsin)
    
    fact=1.
    Rmm=1.
    Rnm(0,0)=sqrt(2.)
    Rnm(1,0)=xtcos*sqrt(3.)*Rnm(0,0)
    do n=2, NN
       Rnm(n, 0)=(xtcos*sqrt((2.*n-1.)*(2.*n+1.))*Rnm(n-1,0)-&
            (n-1)*sqrt((2.*n+1.)/(2.*n-3.))*Rnm(n-2, 0))/n
       
    enddo !n
    arrm=1.0


    do m=0, NN

       Rmm=Rmm*fact*somx2/Factor1_I(m)
       
       Rnm(m+1,m+1)=Rmm*Factor2_I(m) 
       Rnm(m, m+1)=0
       
       fact=fact+2.
       arrm=arr*arrm
       arrn=arrm
       do n=m,NN
          arrn=arr*arrn
          !write(*,*) 'arrn=', arrn, ' n=', n
          if(n> (m+2)) then
             Rnm(n,m+1) = xtcos*Factor1_II(n,m)*Rnm(n-1,m+1)-&
                  Factor2_II(n,m)*Rnm(n-2,m+1)
             
          else if(n > (m+1)) then
             Rnm(n,m+1)=xtcos*Factor3_I(m)*Rnm(m+1,m+1)
          endif

          dRnm=m*xtcos*Rnm(n,m)/xtsin-Rnm(n, m+1)*signsx* Factor3_II(n,m)
          
          bb(1)=bb(1)+(n+1)*arrn*Rnm(n,m)*(cmars(n,m)*xpcos(m)&
               +dmars(n,m)*xpsin(m))
          bb(2)=bb(2)-arrn*dRnm*(cmars(n,m)*&
               xpcos(m)+dmars(n,m)*xpsin(m))
          if(xtsin <= 1e-6) then
             bb(3)=0.
          else
             bb(3)=bb(3)-arrn*Rnm(n,m)*m/xtsin*(-cmars(n,m)*xpsin(m)&
                  +dmars(n,m)*xpcos(m))
          endif
       end do !n
    end do !m


  end subroutine MarsB0

!=====================================================================
