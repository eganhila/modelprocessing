! This is a subroutine I got from Yingjuan Ma for calculating
! the crustal field at a location (r, theta, phi) in 
! geographic coordinates. 


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
