program gpu_workshop
use openacc
use netcdf
implicit none

!declaration of variables
integer :: moist_start, moist_end
real :: dts
real :: epssm

! This is the name of the data file we will read. 
character (len = *), parameter :: FILE_NAME = "vert_imp_coef_work.nc"

character(10) :: nVertLevelsChar
character(10) :: nCellsChar

integer, parameter:: nVertLevels_default = 56
integer, parameter:: nCells_default = 163842
real, parameter:: VERIF_TOL=1.0E-6
real :: rgas, cp, gravity
integer, nVertLevels
integer, nCells

real, dimension (:,:), allocatable :: qtot
real, dimension (:,:), allocatable  :: zz
real, dimension (:,:), allocatable  :: cqw
real, dimension (:,:), allocatable  :: p
real, dimension (:,:), allocatable  :: t
real, dimension (:,:), allocatable  :: rb
real, dimension (:,:), allocatable  :: rtb
real, dimension (:,:), allocatable  :: pb
real, dimension (:,:), allocatable  :: rt
real, dimension (:,:), allocatable  :: cofwr
real, dimension (:,:), allocatable  :: cofwz
real, dimension (:,:), allocatable  :: coftz
real, dimension (:,:), allocatable  :: cofwt
real, dimension (:,:), allocatable  :: a_tri
real, dimension (:,:), allocatable  :: alpha_tri
real, dimension (:,:), allocatable  :: gamma_tri
real, dimension (:), allocatable  :: cofrz
real, dimension (:), allocatable :: rdzw
real, dimension (:), allocatable :: fzm
real, dimension (:), allocatable :: fzp
real, dimension (:), allocatable :: rdzu

integer :: cellStart, cellEnd, edgeStart, edgeEnd
integer :: cellSolveStart, cellSolveEnd, edgeSolveStart, edgeSolveEnd

!
! Local variables
!
integer :: iCell, k, iq
real  :: dtseps, c2, qtotal, rcv
real , dimension(:,:), allocatable :: b_tri, c_tri

!end of declaration of variables

!Get the command line arguments
if(COMMAND_ARGUMENT_COUNT().EQ.0) then
   nVertLevels = nVertLevels_default
   nCells = nCells_default
else if(COMMAND_ARGUMENT_COUNT().EQ.2) then
   call GET_COMMAND_ARGUMENT(1,nVertLevelsChar)   !first, read in the two values
   call GET_COMMAND_ARGUMENT(2,nCellsChar)
   read(nVertLevelsChar,*)nVertLevels
   read(nCellsChar,*)nCells
else
   write(*,*)'ERROR, Usage: ./vert_implicit_coefs.exe nVertLevels nCells'
   stop
endif

print *,'Value of nCells: ',nCells
print *,'Value of nVertLevels',nVertLevels

!allocate variables for the kernels
allocate(qtot(nVertLevels,nCells+1))
allocate(zz(nVertLevels,nCells+1))
allocate(cqw(nVertLevels,nCells+1))
allocate(p(nVertLevels,nCells+1))
allocate(t(nVertLevels,nCells+1))
allocate(rb(nVertLevels,nCells+1))
allocate(rtb(nVertLevels,nCells+1))
allocate(pb(nVertLevels,nCells+1))
allocate(rt(nVertLevels,nCells+1))
allocate(cofwr(nVertLevels,nCells+1))
allocate(cofwz(nVertLevels,nCells+1))
allocate(coftz(nVertLevels+1,nCells+1))
allocate(cofwt(nVertLevels,nCells+1))
allocate(a_tri(nVertLevels,nCells+1))
allocate(alpha_tri(nVertLevels,nCells+1))
allocate(gamma_tri(nVertLevels,nCells+1))
allocate(cofrz(nVertLevels))
allocate(rdzw(nVertLevels))
allocate(fzm(nVertLevels))
allocate(fzp(nVertLevels))
allocate(rdzu(nVertLevels))

allocate(b_tri(nVertLevels ,nCells+1))
allocate(c_tri(nVertLevels ,nCells+1))

!initialize arrays with random numbers
  CALL RANDOM_NUMBER(qtot)
  CALL RANDOM_NUMBER(zz)
  CALL RANDOM_NUMBER(cqw)
  CALL RANDOM_NUMBER(p)
  CALL RANDOM_NUMBER(t)
  CALL RANDOM_NUMBER(rb)
  CALL RANDOM_NUMBER(rtb)
  CALL RANDOM_NUMBER(pb)
  CALL RANDOM_NUMBER(rt)
  CALL RANDOM_NUMBER(cofwr)
  CALL RANDOM_NUMBER(cofwz)
  CALL RANDOM_NUMBER(coftz)
  CALL RANDOM_NUMBER(cofwt)
  CALL RANDOM_NUMBER(cofrz)
  CALL RANDOM_NUMBER(rdzw)
  CALL RANDOM_NUMBER(fzm)
  CALL RANDOM_NUMBER(fzp)
  CALL RANDOM_NUMBER(rdzu)

!readin the variables necessary for the subroutine from the netcdf file


!Kernels of mpas_atm_compute_vert_imp_coefs_work
!$acc data present(cofrz, gamma_tri, a_tri, alpha_tri, &
!$acc coftz, cofwr, cofwt, cofwz, &
!$acc rdzw, cqw, fzm, fzp, p, pb, qtot, rb, rdzu, rt, rtb, t, zz)&
!$acc create(b_tri,c_tri)
  
      !  set coefficients
      dtseps = .5*dts*(1.+epssm)
      rcv = rgas/(cp-rgas)
      c2 = cp*rcv

!$acc parallel num_workers(8) vector_length(32)
!$acc loop vector
! MGD bad to have all threads setting this variable?
      do k=1,nVertLevels
         cofrz(k) = dtseps*rdzw(k)
      end do
!$acc end parallel


!$acc parallel num_workers(8) vector_length(32)
!$acc loop gang worker
      do iCell = cellSolveStart,cellSolveEnd  !  we only need to do cells we are solving for, not halo cells
!DIR$ IVDEP
         do k=2,nVertLevels
            cofwr(k,iCell) =.5*dtseps*gravity*(fzm(k)*zz(k,iCell)+fzp(k)*zz(k-1,iCell))
         end do
         coftz(1,iCell) = 0.0
!DIR$ IVDEP
         do k=2,nVertLevels
            cofwz(k,iCell) = dtseps*c2*(fzm(k)*zz(k,iCell)+fzp(k)*zz(k-1,iCell))  &
                 *rdzu(k)*cqw(k,iCell)*(fzm(k)*p (k,iCell)+fzp(k)*p (k-1,iCell))
            coftz(k,iCell) = dtseps*   (fzm(k)*t (k,iCell)+fzp(k)*t (k-1,iCell))
         end do
         coftz(nVertLevels+1,iCell) = 0.0
        end do
!$acc end parallel	  

!$acc parallel num_workers(8) vector_length(32)
!$acc loop gang worker private(qtotal)
        do iCell = cellSolveStart,cellSolveEnd
!DIR$ IVDEP
         do k=1,nVertLevels

            qtotal = qtot(k,iCell)

            cofwt(k,iCell) = .5*dtseps*rcv*zz(k,iCell)*gravity*rb(k,iCell)/(1.+qtotal)  &
                                *p(k,iCell)/((rtb(k,iCell)+rt(k,iCell))*pb(k,iCell))

         end do
        end do
!$acc end parallel

!$acc parallel num_workers(8) vector_length(32)
!$acc loop gang worker
        do iCell = cellSolveStart,cellSolveEnd
         a_tri(1,iCell) = 0.  ! note, this value is never used
         b_tri(1,iCell) = 1.    ! note, this value is never used
         c_tri(1,iCell) = 0.    ! note, this value is never used
         gamma_tri(1,iCell) = 0.
         alpha_tri(1,iCell) = 0.  ! note, this value is never used
        enddo
!$acc end parallel

!$acc parallel num_workers(8) vector_length(32)
!$acc loop gang worker
        do iCell = cellSolveStart,cellSolveEnd
!DIR$ IVDEP
         do k=2,nVertLevels
            a_tri(k,iCell) = -cofwz(k  ,iCell)* coftz(k-1,iCell)*rdzw(k-1)*zz(k-1,iCell)   &
                         +cofwr(k  ,iCell)* cofrz(k-1  )                       &
                         -cofwt(k-1,iCell)* coftz(k-1,iCell)*rdzw(k-1)
            b_tri(k,iCell) = 1.                                                  &
                         +cofwz(k  ,iCell)*(coftz(k  ,iCell)*rdzw(k  )*zz(k  ,iCell)   &
                                      +coftz(k  ,iCell)*rdzw(k-1)*zz(k-1,iCell))   &
                         -coftz(k  ,iCell)*(cofwt(k  ,iCell)*rdzw(k  )             &
                                       -cofwt(k-1,iCell)*rdzw(k-1))            &
                         +cofwr(k,  iCell)*(cofrz(k    )-cofrz(k-1))
            c_tri(k,iCell) =   -cofwz(k  ,iCell)* coftz(k+1,iCell)*rdzw(k  )*zz(k  ,iCell)   &
                         -cofwr(k  ,iCell)* cofrz(k    )                       &
                         +cofwt(k  ,iCell)* coftz(k+1,iCell)*rdzw(k  )
         end do
!        end do
!MGD VECTOR DEPENDENCE
!        do iCell = cellSolveStart,cellSolveEnd
         do k=2,nVertLevels
            alpha_tri(k,iCell) = 1./(b_tri(k,iCell)-a_tri(k,iCell)*gamma_tri(k-1,iCell))
            gamma_tri(k,iCell) = c_tri(k,iCell)*alpha_tri(k,iCell)
         end do

      end do ! loop over cells
!$acc end parallel

!$acc end data

print *,'Success running the code'

!Deallocate the variables
deallocate(qtot)
deallocate(zz) 
deallocate(cqw)
deallocate(p)
deallocate(t)
deallocate(rb)
deallocate(rtb)
deallocate(pb)
deallocate(rt)
deallocate(cofwr)
deallocate(cofwz)
deallocate(coftz)
deallocate(cofwt)
deallocate(a_tri)
deallocate(alpha_tri)
deallocate(gamma_tri)
deallocate(cofrz)
deallocate(rdzw)
deallocate(fzm)
deallocate(fzp)
deallocate(rdzu)

deallocate(b_tri)
deallocate(c_tri)

end program gpu_workshop
