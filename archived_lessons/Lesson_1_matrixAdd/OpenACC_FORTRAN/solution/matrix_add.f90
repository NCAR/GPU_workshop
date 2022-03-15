program matrix_add
   use openacc
   implicit none
   character(10) :: rowsChar
   character(10) :: colsChar
   integer, parameter:: DEFAULT_DIM=1024
   real, parameter:: MAT_A_VAL=3.0
   real, parameter:: MAT_B_VAL=2.0
   real, parameter:: VERIF_TOL=1.0E-6
   integer :: i, j, rows, cols
   integer :: t1, t2, dt, count_rate, count_max
   real, allocatable, dimension(:,:) :: a, b, c_cpu, c_gpu
   real :: tmp, secs
   logical:: ver_flag


   if(COMMAND_ARGUMENT_COUNT().EQ.0) then
        rows = DEFAULT_DIM
        cols = DEFAULT_DIM 
   else if(COMMAND_ARGUMENT_COUNT().EQ.2) then
        call GET_COMMAND_ARGUMENT(1,rowsChar)   !first, read in the two values
        call GET_COMMAND_ARGUMENT(2,colsChar)
        read(rowsChar,*)rows
        read(colsChar,*)cols
   else
        write(*,*)'ERROR, Usage: ./matrix_add.exe rows cols\n'
        stop
   endif

   call system_clock(count_max=count_max, count_rate=count_rate)

      call system_clock(t1)

      allocate( a(rows,cols), b(rows,cols), c_cpu(rows,cols), c_gpu(rows,cols) )

! Initialize matrices
      do j=1,cols
         do i=1,rows
            a(i,j) = MAT_A_VAL
            b(i,j) = MAT_B_VAL
         enddo
      enddo

      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)
      write(*,"('Initialized Mat A & B, sizes ',i6,' x ',i6,' in ',f12.5,' secs')") rows,cols, secs

! Compute matrix addition on CPU

      call system_clock(t1)
      do j=1,cols
         do i=1,rows
               c_cpu(i,j) = a(i,j) + b(i,j)
         enddo
      enddo
    
      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)
      write(*,"('CPU Matrix Addition completed in ',f12.5,' secs')") secs
 
! Compute matrix addition on GPU

      call system_clock(t1)

!$acc data copyin(a,b) copyout(c_gpu)
!$acc parallel loop collapse(2)
      do j=1,cols
         do i=1,rows
               c_gpu(i,j) = a(i,j) + b(i,j)
         enddo
      enddo
!$acc end parallel
!$acc end data

      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)
      write(*,"('GPU Matrix Addition completed in ',f12.5,' secs')") secs

! Verify GPU results against CPU
      ver_flag = 1
      jloop: do j=1,cols
         do i=1,rows
               if (abs(c_gpu(i,j)-c_cpu(i,j)) > VERIF_TOL) then
                        write(*,"('Verification failed')")
                        write(*,"('   1st error > tolerance encountered at C_gpu[',i6,'][',i6,']')") i, j
                        write(*,"('   C_cpu[',i6,'][',i6,']=',f12.2,'')") i,j,c_cpu(i,j)
                        write(*,"('   C_gpu[',i6,'][',i6,']=',f12.2,'')") i,j,c_gpu(i,j)
                        ver_flag = 0
                        exit jloop
                end if

         enddo
      enddo jloop

      if(ver_flag) then
         write(*,"('Verification passed')")
      end if
      deallocate(a, b,c_cpu,c_gpu)
end program matrix_add

