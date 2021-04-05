program matrix_mult
   use openacc
   implicit none
   character(10) :: rowsAChar
   character(10) :: colsAChar
   character(10) :: rowsBChar
   character(10) :: colsBChar
   integer, parameter:: DEFAULT_DIM=1024
   real, parameter:: MAT_A_VAL=3.0
   real, parameter:: MAT_B_VAL=2.0
   real, parameter:: VERIF_TOL=1.0E-6
   integer :: i, j, k,rowsA, colsA, rowsB, colsB
   integer :: t1, t2, dt, count_rate, count_max
   real, allocatable, dimension(:,:) :: a, b, c_cpu, c_gpu
   real :: tmp, secs
   logical:: ver_flag


   if(COMMAND_ARGUMENT_COUNT().EQ.0) then
        rowsA = DEFAULT_DIM
        colsA = DEFAULT_DIM 
        rowsB = DEFAULT_DIM
        colsB = DEFAULT_DIM
   else if(COMMAND_ARGUMENT_COUNT().EQ.4) then
        call GET_COMMAND_ARGUMENT(1,rowsAChar)   !first, read in the two values
        call GET_COMMAND_ARGUMENT(2,colsAChar)
        call GET_COMMAND_ARGUMENT(3,rowsBChar)
        call GET_COMMAND_ARGUMENT(4,colsBChar)
        read(rowsAChar,*)rowsA
        read(colsAChar,*)colsA
        read(rowsBChar,*)rowsB
        read(colsBChar,*)colsB
        !Verify multiplication is possible
        if(colsA .NE. rowsB) then
          write(*,*)'ERROR, Inner dimension mismatch. # Columns of Mat A must equal # Rows of Mat B'
          write(*,*)'Usage: ./matrix_mult.exe rowsA colsA rowsB colsB'
          stop
        endif
   else
        write(*,*)'ERROR, Usage: ./matrix_mult.exe rowsA colsA rowsB colsB'
        stop
   endif

!Initialize time variables
   call system_clock(count_max=count_max, count_rate=count_rate)

      call system_clock(t1)
!Reserve memory for the matricies on CPU
      allocate( a(rowsA,colsA), b(rowsB,colsB), c_cpu(rowsA,colsB), c_gpu(rowsA,colsB) )

! Initialize matrices with a constant value defined by MAT_X_VAL
      do j=1,colsA
         do i=1,rowsA
            a(i,j) = MAT_A_VAL
         enddo
      enddo

      do j=1,colsB
         do i=1,rowsB
            b(i,j) = MAT_B_VAL
         enddo
      enddo
!Compute Initialization timing data
      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)
      write(*,"('Initialized Mat A, size ',i6,' x ',i6,' and ')") rowsA,colsA
      write(*,"('Initialized Mat B, size ',i6,' x ',i6,' in ',f12.5,'secs')") rowsB,colsB, secs

! Compute matrix Multiplication on CPU

      call system_clock(t1)
      do j=1,colsB
         do i=1,rowsA
            tmp = 0.0
            do k=1,rowsB
                tmp = tmp + a(i,k) * b(k,j)
            enddo
            c_cpu(i,j) = tmp
         enddo
      enddo
    
      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)
      write(*,"('CPU Matrix Multiplication completed in ',f12.5,' secs')") secs
 
! Compute matrix addition on GPU
! ADD OPENACC FORTRAN DIRECTIVES TO OFFLOAD GPU COMPUTATION

      call system_clock(t1)

      do j=1,colsB
         do i=1,rowsA
            tmp = 0.0
            do k=1,rowsB
                tmp = tmp + a(i,k) * b(k,j)
            enddo
            c_gpu(i,j) = tmp
         enddo
      enddo


      call system_clock(t2)
      dt = t2-t1
      secs = real(dt)/real(count_rate)
      write(*,"('GPU Matrix Multiplication completed in ',f12.5,' secs')") secs

! Verify GPU results against CPU results for correctness
      ver_flag = 1
      jloop: do j=1,colsB
         do i=1,rowsA
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
 
! Uncomment section to print CPU and GPU results 
     
!      write(*,"('CPU Results: ')")
!      do i=1,rowsA
!          do j=1,colsB
!               write(*,fmt="(f0.2, tr2)",advance="no") c_cpu(i,j)
!          enddo
!               write(*,"(' ')")
!      enddo
!      write(*,"('GPU Results: ')")
!      do i=1,rowsA
!          do j=1,colsB
!               write(*,fmt="(f0.2, tr2)",advance="no") c_gpu(i,j)
!          enddo
!               write(*,"(' ')")
!      enddo

      if(ver_flag) then
         write(*,"('Verification passed')")
      end if
!Release memory to cleanup program
      deallocate(a, b,c_cpu,c_gpu)
end program matrix_mult

