# CMake generated Testfile for 
# Source directory: /glade/u/home/bneuman/codes/10_Nsight/fortran
# Build directory: /glade/u/home/bneuman/codes/10_Nsight/fortran/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SERIAL_TEST "./check_output.sh" "./serial_test" "1e-13" "4.5e-5")
set_tests_properties(SERIAL_TEST PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;69;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
add_test(MPI_TEST "./check_output.sh" "./mpi_test" "1e-13" "4.5e-5")
set_tests_properties(MPI_TEST PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;87;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
add_test(OPENMP_TEST "./check_output.sh" "./openmp_test" "1e-13" "4.5e-5")
set_tests_properties(OPENMP_TEST PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;112;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
add_test(OPENACC_TEST "./check_output.sh" "./openacc_test" "1e-13" "4.5e-5")
set_tests_properties(OPENACC_TEST PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;137;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
add_test(OPENACC_TEST_EX "./check_output.sh" "./openacc_test_ex" "1e-13" "4.5e-5")
set_tests_properties(OPENACC_TEST_EX PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;153;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
add_test(OPENACC_TEST_EX2 "./check_output.sh" "./openacc_test_ex2" "1e-13" "4.5e-5")
set_tests_properties(OPENACC_TEST_EX2 PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;169;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
add_test(DO_CONCURRENT_TEST "./check_output.sh" "./do_concurrent_test" "1e-13" "4.5e-5")
set_tests_properties(DO_CONCURRENT_TEST PROPERTIES  _BACKTRACE_TRIPLES "/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;227;add_test;/glade/u/home/bneuman/codes/10_Nsight/fortran/CMakeLists.txt;0;")
