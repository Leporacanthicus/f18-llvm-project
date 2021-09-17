! This test checks lowering of OpenMP allocate Directive.
! XFAIL: *
! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program main
  integer :: x, y

  !$omp allocate(x, y)
end
