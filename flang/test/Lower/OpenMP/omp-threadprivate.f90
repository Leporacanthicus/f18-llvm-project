! This test checks lowering of OpenMP threadprivate Directive.
! XFAIL: *
! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

program main
  integer, save :: x, y

  !$omp threadprivate(x, y)
end
