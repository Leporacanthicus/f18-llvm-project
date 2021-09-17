! This test checks lowering of OpenMP declare target Directive.
! XFAIL: *
! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

module mod1
contains
  subroutine sub()
    integer :: x, y
    !$omp declare target
  end
end module
