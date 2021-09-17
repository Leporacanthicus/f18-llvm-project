! This test checks lowering of OpenMP declare simd Directive.
! XFAIL: *
! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

subroutine sub(x, y)
  real, intent(inout) :: x, y

  !$omp declare simd(sub) aligned(x)
  x = 3.14 + y
end
