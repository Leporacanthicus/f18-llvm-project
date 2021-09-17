! This test checks lowering of OpenMP declare reduction Directive.
! XFAIL: *
! RUN: %bbc -fopenmp -emit-fir %s -o  - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

subroutine declare_red()
  integer :: my_var
  !$omp declare reduction (my_red : integer : omp_out = omp_in) initializer (omp_priv = 0)
  my_var = 0
end subroutine declare_red
