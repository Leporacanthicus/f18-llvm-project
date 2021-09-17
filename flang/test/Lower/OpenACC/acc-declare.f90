! This test checks lowering of OpenACC declare Directive.
! XFAIL: *
! RUN: %bbc -fopenacc -emit-fir %s -o - | FileCheck %s

program main
  real, dimension(10) :: aa, bb

  !$acc declare present(aa, bb)
end
