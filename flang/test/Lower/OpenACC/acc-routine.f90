! This test checks lowering of OpenACC routine Directive.
! XFAIL: *
! RUN: %bbc -fopenacc -emit-fir %s -o - | FileCheck %s

program main
  !$acc routine(sub) seq
contains
  subroutine sub(a)
    real :: a(:)
  end
end
