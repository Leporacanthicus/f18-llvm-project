! RUN: bbc %s -o - | FileCheck %s

! CHECK: _QQmain
program test_scale
  real :: x = 207.4e-2
  integer :: i = 3

  print *,"scale(x,i) = ", scale(x,i)
  
end program test_scale
