! RUN: bbc %s -o - | FileCheck %s

! CHECK: _QQmain
program test_nearest
  real :: x = 207.0
  real :: s = 2.0
  real :: negS = -2.0

  print *,"nearest(x,s) = ", nearest(x, s)
  print *,"nearest(x,negS) = ", nearest(x, negS)
  print *,"difference = ", nearest(x, s) - nearest(x, negS)
  
end program test_nearest
