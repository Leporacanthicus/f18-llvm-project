! RUN: bbc %s -o - | FileCheck %s

! CHECK: _QQmain
program test_eoshift
  integer, dimension(3,3) :: a
  real, dimension(3,3) :: b
  logical, dimension(3,3) :: c
  character(3), dimension(3,3) :: d
  integer, dimension(3) :: v

  ! INTEGER vector
  v = (/ 1, 2, 3 /)
  print '(3i3)', v

  v = EOSHIFT(v, SHIFT=1, BOUNDARY=-1)
  print *
  print '(3i3)', v

  ! INTEGER array
  a = reshape( (/ 1, 2, 3, 4, 5, 6, 7, 8, 9 /), (/ 3, 3 /))
  print *
  print '(3i3)', a(1,:)
  print '(3i3)', a(2,:)
  print '(3i3)', a(3,:)

  a = EOSHIFT(a, SHIFT=(/1, 2, 1/), BOUNDARY=-5, DIM=2)
  print *
  print '(3i3)', a(1,:)
  print '(3i3)', a(2,:)
  print '(3i3)', a(3,:)

  ! REAL array
  b = reshape( (/ 1., 2., 3., 4., 5., 6., 7., 8., 9. /), (/ 3, 3 /))
  print *
  print '(3f5.1)', b(1,:)
  print '(3f5.1)', b(2,:)
  print '(3f5.1)', b(3,:)

  b = EOSHIFT(b, SHIFT=(/1, 2, 1/), BOUNDARY=-1.0, DIM=1)
  print *
  print '(3f5.1)', b(1,:)
  print '(3f5.1)', b(2,:)
  print '(3f5.1)', b(3,:)

  ! LOGICAL array, no BOUNDARY or DIM given
  c = reshape( (/ .true., .true., .true., .true., .true., .true., .true., .true., .true. /), (/ 3, 3 /))
  print *
  print '(3l3)', c(1,:)
  print '(3l3)', c(2,:)
  print '(3l3)', c(3,:)

  c = EOSHIFT(c, SHIFT=(/1, 2, 1/))
  print *
  print '(3l3)', c(1,:)
  print '(3l3)', c(2,:)
  print '(3l3)', c(3,:)

  ! CHARACTER array, no BOUNDARY or DIM given
  d = reshape( (/ "foo", "foo", "foo", "bar", "bar", "bar", "baz", "baz", "baz" /), (/ 3, 3 /))
  print *
  print '(3a5)', d(1,:)
  print '(3a5)', d(2,:)
  print '(3a5)', d(3,:)

  d = EOSHIFT(d, SHIFT=(/1, 2, 1/))
  print *
  print '(3a5)', d(1,:)
  print '(3a5)', d(2,:)
  print '(3a5)', d(3,:)

end program test_eoshift
