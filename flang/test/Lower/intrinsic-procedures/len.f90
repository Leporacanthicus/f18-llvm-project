! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: len_test
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1
  ! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i64
  ! CHECK: %[[x:.*]] = fir.convert %[[xx]] : (i64) -> i32
  ! CHECK: fir.store %[[x]] to %arg0
  i = len(c)
end subroutine
