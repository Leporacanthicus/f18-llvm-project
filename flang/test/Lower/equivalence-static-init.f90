! RUN: bbc -emit-fir -o - %s | FileCheck %s

! Test explicit static initialization of equivalence storage

subroutine test_eqv_init
  integer, save :: link(3)
  integer :: i = 5
  integer :: j = 7
  equivalence (j, link(1))
  equivalence (i, link(3))
end subroutine

! CHECK-LABEL: fir.global internal @_QFtest_eqv_initEi : !fir.array<3xi32> {
    ! CHECK: %[[VAL_1:.*]] = fir.undefined !fir.array<3xi32>
    ! CHECK: %[[VAL_2:.*]] = fir.insert_value %0, %c7{{.*}}, [0 : index] : (!fir.array<3xi32>, i32) -> !fir.array<3xi32>
    ! CHECK: %[[VAL_3:.*]] = fir.insert_value %1, %c0{{.*}}, [1 : index] : (!fir.array<3xi32>, i32) -> !fir.array<3xi32>
    ! CHECK: %[[VAL_4:.*]] = fir.insert_value %2, %c5{{.*}}, [2 : index] : (!fir.array<3xi32>, i32) -> !fir.array<3xi32>
    ! CHECK: fir.has_value %[[VAL_4]] : !fir.array<3xi32>
! CHECK: }
