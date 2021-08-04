! RUN: bbc -emit-fir -o - %s | FileCheck %s

! Test explicit static initialization of equivalence storage

subroutine test_eqv_init
  integer, save :: link(3)
  integer :: i = 5
  integer :: j = 7
  equivalence (j, link(1))
  equivalence (i, link(3))
end subroutine

! CHECK-LABEL: fir.global internal @_QFtest_eqv_initEj : tuple<i32, !fir.array<4xi8>, i32> {
  ! CHECK: %[[VAL_0:.*]] = fir.undefined tuple<i32, !fir.array<4xi8>, i32>
  ! CHECK: %[[VAL_1:.*]] = constant 7 : i32
  ! CHECK: %[[VAL_2:.*]] = constant 0 : index
  ! CHECK: %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_1]], [0 : index] : (tuple<i32, !fir.array<4xi8>, i32>, i32) -> tuple<i32, !fir.array<4xi8>, i32>
  ! CHECK: %[[VAL_4:.*]] = constant 5 : i32
  ! CHECK: %[[VAL_5:.*]] = constant 2 : index
  ! CHECK: %[[VAL_6:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_4]], [2 : index] : (tuple<i32, !fir.array<4xi8>, i32>, i32) -> tuple<i32, !fir.array<4xi8>, i32>
  ! CHECK: fir.has_value %[[VAL_6]] : tuple<i32, !fir.array<4xi8>, i32>
! CHECK }
