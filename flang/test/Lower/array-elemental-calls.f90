! Test lowering of elemental calls in array expressions.
! RUN: bbc -o - -emit-fir %s | FileCheck %s

module scalar_in_elem

contains
elemental integer function elem_by_ref(a,b) result(r)
  integer, intent(in) :: a
  real, intent(in) :: b
  r = a + b
end function
elemental integer function elem_by_valueref(a,b) result(r)
  integer, value :: a
  real, value :: b
  r = a + b
end function

! CHECK-LABEL: func @_QMscalar_in_elemPtest_elem_by_ref(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xi32>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>>
subroutine test_elem_by_ref(i, j)
  integer :: i(100), j(100)
  ! CHECK: %[[tmp:.*]] = fir.alloca f32
  ! CHECK: %[[cst:.*]] = constant 4.200000e+01 : f32
  ! CHECK: fir.store %[[cst]] to %[[tmp]] : !fir.ref<f32>

  ! CHECK: fir.do_loop
    ! CHECK: %[[j:.*]] = fir.array_coor %[[arg1]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    ! CHECK: fir.call @_QMscalar_in_elemPelem_by_ref(%[[j]], %[[tmp]]) : (!fir.ref<i32>, !fir.ref<f32>) -> i32
    ! CHECK: fir.result
  i = elem_by_ref(j, 42.)
end

! CHECK-LABEL: func @_QMscalar_in_elemPtest_elem_by_valueref(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xi32>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>>
subroutine test_elem_by_valueref(i, j)
  integer :: i(100), j(100)
  ! CHECK-DAG: %[[tmpA:.*]] = fir.alloca i32 {adapt.valuebyref}
  ! CHECK-DAG: %[[tmpB:.*]] = fir.alloca f32 {adapt.valuebyref}
  ! CHECK: %[[jload:.*]] = fir.array_load %[[arg1]]
  ! CHECK: %[[cst:.*]] = constant 4.200000e+01 : f32
  ! CHECK: fir.store %[[cst]] to %[[tmpB]] : !fir.ref<f32>

  ! CHECK: fir.do_loop
    ! CHECK: %[[j:.*]] = fir.array_fetch %[[jload]], %{{.*}} : (!fir.array<100xi32>, index) -> i32
    ! CHECK: fir.store %[[j]] to %[[tmpA]] : !fir.ref<i32>
    ! CHECK: fir.call @_QMscalar_in_elemPelem_by_valueref(%[[tmpA]], %[[tmpB]]) : (!fir.ref<i32>, !fir.ref<f32>) -> i32
    ! CHECK: fir.result
  i = elem_by_valueref(j, 42.)
end
end module
