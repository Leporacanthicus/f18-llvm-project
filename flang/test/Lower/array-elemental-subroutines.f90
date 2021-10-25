! Test lowering of elemental subroutine calls with array arguments
! RUN: bbc -o - -emit-fir %s | FileCheck %s

! CHECK-LABEL: func @_QPtest_elem_sub(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME:                         %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
! CHECK-SAME:                         %[[VAL_2:.*]]: !fir.ref<i32>,
! CHECK-SAME:                         %[[VAL_3:.*]]: !fir.ref<!fir.complex<4>>) {
subroutine test_elem_sub(x, c, i, z)
  real :: x(:)
  character(*) :: c(:)
  integer :: i
  complex :: z
  interface
    elemental subroutine foo(x, c, i, z)
      real, intent(out) :: x
      character(*), intent(inout) :: c
      integer, intent(in) :: i
      complex, value :: z
    end subroutine
  end interface

  call foo(x, c(10:1:-1), i, z)
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.complex<4> {adapt.valuebyref}
  ! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : i64
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_7:.*]] = arith.constant -1 : i64
  ! CHECK:         %[[VAL_8:.*]] = fir.slice %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.complex<4>>
  ! CHECK:         fir.store %[[VAL_9]] to %[[VAL_4]] : !fir.ref<!fir.complex<4>>
  ! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_10]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_11]]#1, %[[VAL_12]] : index
  ! CHECK:         fir.do_loop %[[VAL_15:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] {
  ! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : index
  ! CHECK:           %[[VAL_18:.*]] = fir.array_coor %[[VAL_0]] %[[VAL_17]] : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  ! CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_15]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_21:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_8]]] %[[VAL_20]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:           %[[VAL_22:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  ! CHECK:           %[[VAL_23:.*]] = fir.emboxchar %[[VAL_21]], %[[VAL_22]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:           fir.call @_QPfoo(%[[VAL_18]], %[[VAL_23]], %[[VAL_2]], %[[VAL_4]]) : (!fir.ref<f32>, !fir.boxchar<1>, !fir.ref<i32>, !fir.ref<!fir.complex<4>>) -> ()
  ! CHECK:         }
  ! CHECK:         return
  ! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QPtest_elem_sub_no_array_args(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<i32>,
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.ref<i32>) {
subroutine test_elem_sub_no_array_args(i, j)
  integer :: i, j
  interface
    elemental subroutine bar(i, j)
      integer, intent(out) :: i
      integer, intent(in) :: j
    end subroutine
  end interface
  call bar(i, j)
  ! CHECK:         fir.call @_QPbar(%[[VAL_0]], %[[VAL_1]]) : (!fir.ref<i32>, !fir.ref<i32>) -> ()
end subroutine
