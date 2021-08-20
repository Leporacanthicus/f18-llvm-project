! Test derived type assignment lowering inside array expression
! RUN: bbc %s -o - | FileCheck %s

module array_derived_assign
  type simple_copy
    integer :: i
    character(10) :: c(20)
    real, pointer :: p(:)
  end type
  type deep_copy
    integer :: i
    real, allocatable :: a(:)
  end type
contains

! Simple copies are implemented inline.
! CHECK-LABEL: func @_QMarray_derived_assignPtest_simple_copy(
! CHECK-SAME: %[[T1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>,
! CHECK-SAME: %[[T2:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>) {
subroutine test_simple_copy(t1, t2)
  type(simple_copy) :: t1(10), t2(10)
! CHECK-DAG:   %[[VAL_0:.*]] = constant 10 : index
! CHECK-DAG:   %[[VAL_1:.*]] = constant 0 : index
! CHECK-DAG:   %[[VAL_2:.*]] = constant 1 : index
! CHECK:   %[[VAL_3:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_1]], %[[VAL_0]] : index, index)
! CHECK: ^bb1(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
! CHECK:   %[[VAL_6:.*]] = cmpi sgt, %[[VAL_5]], %[[VAL_1]] : index
! CHECK:   cond_br %[[VAL_6]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_7:.*]] = addi %[[VAL_4]], %[[VAL_2]] : index
! CHECK:   %[[VAL_8:.*]] = fir.array_coor %[[T2:.*]](%[[VAL_3]]) %[[VAL_7]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{{.*}}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{{.*}}>>
! CHECK:   %[[VAL_10:.*]] = fir.array_coor %[[T1:.*]](%[[VAL_3]]) %[[VAL_7]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{{.*}}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{{.*}}>>
! CHECK:   %[[VAL_12:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{{.*}}>>
! CHECK:   fir.store %[[VAL_12]] to %[[VAL_10]] : !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{{.*}}>>
! CHECK:   %[[VAL_13:.*]] = subi %[[VAL_5]], %[[VAL_2]] : index
! CHECK:   br ^bb1(%[[VAL_7]], %[[VAL_13]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  t1 = t2
end subroutine

! Types require more complex assignments are passed to the runtime
! CHECK-LABEL: func @_QMarray_derived_assignPtest_deep_copy(
! CHECK-SAME: %[[T1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>,
! CHECK-SAME: %[[T2:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) {
subroutine test_deep_copy(t1, t2)
  type(deep_copy) :: t1(10), t2(10)
! CHECK-DAG:   %[[VAL_15:.*]] = constant 10 : index
! CHECK-DAG:   %[[VAL_16:.*]] = constant 0 : index
! CHECK-DAG:   %[[VAL_17:.*]] = constant 1 : index
! CHECK:   %[[VAL_18:.*]] = fir.alloca !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>> {uniq_name = ""}
! CHECK:   %[[VAL_19:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_16]], %[[VAL_15]] : index, index)
! CHECK: ^bb1(%[[VAL_20:.*]]: index, %[[VAL_21:.*]]: index):
! CHECK:   %[[VAL_22:.*]] = cmpi sgt, %[[VAL_21]], %[[VAL_16]] : index
! CHECK:   cond_br %[[VAL_22]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_23:.*]] = addi %[[VAL_20]], %[[VAL_17]] : index
! CHECK:   %[[VAL_24:.*]] = fir.array_coor %[[T2:.*]](%[[VAL_19]]) %[[VAL_23]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>
! CHECK:   %[[VAL_26:.*]] = fir.array_coor %[[T1:.*]](%[[VAL_19]]) %[[VAL_23]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>
! CHECK:   %[[VAL_28:.*]] = fir.embox %[[VAL_26]] : (!fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>) -> !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>
! CHECK:   %[[VAL_29:.*]] = fir.embox %[[VAL_24]] : (!fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>) -> !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>
! CHECK:   fir.store %[[VAL_28]] to %[[VAL_18]] : !fir.ref<!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>>
! CHECK:   %[[VAL_30:.*]] = fir.address_of({{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:   %[[VAL_31:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>>) -> !fir.ref<!fir.box<none>>
! CHECK:   %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{{.*}}>>) -> !fir.box<none>
! CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_34:.*]] = fir.call @_FortranAAssign(%[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:   %[[VAL_35:.*]] = subi %[[VAL_21]], %[[VAL_17]] : index
! CHECK:   br ^bb1(%[[VAL_23]], %[[VAL_35]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  t1 = t2
end subroutine
  
end module
