! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPimplied_iters_allocatable(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>,
! CHECK-SAME %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine implied_iters_allocatable(thing, a1)
  ! No dependence between lhs and rhs.
  ! Lhs may need to be reallocated to conform.
  real :: a1(:)
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(:)
  integer :: i
  
  forall (i=5:13)
  ! commenting out this test for the moment
  !  thing(i)%arr = a1
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine implied_iters_allocatable

! CHECK-LABEL: func @_QPconflicting_allocatable(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFconflicting_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, %[[VAL_1:.*]]: !fir.ref<i32>, %[[VAL_2:.*]]: !fir.ref<i32>) {
subroutine conflicting_allocatable(thing, lo, hi)
  ! Introduce a crossing dependence to incite a (deep) copy.
  integer :: lo,hi
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(:)
  integer :: i
  
  forall (i = lo:hi)
  ! commenting out this test for the moment
  !  thing(i)%arr = thing(hi-i)%arr
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine conflicting_allocatable

! CHECK-LABEL: func @_QPforall_pointer_assign(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>,
! CHECK-SAME: %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>>> {fir.target},
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<i32>, %[[VAL_3:.*]]: !fir.ref<i32>) {
subroutine forall_pointer_assign(ap, at, ii, ij)
  ! Set pointer members in an array of derived type to targets.
  ! No conflicts (multiple-assignment being forbidden, of course).
  type t
     real, pointer :: ptr(:)
  end type t
  type u
     real :: targ(20)
  end type u
  type(t) :: ap(:)
  type(u), target :: at(:)
  integer :: ii, ij

  forall (i = ii:ij:8)
  ! commenting out this test for the moment
  !   ap(i)%ptr => at(i-4)%targ
  end forall
  ! CHECK: return
  ! CHECK: }  
end subroutine forall_pointer_assign

! CHECK-LABEL: func @_QPslice_with_explicit_iters() {
subroutine slice_with_explicit_iters
  ! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_1:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10xi32> {bindc_name = "a", uniq_name = "_QFslice_with_explicit_itersEa"}
  ! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.array<10x10xi32>
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
  ! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
  ! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:           %[[VAL_20:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
  ! CHECK:           %[[VAL_22:.*]] = subi %[[VAL_19]], %[[VAL_16]] : index
  ! CHECK:           %[[VAL_23:.*]] = addi %[[VAL_22]], %[[VAL_21]] : index
  ! CHECK:           %[[VAL_24:.*]] = divi_signed %[[VAL_23]], %[[VAL_21]] : index
  ! CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_26:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_27:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_28:.*]] = subi %[[VAL_24]], %[[VAL_26]] : index
  ! CHECK:           %[[VAL_29:.*]] = subi %[[VAL_2]], %[[VAL_26]] : index
  ! CHECK:           %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_27]] to %[[VAL_29]] step %[[VAL_26]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_10]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:             %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_26]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_32]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:               %[[VAL_36:.*]] = constant 0 : i32
  ! CHECK:               %[[VAL_37:.*]] = subi %[[VAL_36]], %[[VAL_25]] : i32
  ! CHECK:               %[[VAL_38:.*]] = constant 1 : i64
  ! CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
  ! CHECK:               %[[VAL_40:.*]] = constant 1 : i64
  ! CHECK:               %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i64) -> index
  ! CHECK:               %[[VAL_42:.*]] = muli %[[VAL_34]], %[[VAL_41]] : index
  ! CHECK:               %[[VAL_43:.*]] = addi %[[VAL_39]], %[[VAL_42]] : index
  ! CHECK:               %[[VAL_44:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
  ! CHECK:               %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i64) -> index
  ! CHECK:               %[[VAL_47:.*]] = fir.array_update %[[VAL_13]], %[[VAL_37]], %[[VAL_43]], %[[VAL_46]] {Fortran.offsets} : (!fir.array<10x10xi32>, i32, index, index) -> !fir.array<10x10xi32>
  ! CHECK:               fir.result %[[VAL_47]] : !fir.array<10x10xi32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_48:.*]] : !fir.array<10x10xi32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_49:.*]] : !fir.array<10x10xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_50:.*]] to %[[VAL_3]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10xi32>>

  integer :: a(10,10)
  forall (i=1:5)
     a(1:i, i) = -i
  end forall
  ! CHECK:         return
  ! CHECK:       }
end subroutine slice_with_explicit_iters
