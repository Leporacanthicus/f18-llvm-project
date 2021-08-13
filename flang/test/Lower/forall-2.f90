! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPimplied_iters_allocatable(
subroutine implied_iters_allocatable(a1)
  real :: a1(:)
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(20)
  forall (i=5:13)
  !  thing(i)%arr = a1
  end forall
  ! CHECK: return
end subroutine implied_iters_allocatable

! CHECK-LABEL: func @_QPforall_pointer_assign(
subroutine forall_pointer_assign(ap, at, ii, ij)
  type t
     real, pointer :: ptr(:)
  end type t
  type u
     real :: targ(20)
  end type u
  type(t) :: ap(:)
  type(u) :: at(:)
  integer :: ii, ij

  forall (i = ii:ij:8)
  !  ap(i)%ptr => at(i-4)%targ
  end forall
  ! CHECK: return
end subroutine forall_pointer_assign

! CHECK-LABEL: func @_QPslice_with_explicit_iters(
subroutine slice_with_explicit_iters
  integer :: a(10,10)
  forall (i=1:5)
     a(1:i, i) = -i
  end forall
  ! CHECK-NEXT:    %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_1:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10xi32> {bindc_name = "a", uniq_name = "_QFslice_with_explicit_itersEa"}
  ! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_3]](%[[VAL_4]]) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.array<10x10xi32>
  ! CHECK:         %[[VAL_6:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_7:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_8:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_9:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = constant 1 : index

  ! CHECK-NEXT:   %[[V_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_13]] unordered iter_args(%[[VAL_2:.*]] = %[[VAL_5]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:           %[[V_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
  ! CHECK:           fir.store %[[V_3]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[V_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
  ! CHECK:           %[[V_6:.*]] = fir.convert %[[V_5]] : (i64) -> index
  ! CHECK:           %[[V_7:.*]] = constant 1 : i64
  ! CHECK:           %[[V_8:.*]] = fir.convert %[[V_7]] : (i64) -> index
  ! CHECK:           %[[VAL_9:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
  ! CHECK:           %[[VAL_11:.*]] = subi %[[V_6]], %[[V_8]] : index
  ! CHECK:           %[[VAL_12:.*]] = addi %[[VAL_11]], %[[VAL_10]] : index
  ! CHECK:           %[[VAL_13:.*]] = divi_signed %[[VAL_12]], %[[VAL_10]] : index
  ! CHECK:           %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_8]] to %[[VAL_13]] step %[[VAL_7]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_2]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:             %[[VAL_17:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_18:.*]] = constant 0 : i32
  ! CHECK:             %[[VAL_19:.*]] = subi %[[VAL_18]], %[[VAL_17]] : i32
  ! CHECK:             %[[VAL_20:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
  ! CHECK:             %[[VAL_22:.*]] = muli %[[VAL_21]], %[[VAL_15]] : index
  ! CHECK:             %[[VAL_23:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
  ! CHECK:             %[[VAL_25:.*]] = addi %[[VAL_22]], %[[VAL_24]] : index
  ! CHECK:             %[[VAL_26:.*]] = subi %[[VAL_25]], %[[VAL_6]] : index
  ! CHECK:             %[[VAL_27:.*]] = fir.array_update %[[VAL_16]], %[[VAL_19]], %[[VAL_26]], %[[VAL_15]] : (!fir.array<10x10xi32>, i32, index, index) -> !fir.array<10x10xi32>
  ! CHECK:             fir.result %[[VAL_27]] : !fir.array<10x10xi32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_14]] : !fir.array<10x10xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_5]], %[[V_0]] to %[[VAL_3]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10xi32>>
  ! CHECK:         return

end subroutine slice_with_explicit_iters
