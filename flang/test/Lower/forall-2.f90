! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPimplied_iters_allocatable(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine implied_iters_allocatable(a1)
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_2:.*]] = constant 20 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>> {bindc_name = "thing", uniq_name = "_QFimplied_iters_allocatableEthing"}
  ! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_3]](%[[VAL_4]]) : (!fir.ref<!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>
  ! CHECK:         %[[VAL_6:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_7:.*]] = constant {{.*}} : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_10:.*]] = fir.call @_FortranAInitialize(%[[VAL_8]], %[[VAL_9]], %[[VAL_7]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_11:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = constant 13 : i32
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
  ! CHECK:         %[[VAL_15:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_3]](%[[VAL_16]]) : (!fir.ref<!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>) -> !fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_15]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_17]]) -> (!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) {
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_21]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
  ! CHECK:           %[[VAL_24:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_25:.*]] = subi %[[VAL_23]], %[[VAL_24]] : i64
  ! CHECK:           %[[VAL_26:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_25]] : (!fir.ref<!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:           %[[VAL_27:.*]] = fir.field_index arr, !fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK:           %[[VAL_28:.*]] = fir.coordinate_of %[[VAL_26]], %[[VAL_27]] : (!fir.ref<!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:           %[[VAL_29:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK:           %[[VAL_30:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_31:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_30]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_28]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:           %[[VAL_33:.*]] = fir.box_addr %[[VAL_32]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (!fir.heap<!fir.array<?xf32>>) -> i64
  ! CHECK:           %[[VAL_35:.*]] = constant 0 : i64
  ! CHECK:           %[[VAL_36:.*]] = cmpi ne, %[[VAL_34]], %[[VAL_35]] : i64
  ! CHECK:           fir.if %[[VAL_36]] {
  ! CHECK:             %[[VAL_37:.*]] = constant false
  ! CHECK:             %[[VAL_38:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_32]], %[[VAL_38]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK:             %[[VAL_40:.*]] = cmpi ne, %[[VAL_39]]#1, %[[VAL_31]]#1 : index
  ! CHECK:             %[[VAL_41:.*]] = select %[[VAL_40]], %[[VAL_40]], %[[VAL_37]] : i1
  ! CHECK:             fir.if %[[VAL_41]] {
  ! CHECK:               fir.freemem %[[VAL_33]] : !fir.heap<!fir.array<?xf32>>
  ! CHECK:               %[[VAL_42:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_31]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:               %[[VAL_43:.*]] = fir.shape %[[VAL_31]]#1 : (index) -> !fir.shape<1>
  ! CHECK:               %[[VAL_44:.*]] = fir.embox %[[VAL_42]](%[[VAL_43]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:               fir.store %[[VAL_44]] to %[[VAL_28]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:             }
  ! CHECK:           } else {
  ! CHECK:             %[[VAL_45:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_31]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:             %[[VAL_46:.*]] = fir.shape %[[VAL_31]]#1 : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_47:.*]] = fir.embox %[[VAL_45]](%[[VAL_46]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:             fir.store %[[VAL_47]] to %[[VAL_28]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:           }
  ! CHECK:           %[[VAL_48:.*]] = fir.load %[[VAL_28]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:           %[[VAL_49:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_50:.*]]:3 = fir.box_dims %[[VAL_48]], %[[VAL_49]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK:           %[[VAL_51:.*]] = fir.box_addr %[[VAL_48]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK:           %[[VAL_52:.*]] = fir.shape_shift %[[VAL_50]]#0, %[[VAL_50]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:           %[[VAL_53:.*]] = fir.array_load %[[VAL_51]](%[[VAL_52]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK:           %[[VAL_54:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_55:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_56:.*]] = subi %[[VAL_31]]#1, %[[VAL_54]] : index
  ! CHECK:           %[[VAL_57:.*]] = fir.do_loop %[[VAL_58:.*]] = %[[VAL_55]] to %[[VAL_56]] step %[[VAL_54]] unordered iter_args(%[[VAL_59:.*]] = %[[VAL_53]]) -> (!fir.array<?xf32>) {
  ! CHECK:             %[[VAL_60:.*]] = fir.array_fetch %[[VAL_29]], %[[VAL_58]] : (!fir.array<?xf32>, index) -> f32
  ! CHECK:             %[[VAL_61:.*]] = fir.array_update %[[VAL_59]], %[[VAL_60]], %[[VAL_58]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:             fir.result %[[VAL_61]] : !fir.array<?xf32>
  ! CHECK:           }
  ! CHECK:           fir.array_merge_store %[[VAL_53]], %[[VAL_62:.*]] to %[[VAL_51]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
  ! CHECK:           fir.result %[[VAL_20]] : !fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_17]], %[[VAL_63:.*]] to %[[VAL_3]] : !fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<!fir.array<20x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>
  real :: a1(:)
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(20)
  forall (i=5:13)
    thing(i)%arr = a1
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine implied_iters_allocatable

! CHECK-LABEL: func @_QPforall_pointer_assign(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>>> {fir.target},
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<i32>, %[[VAL_3:.*]]: !fir.ref<i32>) {
subroutine forall_pointer_assign(ap, at, ii, ij)
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
  ! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = constant 8 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>) -> !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_10]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) {
  ! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_15]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
  ! CHECK:           %[[VAL_18:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_19:.*]] = subi %[[VAL_17]], %[[VAL_18]] : i64
  ! CHECK:           %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_19]] : (!fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, i64) -> !fir.ref<!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:           %[[VAL_21:.*]] = fir.field_index ptr, !fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK:           %[[VAL_22:.*]] = fir.coordinate_of %[[VAL_20]], %[[VAL_21]] : (!fir.ref<!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_24:.*]] = constant 4 : i32
  ! CHECK:           %[[VAL_25:.*]] = subi %[[VAL_23]], %[[VAL_24]] : i32
  ! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
  ! CHECK:           %[[VAL_27:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_28:.*]] = subi %[[VAL_26]], %[[VAL_27]] : i64
  ! CHECK:           %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_28]] : (!fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>>>, i64) -> !fir.ref<!fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>>
  ! CHECK:           %[[VAL_30:.*]] = fir.field_index targ, !fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>
  ! CHECK:           %[[VAL_31:.*]] = fir.coordinate_of %[[VAL_29]], %[[VAL_30]] : (!fir.ref<!fir.type<_QFforall_pointer_assignTu{targ:!fir.array<20xf32>}>>, !fir.field) -> !fir.ref<!fir.array<20xf32>>
  ! CHECK:           %[[VAL_32:.*]] = constant 20 : index
  ! CHECK:           %[[VAL_33:.*]] = fir.shape %[[VAL_32]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_34:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_35:.*]] = fir.slice %[[VAL_34]], %[[VAL_32]], %[[VAL_34]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:           %[[VAL_36:.*]] = fir.embox %[[VAL_31]](%[[VAL_33]]) {{\[}}%[[VAL_35]]] : (!fir.ref<!fir.array<20xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK:           %[[VAL_37:.*]] = fir.rebox %[[VAL_36]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK:           fir.store %[[VAL_37]] to %[[VAL_22]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:           fir.result %[[VAL_14]] : !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_38:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>

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
     ap(i)%ptr => at(i-4)%targ
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
  ! CHECK:           %[[VAL_15:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_16:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
  ! CHECK:           %[[VAL_19:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
  ! CHECK:           %[[VAL_22:.*]] = fir.undefined index
  ! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
  ! CHECK:           %[[VAL_24:.*]] = subi %[[VAL_23]], %[[VAL_15]] : index
  ! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:           %[[VAL_26:.*]] = fir.slice %[[VAL_16]], %[[VAL_18]], %[[VAL_19]], %[[VAL_21]], %[[VAL_22]], %[[VAL_22]] : (i64, i64, i64, i64, index, index) -> !fir.slice<2>
  ! CHECK:           %[[VAL_27:.*]] = fir.array_load %[[VAL_3]](%[[VAL_25]]) {{\[}}%[[VAL_26]]] : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x10xi32>
  ! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_29:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
  ! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
  ! CHECK:           %[[VAL_33:.*]] = subi %[[VAL_31]], %[[VAL_30]] : index
  ! CHECK:           %[[VAL_34:.*]] = addi %[[VAL_33]], %[[VAL_32]] : index
  ! CHECK:           %[[VAL_35:.*]] = divi_signed %[[VAL_34]], %[[VAL_32]] : index
  ! CHECK:           %[[VAL_36:.*]] = cmpi sgt, %[[VAL_35]], %[[VAL_29]] : index
  ! CHECK:           %[[VAL_37:.*]] = select %[[VAL_36]], %[[VAL_35]], %[[VAL_29]] : index
  ! CHECK:           %[[VAL_38:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_39:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_40:.*]] = subi %[[VAL_37]], %[[VAL_38]] : index
  ! CHECK:           %[[VAL_41:.*]] = fir.do_loop %[[VAL_42:.*]] = %[[VAL_39]] to %[[VAL_40]] step %[[VAL_38]] unordered iter_args(%[[VAL_43:.*]] = %[[VAL_27]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:             %[[VAL_44:.*]] = constant 0 : i32
  ! CHECK:             %[[VAL_45:.*]] = subi %[[VAL_44]], %[[VAL_28]] : i32
  ! CHECK:             %[[VAL_46:.*]] = fir.array_update %[[VAL_43]], %[[VAL_45]], %[[VAL_42]], %[[VAL_24]] : (!fir.array<10x10xi32>, i32, index, index) -> !fir.array<10x10xi32>
  ! CHECK:             fir.result %[[VAL_46]] : !fir.array<10x10xi32>
  ! CHECK:           }
  ! CHECK:           fir.array_merge_store %[[VAL_27]], %[[VAL_47:.*]] to %[[VAL_3]]{{\[}}%[[VAL_26]]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10xi32>>, !fir.slice<2>
  ! CHECK:           fir.result %[[VAL_13]] : !fir.array<10x10xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_48:.*]] to %[[VAL_3]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10xi32>>

  integer :: a(10,10)
  forall (i=1:5)
     a(1:i, i) = -i
  end forall
  ! CHECK:         return
  ! CHECK:       }
end subroutine slice_with_explicit_iters
