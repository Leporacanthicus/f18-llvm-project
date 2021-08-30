! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** This FORALL construct does present a potential loop-carried dependence if
!*** implemented naively (and incorrectly). The final value of a(3) must be the
!*** value of a(2) before loopy begins execution added to b(2).
! CHECK-LABEL: func @_QPtest9
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[n:[^:]*]]: !fir.ref<i32>)
subroutine test9(a,b,n)
  ! CHECK-DAG: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_1:.*]] = fir.load %[[n]] : !fir.ref<i32>
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (i32) -> i64
  ! CHECK: %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[n]] : !fir.ref<i32>
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
  ! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_9:.*]] = fir.array_load %[[a]](%[[VAL_8]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_11:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_12:.*]] = fir.array_load %[[a]](%[[VAL_11]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_13:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[b]](%[[VAL_13]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_17:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_19:.*]] = fir.load %[[n]] : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_20:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_21:.*]] = subi %[[VAL_19]], %[[VAL_20]] : i32
  ! CHECK-DAG: %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_16:.*]] = constant 1 : index
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  loopy: FORALL (i=1:n-1)
     ! CHECK: %[[V_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[VAL_18]] to %[[VAL_22]] step %[[VAL_16]] unordered iter_args(%[[VAL_2:.*]] = %[[VAL_9]]) -> (!fir.array<?xf32>) {
     ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
     ! CHECK: fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<i32>
     ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
     ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
     ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
     ! CHECK: %[[VAL_7:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_6]] : (!fir.array<?xf32>, index) -> f32
     ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
     ! CHECK: %[[V_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
     ! CHECK: %[[VAL_10:.*]] = fir.convert %[[V_9]] : (i64) -> index
     ! CHECK: %[[VAL_11:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_10]] : (!fir.array<?xf32>, index) -> f32
     ! CHECK: %[[VAL_12:.*]] = addf %[[VAL_7]], %[[VAL_11]] : f32
     ! CHECK-DAG: %[[VAL_13:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
     ! CHECK-DAG: %[[VAL_14:.*]] = constant 1 : i32
     ! CHECK: %[[VAL_15:.*]] = addi %[[VAL_13]], %[[VAL_14]] : i32
     ! CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
     ! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
     ! CHECK: %[[VAL_18:.*]] = fir.array_update %[[VAL_2]], %[[VAL_12]], %[[VAL_17]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
     ! CHECK: fir.result %[[VAL_18]] : !fir.array<?xf32>
     a(i+1) = a(i) + b(i)
     ! CHECK: }
     ! CHECK: fir.array_merge_store %[[VAL_9]], %[[V_0]] to %[[a]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>
  END FORALL loopy
  ! CHECK: return
end subroutine test9

!*** Test a FORALL statement
! CHECK-LABEL: func @_QPtest_forall_stmt(
! CHECK-SAME: %[[x:[^:]*]]: !fir.ref<!fir.array<200xf32>>,
! CHECK-SAME: %[[mask:[^:]*]]: !fir.ref<!fir.array<200x!fir.logical<4>>>)
subroutine test_forall_stmt(x, mask)
  ! CHECK-DAG: %[[VAL_20:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_21:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_22:.*]] = constant 200 : index
  ! CHECK: %[[VAL_23:.*]] = fir.shape %[[VAL_22]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_24:.*]] = fir.array_load %[[x]](%[[VAL_23]]) : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> !fir.array<200xf32>
  ! CHECK-DAG: %[[VAL_26:.*]] = constant 1.000000e+00 : f32
  ! CHECK-DAG: %[[VAL_27:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
  ! CHECK: %[[VAL_29:.*]] = constant 100 : i32
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
  ! CHECK: %[[VAL_31:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
  ! CHECK: %[[VAL_33:.*]] = subi %[[VAL_29]], %[[VAL_27]] : i32
  ! CHECK: %[[VAL_34:.*]] = addi %[[VAL_33]], %[[VAL_31]] : i32
  ! CHECK: %[[VAL_35:.*]] = divi_signed %[[VAL_34]], %[[VAL_31]] : i32
  ! CHECK: %[[VAL_36:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_37:.*]] = cmpi sgt, %[[VAL_35]], %[[VAL_36]] : i32
  ! CHECK: %[[VAL_38:.*]] = select %[[VAL_37]], %[[VAL_35]], %[[VAL_36]] : i32
  ! CHECK: %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> index
  ! CHECK: %[[VAL_40:.*]] = fir.shape %[[VAL_39]] : (index) -> !fir.shape<1>
  logical :: mask(200)
  real :: x(200)
  ! CHECK: %[[VAL_41:.*]] = fir.allocmem !fir.array<?x!fir.logical<4>>, %[[VAL_39]] {uniq_name = ".array.expr"}
  ! CHECK: %[[VAL_42:.*]] = fir.shape %[[VAL_39]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_43:.*]] = fir.array_load %[[VAL_41]](%[[VAL_42]]) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<?x!fir.logical<4>>
  ! CHECK: %[[VAL_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[VAL_28]] to %[[VAL_30]] step %[[VAL_32]] unordered iter_args(%[[VAL_2:.*]] = %[[VAL_43]]) -> (!fir.array<?x!fir.logical<4>>) {
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_3]] to %[[VAL_21]] : !fir.ref<i32>
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK: %[[VAL_6:.*]] = subi %[[VAL_5]], %[[VAL_28]] : index
  ! CHECK: %[[VAL_7:.*]] = divi_signed %[[VAL_6]], %[[VAL_32]] : index
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
  ! CHECK: %[[VAL_10:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_11:.*]] = subi %[[VAL_9]], %[[VAL_10]] : i64
  ! CHECK: %[[VAL_12:.*]] = fir.coordinate_of %[[mask]], %[[VAL_11]] : (!fir.ref<!fir.array<200x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_15:.*]] = fir.array_update %[[VAL_2]], %[[VAL_14]], %[[VAL_7]] : (!fir.array<?x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<?x!fir.logical<4>>
  ! CHECK: fir.result %[[VAL_15]] : !fir.array<?x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_43]], %[[VAL_0]] to %[[VAL_41]] : !fir.array<?x!fir.logical<4>>, !fir.array<?x!fir.logical<4>>, !fir.heap<!fir.array<?x!fir.logical<4>>>
  forall (i=1:100,mask(i)) x(i) = 1.
  ! CHECK: %[[VAL_17:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK: %[[VAL_19:.*]] = constant 100 : i32
  ! CHECK: %[[V_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> index
  ! CHECK: %[[VAL_21:.*]] = constant 1 : index
  ! CHECK: %[[VAL_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[VAL_18]] to %[[V_20]] step %[[VAL_21]] unordered iter_args(%[[VAL_2:.*]] = %[[VAL_24]]) -> (!fir.array<200xf32>) {
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_3]] to %[[VAL_20]] : !fir.ref<i32>
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK: %[[VAL_6:.*]] = subi %[[VAL_5]], %[[VAL_18]] : index
  ! CHECK: %[[VAL_7:.*]] = divi_signed %[[VAL_6]], %[[VAL_21]] : index
  ! CHECK: %[[VAL_8:.*]] = fir.array_coor %[[VAL_41]](%[[VAL_40]]) %[[VAL_7]] : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_11:.*]] = fir.if %[[VAL_10]] -> (!fir.array<200xf32>) {
  ! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
  ! CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
  ! CHECK: %[[VAL_15:.*]] = fir.array_update %[[VAL_2]], %[[VAL_26]], %[[VAL_14]] : (!fir.array<200xf32>, f32, index) -> !fir.array<200xf32>
  ! CHECK:             fir.result %[[VAL_15]] : !fir.array<200xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[VAL_2]] : !fir.array<200xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_16:.*]] : !fir.array<200xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_24]], %[[VAL_0]] to %[[x]] : !fir.array<200xf32>, !fir.array<200xf32>, !fir.ref<!fir.array<200xf32>>
  ! CHECK: fir.freemem %[[VAL_41]] : !fir.heap<!fir.array<?x!fir.logical<4>>>
  ! CHECK: return
end subroutine test_forall_stmt

!*** Test a FORALL construct
! CHECK-LABEL: func @_QPtest_forall_construct(
! CHECK-SAME: %[[a:[^:]*]]: !fir.box<!fir.array<?x?xf32>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.box<!fir.array<?x?xf32>>)
subroutine test_forall_construct(a,b)
  real :: a(:,:), b(:,:)
  ! CHECK-DAG: %[[VAL_18:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_19:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_20:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_21:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_22:.*]] = fir.array_load %[[a]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
  ! CHECK: %[[VAL_24:.*]] = fir.array_load %[[b]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
  ! CHECK: %[[VAL_26:.*]] = constant 3.140000e+00 : f32
  ! CHECK: %[[VAL_27:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_29:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_30:.*]]:3 = fir.box_dims %[[a]], %[[VAL_29]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_31:.*]] = fir.convert %[[VAL_30]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_32:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK-DAG: %[[VAL_34:.*]] = addi %[[VAL_31]], %[[VAL_33]] : i64
  ! CHECK-DAG: %[[VAL_35:.*]] = constant 1 : i64
  ! CHECK-DAG: %[[VAL_36:.*]] = subi %[[VAL_34]], %[[VAL_35]] : i64
  ! CHECK: %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i64) -> i32
  ! CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> index
  ! CHECK: %[[VAL_39:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> index
  ! CHECK: %[[VAL_41:.*]] = subi %[[VAL_37]], %[[VAL_27]] : i32
  ! CHECK: %[[VAL_42:.*]] = addi %[[VAL_41]], %[[VAL_39]] : i32
  ! CHECK: %[[VAL_43:.*]] = divi_signed %[[VAL_42]], %[[VAL_39]] : i32
  ! CHECK: %[[VAL_44:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_45:.*]] = cmpi sgt, %[[VAL_43]], %[[VAL_44]] : i32
  ! CHECK: %[[VAL_46:.*]] = select %[[VAL_45]], %[[VAL_43]], %[[VAL_44]] : i32
  ! CHECK: %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> index
  ! CHECK: %[[VAL_48:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_50:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_51:.*]]:3 = fir.box_dims %[[a]], %{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_52:.*]] = fir.convert %[[VAL_51]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_53:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_54:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK-DAG: %[[VAL_55:.*]] = addi %[[VAL_52]], %[[VAL_54]] : i64
  ! CHECK-DAG: %[[VAL_56:.*]] = constant 1 : i64
  ! CHECK-NEXT: %[[VAL_57:.*]] = subi %[[VAL_55]], %{{.*}} : i64
  ! CHECK: %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> i32
  ! CHECK: %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (i32) -> index
  ! CHECK: %[[VAL_60:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i32) -> index
  ! CHECK: %[[VAL_62:.*]] = subi %[[VAL_58]], %[[VAL_48]] : i32
  ! CHECK: %[[VAL_63:.*]] = addi %[[VAL_62]], %[[VAL_60]] : i32
  ! CHECK: %[[VAL_64:.*]] = divi_signed %[[VAL_63]], %[[VAL_60]] : i32
  ! CHECK: %[[VAL_65:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_66:.*]] = cmpi sgt, %[[VAL_64]], %[[VAL_65]] : i32
  ! CHECK: %[[VAL_67:.*]] = select %[[VAL_66]], %[[VAL_64]], %[[VAL_65]] : i32
  ! CHECK: %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (i32) -> index
  ! CHECK: %[[VAL_69:.*]] = fir.shape %[[VAL_47]], %[[VAL_68]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_70:.*]] = fir.allocmem !fir.array<?x?x!fir.logical<4>>, %[[VAL_47]], %[[VAL_68]] {uniq_name = ".array.expr"}
  ! CHECK: %[[VAL_71:.*]] = fir.shape %[[VAL_47]], %[[VAL_68]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_72:.*]] = fir.array_load %[[VAL_70]](%[[VAL_71]]) : (!fir.heap<!fir.array<?x?x!fir.logical<4>>>, !fir.shape<2>) -> !fir.array<?x?x!fir.logical<4>>
  ! CHECK: %[[VAL_0:.*]] = fir.do_loop %[[VAL_1:.*]] = %[[VAL_28]] to %[[VAL_38]] step %[[VAL_40]] unordered iter_args(%[[VAL_2:.*]] = %[[VAL_72]]) -> (!fir.array<?x?x!fir.logical<4>>) {
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_3]] to %[[VAL_21]] : !fir.ref<i32>
  ! CHECK: %[[VAL_4:.*]] = fir.do_loop %[[VAL_5:.*]] = %[[VAL_49]] to %[[VAL_59]] step %[[VAL_61]] unordered iter_args(%[[VAL_6:.*]] = %[[VAL_2]]) -> (!fir.array<?x?x!fir.logical<4>>) {
  ! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_7]] to %[[VAL_20]] : !fir.ref<i32>
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
  ! CHECK: %[[VAL_10:.*]] = subi %[[VAL_9]], %[[VAL_28]] : index
  ! CHECK: %[[VAL_11:.*]] = divi_signed %[[VAL_10]], %[[VAL_40]] : index
  ! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
  ! CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
  ! CHECK: %[[VAL_14:.*]] = subi %[[VAL_13]], %[[VAL_49]] : index
  ! CHECK: %[[VAL_15:.*]] = divi_signed %[[VAL_14]], %[[VAL_61]] : index
  ! CHECK-DAG: %[[VAL_16:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
  ! CHECK-DAG: %[[V_18:.*]] = constant 1 : i64
  ! CHECK-DAG: %[[V_19:.*]] = subi %[[VAL_17]], %[[V_18]] : i64
  ! CHECK-DAG: %[[VAL_20:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
  ! CHECK-DAG: %[[V_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
  ! CHECK-DAG: %[[V_22:.*]] = constant 1 : i64
  ! CHECK-DAG: %[[VAL_23:.*]] = subi %[[V_21]], %[[V_22]] : i64
  ! CHECK-DAG: %[[V_24:.*]] = fir.coordinate_of %[[b]], %[[V_19]], %[[VAL_23]] : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
  ! CHECK-DAG: %[[VAL_25:.*]] = fir.load %[[V_24]] : !fir.ref<f32>
  ! CHECK-DAG: %[[V_26:.*]] = constant 0.000000e+00 : f32
  ! CHECK: %[[VAL_27:.*]] = cmpf ogt, %[[VAL_25]], %[[V_26]] : f32
  ! CHECK: %[[V_28:.*]] = fir.convert %[[VAL_27]] : (i1) -> !fir.logical<4>
  ! CHECK: %[[VAL_29:.*]] = fir.array_update %[[VAL_6]], %[[V_28]], %[[VAL_11]], %[[VAL_15]] : (!fir.array<?x?x!fir.logical<4>>, !fir.logical<4>, index, index) -> !fir.array<?x?x!fir.logical<4>>
  ! CHECK: fir.result %[[VAL_29]] : !fir.array<?x?x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_4]] : !fir.array<?x?x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_72]], %[[VAL_0]] to %[[VAL_70]] : !fir.array<?x?x!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>, !fir.heap<!fir.array<?x?x!fir.logical<4>>>
  ! CHECK: %[[VAL_32:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_34:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_35:.*]]:3 = fir.box_dims %[[a]], %[[VAL_34]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_36:.*]] = fir.convert %[[VAL_35]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_37:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (index) -> i64
  ! CHECK-DAG: %[[VAL_39:.*]] = addi %[[VAL_36]], %[[VAL_38]] : i64
  ! CHECK-DAG: %[[VAL_40:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_41:.*]] = subi %[[VAL_39]], %[[VAL_40]] : i64
  ! CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i64) -> i32
  ! CHECK: %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i32) -> index
  ! CHECK: %[[VAL_44:.*]] = constant 1 : index
  forall (i=1:ubound(a,1), j=1:ubound(a,2), b(j,i) > 0.0)
  ! CHECK: %[[WAL_0:.*]] = fir.do_loop %[[WAL_1:.*]] = %[[VAL_33]] to %[[VAL_43]] step %[[VAL_44]] unordered iter_args(%[[WAL_2:.*]] = %[[VAL_22]]) -> (!fir.array<?x?xf32>) {
  ! CHECK: %[[WAL_3:.*]] = fir.convert %[[WAL_1]] : (index) -> i32
  ! CHECK: fir.store %[[WAL_3]] to %[[VAL_19]] : !fir.ref<i32>
  ! CHECK: %[[WAL_8:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
  ! CHECK: %[[WAL_9:.*]] = fir.convert %[[WAL_8]] : (i32) -> index
  ! CHECK: %[[WAL_10:.*]] = subi %[[WAL_9]], %[[VAL_33]] : index
  ! CHECK: %[[WAL_11:.*]] = divi_signed %[[WAL_10]], %[[VAL_44]] : index
  ! CHECK: %[[VAL_45:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_47:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_48:.*]]:3 = fir.box_dims %[[a]], %{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_49:.*]] = fir.convert %[[VAL_48]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_50:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_51:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK-DAG: %[[VAL_52:.*]] = addi %[[VAL_49]], %[[VAL_51]] : i64
  ! CHECK-DAG: %[[VAL_53:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_54:.*]] = subi %[[VAL_52]], %[[VAL_53]] : i64
  ! CHECK: %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i64) -> i32
  ! CHECK: %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i32) -> index
  ! CHECK: %[[VAL_57:.*]] = constant 1 : index
  ! CHECK: %[[WAL_4:.*]] = fir.do_loop %[[WAL_5:.*]] = %[[VAL_46]] to %[[VAL_56]] step %[[VAL_57]] unordered iter_args(%[[WAL_6:.*]] = %[[VAL_2]]) -> (!fir.array<?x?xf32>) {
  ! CHECK: %[[WAL_7:.*]] = fir.convert %[[WAL_5]] : (index) -> i32
  ! CHECK: fir.store %[[WAL_7]] to %[[VAL_18]] : !fir.ref<i32>
  ! CHECK: %[[WAL_12:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
  ! CHECK: %[[WAL_13:.*]] = fir.convert %[[WAL_12]] : (i32) -> index
  ! CHECK: %[[WAL_14:.*]] = subi %[[WAL_13]], %[[VAL_46]] : index
  ! CHECK: %[[WAL_15:.*]] = divi_signed %[[WAL_14]], %[[VAL_57]] : index
  ! CHECK: %[[WAL_16:.*]] = fir.array_coor %[[VAL_70]](%[[VAL_69]]) %[[WAL_11]], %[[WAL_15]] : (!fir.heap<!fir.array<?x?x!fir.logical<4>>>, !fir.shape<2>, index, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[WAL_17:.*]] = fir.load %[[WAL_16]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[WAL_18:.*]] = fir.convert %[[WAL_17]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[WAL_19:.*]] = fir.if %[[WAL_18]] -> (!fir.array<?x?xf32>) {
  ! CHECK-NEXT: %[[WAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
  ! CHECK: %[[WAL_21:.*]] = fir.convert %[[WAL_20]] : (i32) -> i64
  ! CHECK: %[[WAL_22:.*]] = fir.convert %[[WAL_21]] : (i64) -> index
  ! CHECK: %[[WAL_23:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
  ! CHECK: %[[WAL_24:.*]] = fir.convert %[[WAL_23]] : (i32) -> i64
  ! CHECK: %[[WAL_25:.*]] = fir.convert %[[WAL_24]] : (i64) -> index
  ! CHECK-NEXT: %[[WAL_26:.*]] = fir.array_fetch %[[VAL_24]], %[[WAL_25]], %[[WAL_22]] : (!fir.array<?x?xf32>, index, index) -> f32
  ! CHECK: %[[WAL_27:.*]] = divf %[[WAL_26]], %[[VAL_26]] : f32
  ! CHECK: %[[WAL_28:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
  ! CHECK: %[[WAL_29:.*]] = fir.convert %[[WAL_28]] : (i32) -> i64
  ! CHECK: %[[WAL_30:.*]] = fir.convert %[[WAL_29]] : (i64) -> index
  ! CHECK: %[[WAL_31:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
  ! CHECK: %[[WAL_32:.*]] = fir.convert %[[WAL_31]] : (i32) -> i64
  ! CHECK: %[[WAL_33:.*]] = fir.convert %[[WAL_32]] : (i64) -> index
  ! CHECK-NEXT: %[[WAL_34:.*]] = fir.array_update %[[WAL_6]], %[[WAL_27]], %[[WAL_33]], %[[WAL_30]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  ! CHECK: fir.result %[[WAL_34]] : !fir.array<?x?xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[WAL_6]] : !fir.array<?x?xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[WAL_19]] : !fir.array<?x?xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[WAL_4]] : !fir.array<?x?xf32>
     a(i,j) = b(j,i) / 3.14
  end forall
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_22]], %[[WAL_0]] to %[[a]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  ! CHECK: fir.freemem %[[VAL_70]] : !fir.heap<!fir.array<?x?x!fir.logical<4>>>
  ! CHECK: return
end subroutine test_forall_construct

!*** Test forall with multiple assignment statements
! CHECK-LABEL: func @_QPtest2_forall_construct(
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<100x400xf32>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.ref<!fir.array<200x200xf32>>) {
subroutine test2_forall_construct(a,b)
  ! CHECK-DAG: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_4:.*]] = constant 100 : index
  ! CHECK-DAG: %[[VAL_5:.*]] = constant 400 : index
  ! CHECK-DAG: %[[VAL_6:.*]] = constant 200 : index
  ! CHECK-DAG: %[[VAL_7:.*]] = constant 200 : index
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_9:.*]] = fir.array_load %[[a]](%[[VAL_8]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK-NEXT: %[[VAL_11:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_12:.*]] = fir.array_load %[[b]](%[[VAL_11]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK-NEXT: %[[VAL_14:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_15:.*]] = fir.array_load %[[b]](%[[VAL_14]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK: %[[VAL_16:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
  ! CHECK: %[[VAL_18:.*]] = constant 100 : i32
  ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK: %[[VAL_20:.*]] = constant 1 : index
  real :: a(100,400), b(200,200)
  forall (i=1:100, j=1:200)
  ! CHECK: %[[V_0:.*]] = fir.do_loop %[[V_1:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_20]] unordered iter_args(%[[V_2:.*]] = %[[VAL_9]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[V_3:.*]] = fir.convert %[[V_1]] : (index) -> i32
  ! CHECK: fir.store %[[V_3]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_21:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> index
  ! CHECK: %[[VAL_23:.*]] = constant 200 : i32
  ! CHECK: %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> index
  ! CHECK: %[[VAL_25:.*]] = constant 1 : index
  ! CHECK: %[[V_4:.*]] = fir.do_loop %[[V_5:.*]] = %[[VAL_22]] to %[[VAL_24]] step %[[VAL_25]] unordered iter_args(%[[V_6:.*]] = %[[V_2]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[V_7:.*]] = fir.convert %[[V_5]] : (index) -> i32
  ! CHECK: fir.store %[[V_7]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[V_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
  ! CHECK: %[[V_10:.*]] = fir.convert %[[V_9]] : (i64) -> index
  ! CHECK: %[[VAL_11:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[V_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
  ! CHECK: %[[VAL_13:.*]] = fir.convert %[[V_12]] : (i64) -> index
  ! CHECK-NEXT: %[[VAL_14:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_13]], %[[V_10]] : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK: %[[V_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[W_16:.*]] = fir.convert %[[V_15]] : (i32) -> i64   
  ! CHECK: %[[W_17:.*]] = fir.convert %[[W_16]] : (i64) -> index   
  ! CHECK-DAG: %[[VAL_20:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_16:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_17:.*]] = addi %[[VAL_20]], %[[VAL_16]] : i32
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
  ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK-NEXT: %[[VAL_23:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_19]], %[[W_17]] : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK: %[[VAL_24:.*]] = addf %[[VAL_14]], %[[VAL_23]] : f32
  ! CHECK: %[[VAL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
  ! CHECK: %[[VAL_25:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
  ! CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
  ! CHECK: %[[VAL_31:.*]] = fir.array_update %[[V_6]], %[[VAL_24]], %[[VAL_27]], %[[VAL_30]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK: fir.result %[[VAL_31]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_32:.*]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_9]], %[[V_0]] to %[[a]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
     a(i,j) = b(i,j) + b(i+1,j)
  ! CHECK: %[[VAL_34:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_35:.*]] = fir.array_load %[[a]](%[[VAL_34]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK: %[[VAL_36:.*]] = constant 1.000000e+00 : f32
  ! CHECK: %[[VAL_37:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_38:.*]] = fir.array_load %[[b]](%[[VAL_37]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK: %[[VAL_39:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> index
  ! CHECK: %[[VAL_41:.*]] = constant 100 : i32
  ! CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
  ! CHECK: %[[VAL_43:.*]] = constant 1 : index
  ! CHECK: %[[V_0:.*]] = fir.do_loop %[[V_1:.*]] = %[[VAL_40]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_2:.*]] = %[[VAL_35]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[V_3:.*]] = fir.convert %[[V_1]] : (index) -> i32
  ! CHECK: fir.store %[[V_3]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[VAL_44:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> index
  ! CHECK: %[[VAL_46:.*]] = constant 200 : i32
  ! CHECK: %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> index
  ! CHECK: %[[VAL_48:.*]] = constant 1 : index
  ! CHECK: %[[VAL_4:.*]] = fir.do_loop %[[VAL_5:.*]] = %[[VAL_45]] to %[[VAL_47]] step %[[VAL_48]] unordered iter_args(%[[VAL_6:.*]] = %[[VAL_2]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_7]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
  ! CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
  ! CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
  ! CHECK-NEXT: %[[VAL_14:.*]] = fir.array_fetch %[[VAL_38]], %[[VAL_10]], %[[VAL_13]] : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK: %[[VAL_15:.*]] = divf %[[VAL_36]], %[[VAL_14]] : f32
  ! CHECK-DAG: %[[VAL_19:.*]] = constant 200 : i32
  ! CHECK-DAG: %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[VAL_21:.*]] = addi %[[VAL_19]], %[[VAL_20]] : i32
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
  ! CHECK: %[[VAL_16:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
  ! CHECK-NEXT: %[[VAL_24:.*]] = fir.array_update %[[VAL_6]], %[[VAL_15]], %[[VAL_18]], %[[VAL_23]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK: fir.result %[[VAL_24]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_25:.*]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_35]], %[[V_0]] to %[[a]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
     a(i,200+j) = 1.0 / b(j, i)
  end forall
  ! CHECK: return
end subroutine test2_forall_construct

!*** Test forall with multiple assignment statements and mask
! CHECK-LABEL: func @_QPtest3_forall_construct(
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<100x400xf32>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.ref<!fir.array<200x200xf32>>,
! CHECK-SAME: %[[mask:[^:]*]]: !fir.ref<!fir.array<100x200x!fir.logical<4>>>) {
subroutine test3_forall_construct(a,b, mask)
  real :: a(100,400), b(200,200)
  logical :: mask(100,200)
  ! CHECK-DAG: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK-DAG: %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_6:.*]] = constant 100 : index
  ! CHECK-DAG: %[[VAL_7:.*]] = constant 400 : index
  ! CHECK-DAG: %[[VAL_8:.*]] = constant 200 : index
  ! CHECK-DAG: %[[VAL_9:.*]] = constant 200 : index
  ! CHECK: %[[VAL_10:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_11:.*]] = fir.array_load %[[a]](%[[VAL_10]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK: %[[VAL_13:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[b]](%[[VAL_13]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK: %[[VAL_16:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_17:.*]] = fir.array_load %[[b]](%[[VAL_16]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK: %[[VAL_18:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK: %[[VAL_20:.*]] = constant 100 : i32
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK: %[[VAL_22:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> index
  ! CHECK: %[[VAL_24:.*]] = subi %[[VAL_20]], %[[VAL_18]] : i32
  ! CHECK: %[[VAL_25:.*]] = addi %[[VAL_24]], %[[VAL_22]] : i32
  ! CHECK: %[[VAL_26:.*]] = divi_signed %[[VAL_25]], %[[VAL_22]] : i32
  ! CHECK: %[[VAL_27:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_28:.*]] = cmpi sgt, %[[VAL_26]], %[[VAL_27]] : i32
  ! CHECK: %[[VAL_29:.*]] = select %[[VAL_28]], %[[VAL_26]], %[[VAL_27]] : i32
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
  ! CHECK: %[[VAL_31:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
  ! CHECK: %[[VAL_33:.*]] = constant 200 : i32
  ! CHECK: %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
  ! CHECK: %[[VAL_35:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
  ! CHECK: %[[VAL_37:.*]] = subi %[[VAL_33]], %[[VAL_31]] : i32
  ! CHECK: %[[VAL_38:.*]] = addi %[[VAL_37]], %[[VAL_35]] : i32
  ! CHECK: %[[VAL_39:.*]] = divi_signed %[[VAL_38]], %[[VAL_35]] : i32
  ! CHECK: %[[VAL_40:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_41:.*]] = cmpi sgt, %[[VAL_39]], %[[VAL_40]] : i32
  ! CHECK: %[[VAL_42:.*]] = select %[[VAL_41]], %[[VAL_39]], %[[VAL_40]] : i32
  ! CHECK: %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i32) -> index
  ! CHECK: %[[VAL_44:.*]] = fir.shape %[[VAL_30]], %[[VAL_43]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_45:.*]] = fir.allocmem !fir.array<?x?x!fir.logical<4>>, %[[VAL_30]], %[[VAL_43]] {uniq_name = ".array.expr"}
  ! CHECK: %[[VAL_46:.*]] = fir.shape %[[VAL_30]], %[[VAL_43]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_47:.*]] = fir.array_load %[[VAL_45]](%[[VAL_46]]) : (!fir.heap<!fir.array<?x?x!fir.logical<4>>>, !fir.shape<2>) -> !fir.array<?x?x!fir.logical<4>>
  forall (i=1:100, j=1:200, mask(i,j))
  ! CHECK-NEXT: %[[AL_0:.*]] = fir.do_loop %[[AL_1:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_23]] unordered iter_args(%[[AL_2:.*]] = %[[VAL_47]]) -> (!fir.array<?x?x!fir.logical<4>>) {
  ! CHECK: %[[AL_3:.*]] = fir.convert %[[AL_1]] : (index) -> i32
  ! CHECK: fir.store %[[AL_3]] to %[[VAL_5]] : !fir.ref<i32>
  ! CHECK-NEXT: %[[AL_4:.*]] = fir.do_loop %[[AL_5:.*]] = %[[VAL_32]] to %[[VAL_34]] step %[[VAL_36]] unordered iter_args(%[[AL_6:.*]] = %[[AL_2]]) -> (!fir.array<?x?x!fir.logical<4>>) {
  ! CHECK-NEXT: %[[AL_7:.*]] = fir.convert %[[AL_5]] : (index) -> i32
  ! CHECK: fir.store %[[AL_7]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[AL_8:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[AL_9:.*]] = fir.convert %[[AL_8]] : (i32) -> index
  ! CHECK: %[[AL_10:.*]] = subi %[[AL_9]], %[[VAL_19]] : index
  ! CHECK: %[[AL_11:.*]] = divi_signed %[[AL_10]], %[[VAL_23]] : index
  ! CHECK: %[[AL_12:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[AL_13:.*]] = fir.convert %[[AL_12]] : (i32) -> index
  ! CHECK: %[[AL_14:.*]] = subi %[[AL_13]], %[[VAL_32]] : index
  ! CHECK: %[[AL_15:.*]] = divi_signed %[[AL_14]], %[[VAL_36]] : index
  ! CHECK: %[[AL_16:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[AL_17:.*]] = fir.convert %[[AL_16]] : (i32) -> i64
  ! CHECK: %[[AL_18:.*]] = constant 1 : i64
  ! CHECK: %[[AL_19:.*]] = subi %[[AL_17]], %[[AL_18]] : i64
  ! CHECK: %[[AL_20:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[AL_21:.*]] = fir.convert %[[AL_20]] : (i32) -> i64
  ! CHECK: %[[AL_22:.*]] = constant 1 : i64
  ! CHECK: %[[AL_23:.*]] = subi %[[AL_21]], %[[AL_22]] : i64
  ! CHECK: %[[AL_24:.*]] = fir.coordinate_of %[[mask]], %[[AL_19]], %[[AL_23]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[AL_26:.*]] = fir.load %[[AL_24]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[AL_27:.*]] = fir.array_update %[[AL_6]], %[[AL_26]], %[[AL_11]], %[[AL_15]] : (!fir.array<?x?x!fir.logical<4>>, !fir.logical<4>, index, index) -> !fir.array<?x?x!fir.logical<4>>
  ! CHECK: fir.result %[[AL_27]] : !fir.array<?x?x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_4]] : !fir.array<?x?x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_47]], %[[AL_0]] to %[[VAL_45]] : !fir.array<?x?x!fir.logical<4>>, !fir.array<?x?x!fir.logical<4>>, !fir.heap<!fir.array<?x?x!fir.logical<4>>>
     
  ! CHECK: %[[VAL_30:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> index
  ! CHECK: %[[VAL_32:.*]] = constant 100 : i32
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> index
  ! CHECK: %[[VAL_34:.*]] = constant 1 : index

  ! CHECK-NEXT: %[[AL_0:.*]] = fir.do_loop %[[AL_1:.*]] = %[[VAL_31]] to %[[VAL_33]] step %[[VAL_34]] unordered iter_args(%[[AL_2:.*]] = %[[VAL_11]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[AL_3:.*]] = fir.convert %[[AL_1]] : (index) -> i32
  ! CHECK: fir.store %[[AL_3]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[AL_8:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[AL_9:.*]] = fir.convert %[[AL_8]] : (i32) -> index
  ! CHECK: %[[AL_10:.*]] = subi %[[AL_9]], %[[VAL_31]] : index
  ! CHECK: %[[AL_11:.*]] = divi_signed %[[AL_10]], %[[VAL_34]] : index
  ! CHECK: %[[VAL_35:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
  ! CHECK: %[[VAL_37:.*]] = constant 200 : i32
  ! CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> index
  ! CHECK: %[[VAL_39:.*]] = constant 1 : index
  ! CHECK-NEXT: %[[AL_4:.*]] = fir.do_loop %[[AL_5:.*]] = %[[VAL_36]] to %[[VAL_38]] step %[[VAL_39]] unordered iter_args(%[[AL_6:.*]] = %[[AL_2]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[AL_7:.*]] = fir.convert %[[AL_5]] : (index) -> i32
  ! CHECK: fir.store %[[AL_7]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[AL_12:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[AL_13:.*]] = fir.convert %[[AL_12]] : (i32) -> index
  ! CHECK: %[[AL_14:.*]] = subi %[[AL_13]], %[[VAL_36]] : index
  ! CHECK: %[[AL_15:.*]] = divi_signed %[[AL_14]], %[[VAL_39]] : index
  ! CHECK-NEXT: %[[AL_16:.*]] = fir.array_coor %[[VAL_45]](%[[VAL_44]]) %[[AL_11]], %[[AL_15]] : (!fir.heap<!fir.array<?x?x!fir.logical<4>>>, !fir.shape<2>, index, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[AL_17:.*]] = fir.load %[[AL_16]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[AL_18:.*]] = fir.convert %[[AL_17]] : (!fir.logical<4>) -> i1
  ! CHECK-NEXT: %[[AL_19:.*]] = fir.if %[[AL_18]] -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[AL_23:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[AL_24:.*]] = fir.convert %[[AL_23]] : (i32) -> i64
  ! CHECK: %[[AL_25:.*]] = fir.convert %[[AL_24]] : (i64) -> index
  ! CHECK: %[[AL_20:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[AL_21:.*]] = fir.convert %[[AL_20]] : (i32) -> i64
  ! CHECK: %[[AL_22:.*]] = fir.convert %[[AL_21]] : (i64) -> index
  ! CHECK: %[[AL_26:.*]] = fir.array_fetch %[[VAL_14]], %[[AL_22]], %[[AL_25]] : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK: %[[AL_32:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[AL_33:.*]] = fir.convert %[[AL_32]] : (i32) -> i64
  ! CHECK: %[[AL_34:.*]] = fir.convert %[[AL_33]] : (i64) -> index
  ! CHECK-DAG: %[[AL_27:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK-DAG: %[[AL_28:.*]] = constant 1 : i32
  ! CHECK: %[[AL_29:.*]] = addi %[[AL_27]], %[[AL_28]] : i32
  ! CHECK: %[[AL_30:.*]] = fir.convert %[[AL_29]] : (i32) -> i64
  ! CHECK: %[[AL_31:.*]] = fir.convert %[[AL_30]] : (i64) -> index
  ! CHECK: %[[AL_35:.*]] = fir.array_fetch %[[VAL_17]], %[[AL_31]], %[[AL_34]] : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK: %[[AL_36:.*]] = addf %[[AL_26]], %[[AL_35]] : f32
  ! CHECK: %[[AL_40:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[AL_41:.*]] = fir.convert %[[AL_40]] : (i32) -> i64
  ! CHECK: %[[AL_42:.*]] = fir.convert %[[AL_41]] : (i64) -> index
  ! CHECK: %[[AL_37:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[AL_38:.*]] = fir.convert %[[AL_37]] : (i32) -> i64
  ! CHECK: %[[AL_39:.*]] = fir.convert %[[AL_38]] : (i64) -> index
  ! CHECK-NEXT: %[[AL_43:.*]] = fir.array_update %[[AL_6]], %[[AL_36]], %[[AL_39]], %[[AL_42]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK: fir.result %[[AL_43]] : !fir.array<100x400xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[AL_6]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_19]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_4]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_11]], %[[AL_0]] to %[[a]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>

     a(i,j) = b(i,j) + b(i+1,j)

  ! CHECK: %[[VAL_47:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_48:.*]] = fir.array_load %[[a]](%[[VAL_47]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK: %[[VAL_49:.*]] = constant 1.000000e+00 : f32
  ! CHECK: %[[VAL_50:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_51:.*]] = fir.array_load %[[b]](%[[VAL_50]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK: %[[VAL_52:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> index
  ! CHECK: %[[VAL_54:.*]] = constant 100 : i32
  ! CHECK: %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i32) -> index
  ! CHECK: %[[VAL_56:.*]] = constant 1 : index

  ! CHECK: %[[AL_0:.*]] = fir.do_loop %[[AL_1:.*]] = %[[VAL_53]] to %[[VAL_55]] step %[[VAL_56]] unordered iter_args(%[[AL_2:.*]] = %[[VAL_48]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[AL_3:.*]] = fir.convert %[[AL_1]] : (index) -> i32
  ! CHECK: fir.store %[[AL_3]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[AL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[AL_9:.*]] = fir.convert %[[AL_8]] : (i32) -> index
  ! CHECK: %[[AL_10:.*]] = subi %[[AL_9]], %[[VAL_53]] : index
  ! CHECK: %[[AL_11:.*]] = divi_signed %[[AL_10]], %[[VAL_56]] : index
  ! CHECK: %[[VAL_57:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i32) -> index
  ! CHECK: %[[VAL_59:.*]] = constant 200 : i32
  ! CHECK: %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> index
  ! CHECK: %[[VAL_61:.*]] = constant 1 : index
  ! CHECK: %[[AL_4:.*]] = fir.do_loop %[[AL_5:.*]] = %[[VAL_58]] to %[[VAL_60]] step %[[VAL_61]] unordered iter_args(%[[AL_6:.*]] = %[[AL_2]]) -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[AL_7:.*]] = fir.convert %[[AL_5]] : (index) -> i32
  ! CHECK: fir.store %[[AL_7]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[AL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[AL_13:.*]] = fir.convert %[[AL_12]] : (i32) -> index
  ! CHECK: %[[AL_14:.*]] = subi %[[AL_13]], %[[VAL_58]] : index
  ! CHECK: %[[AL_15:.*]] = divi_signed %[[AL_14]], %[[VAL_61]] : index
  ! CHECK: %[[AL_16:.*]] = fir.array_coor %[[VAL_45]](%[[VAL_44]]) %[[AL_11]], %[[AL_15]] : (!fir.heap<!fir.array<?x?x!fir.logical<4>>>, !fir.shape<2>, index, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[AL_17:.*]] = fir.load %[[AL_16]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[AL_18:.*]] = fir.convert %[[AL_17]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[AL_19:.*]] = fir.if %[[AL_18]] -> (!fir.array<100x400xf32>) {
  ! CHECK: %[[AL_23:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[AL_24:.*]] = fir.convert %[[AL_23]] : (i32) -> i64
  ! CHECK: %[[AL_25:.*]] = fir.convert %[[AL_24]] : (i64) -> index
  ! CHECK: %[[AL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[AL_21:.*]] = fir.convert %[[AL_20]] : (i32) -> i64
  ! CHECK: %[[AL_22:.*]] = fir.convert %[[AL_21]] : (i64) -> index
  ! CHECK: %[[AL_26:.*]] = fir.array_fetch %[[VAL_51]], %[[AL_22]], %[[AL_25]] : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK-NEXT: %[[AL_27:.*]] = divf %[[VAL_49]], %[[AL_26]] : f32
  ! CHECK-DAG: %[[AL_31:.*]] = constant 200 : i32
  ! CHECK-DAG: %[[AL_32:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[AL_33:.*]] = addi %[[AL_31]], %[[AL_32]] : i32
  ! CHECK: %[[AL_34:.*]] = fir.convert %[[AL_33]] : (i32) -> i64
  ! CHECK: %[[AL_35:.*]] = fir.convert %[[AL_34]] : (i64) -> index
  ! CHECK: %[[AL_28:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[AL_29:.*]] = fir.convert %[[AL_28]] : (i32) -> i64
  ! CHECK: %[[AL_30:.*]] = fir.convert %[[AL_29]] : (i64) -> index
  ! CHECK: %[[AL_36:.*]] = fir.array_update %[[AL_6]], %[[AL_27]], %[[AL_30]], %[[AL_35]] : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK: fir.result %[[AL_36]] : !fir.array<100x400xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[AL_6]] : !fir.array<100x400xf32>

     a(i,200+j) = 1.0 / b(j, i)
  end forall
  ! CHECK: }
  ! CHECK: fir.result %[[AL_19]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_4]] : !fir.array<100x400xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_48]], %[[AL_0]] to %[[a]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK: fir.freemem %[[VAL_45]] : !fir.heap<!fir.array<?x?x!fir.logical<4>>>
  ! CHECK: return
end subroutine test3_forall_construct

!*** Test a FORALL construct with an array assignment
!    This is similar to the following embedded WHERE construct test, but the
!    elements are assigned unconditionally.
! CHECK-LABEL: func @_QPtest_forall_with_array_assignment(
! CHECK-SAME: %[[a:[^:]*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>) {
subroutine test_forall_with_array_assignment(aa,bb)
  type t
     integer(kind=8) :: block1(64)
     integer(kind=8) :: block2(64)
  end type t
  type(t) :: aa(10), bb(10)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-NEXT: %[[VAL_1:.*]] = constant 10 : index
  ! CHECK: %[[VAL_2:.*]] = constant 10 : index
  ! CHECK: %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_4:.*]] = fir.array_load %[[a]](%[[VAL_3]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_13:.*]] = fir.array_load %[[b]](%[[VAL_12]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK-DAG: %[[VAL_15:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_16:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_18:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK: %[[VAL_20:.*]] = constant 10 : i32
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK: %[[VAL_22:.*]] = constant 2 : i32
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> index

  forall (i=1:10:2)
  ! CHECK-NEXT: %[[AL_0:.*]] = fir.do_loop %[[AL_1:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_23]] unordered iter_args(%[[AL_2:.*]] = %[[VAL_4]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK: %[[AL_3:.*]] = fir.convert %[[AL_1]] : (index) -> i32
  ! CHECK: fir.store %[[AL_3]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[NV_1:.*]] = constant 64 : i64
  ! CHECK: %[[NV_2:.*]] = fir.convert %[[NV_1]] : (i64) -> index
  ! CHECK: %[[AL_4:.*]] = fir.do_loop %[[AL_5:.*]] = %[[VAL_16]] to %[[NV_2]] step %[[VAL_15]] unordered iter_args(%[[AL_6:.*]] = %[[AL_2]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK: %[[AL_7:.*]] = fir.field_index block2, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK-DAG: %[[AL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK-DAG: %[[AL_9:.*]] = constant 1 : i32
  ! CHECK: %[[AL_10:.*]] = addi %[[AL_8]], %[[AL_9]] : i32
  ! CHECK: %[[AL_11:.*]] = fir.convert %[[AL_10]] : (i32) -> i64
  ! CHECK: %[[AL_12:.*]] = fir.convert %[[AL_11]] : (i64) -> index
  ! CHECK: %[[AL_13:.*]] = fir.array_fetch %[[VAL_13]], %[[AL_12]], %[[AL_7]], %[[AL_5]] : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, index, !fir.field, index) -> i64
  ! CHECK: %[[AL_14:.*]] = fir.field_index block1, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK: %[[AL_15:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[AL_16:.*]] = fir.convert %[[AL_15]] : (i32) -> i64
  ! CHECK: %[[AL_17:.*]] = fir.convert %[[AL_16]] : (i64) -> index
  ! CHECK: %[[AL_18:.*]] = fir.array_update %[[AL_6]], %[[AL_13]], %[[AL_17]], %[[AL_14]], %[[AL_5]] : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, i64, index, !fir.field, index) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK: fir.result %[[AL_18]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_4]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
     aa(i)%block1 = bb(i+1)%block2
  end forall
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_4]], %[[AL_0]] to %[[a]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>
  ! CHECK: return
end subroutine test_forall_with_array_assignment

!*** Test a FORALL construct with a nested WHERE construct.
!    This has both an explicit and implicit iteration space. The WHERE construct
!    makes the assignments conditional and the where mask evaluation must happen
!    prior to evaluating the array assignment statement.
! CHECK-LABEL: func @_QPtest_nested_forall_where(
! CHECK-SAME: %[[a:[^:]*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>,
! CHECK-SAME: %[[b:[^:]*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>)
subroutine test_nested_forall_where(a,b)
  ! CHECK-NEXT: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK: %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK: %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK: %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_6:.*]] = fir.array_load %[[a]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[b]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK-DAG: %[[VAL_16:.*]] = constant 3.140000e+00 : f32
  ! CHECK-DAG: %[[VAL_17:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[VAL_18:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_19:.*]]:3 = fir.box_dims %[[a]], %[[VAL_18]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_20:.*]] = fir.convert %[[VAL_19]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_21:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
  ! CHECK-DAG: %[[VAL_23:.*]] = addi %[[VAL_20]], %[[VAL_22]] : i64
  ! CHECK-DAG: %[[VAL_24:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_25:.*]] = subi %[[VAL_23]], %[[VAL_24]] : i64
  ! CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> i32
  ! CHECK: %[[VAL_27:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_28:.*]] = subi %[[VAL_26]], %[[VAL_17]] : i32
  ! CHECK: %[[VAL_29:.*]] = addi %[[VAL_28]], %[[VAL_27]] : i32
  ! CHECK: %[[VAL_30:.*]] = divi_signed %[[VAL_29]], %[[VAL_27]] : i32
  ! CHECK: %[[VAL_31:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_32:.*]] = cmpi sgt, %[[VAL_30]], %[[VAL_31]] : i32
  ! CHECK: %[[VAL_33:.*]] = select %[[VAL_32]], %[[VAL_30]], %[[VAL_31]] : i32
  ! CHECK: %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_35:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[VAL_36:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_37:.*]]:3 = fir.box_dims %[[a]], %{{.*}} : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_38:.*]] = fir.convert %[[VAL_37]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_39:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_40:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK-DAG: %[[VAL_41:.*]] = addi %[[VAL_38]], %[[VAL_40]] : i64
  ! CHECK-DAG: %[[VAL_42:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_43:.*]] = subi %[[VAL_41]], %[[VAL_42]] : i64
  ! CHECK: %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i64) -> i32
  ! CHECK: %[[VAL_45:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_46:.*]] = subi %[[VAL_44]], %[[VAL_35]] : i32
  ! CHECK: %[[VAL_47:.*]] = addi %[[VAL_46]], %[[VAL_45]] : i32
  ! CHECK: %[[VAL_48:.*]] = divi_signed %[[VAL_47]], %[[VAL_45]] : i32
  ! CHECK: %[[VAL_49:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_50:.*]] = cmpi sgt, %[[VAL_48]], %[[VAL_49]] : i32
  ! CHECK: %[[VAL_51:.*]] = select %[[VAL_50]], %[[VAL_48]], %[[VAL_49]] : i32
  ! CHECK: %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> index
  
  type t
     real data(100)
  end type t
  type(t) :: a(:,:), b(:,:)
  
  ! CHECK: %[[NV_2:.*]] = constant 100 : index
  ! CHECK: %[[VAL_53:.*]] = fir.shape %[[VAL_34]], %[[VAL_52]], %[[NV_2]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[VAL_54:.*]] = fir.allocmem !fir.array<?x?x?xi1>, %[[VAL_34]], %[[VAL_52]], %[[NV_2]] {uniq_name = ".array.expr"}
  ! CHECK: %[[VAL_55:.*]] = fir.shape %[[VAL_34]], %[[VAL_52]], %[[NV_2]] : (index, index, index) -> !fir.shape<3>
  ! CHECK: %[[VAL_56:.*]] = fir.array_load %[[VAL_54]](%[[VAL_55]]) : (!fir.heap<!fir.array<?x?x?xi1>>, !fir.shape<3>) -> !fir.array<?x?x?xi1>
  ! CHECK: %[[VAL_57:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_59:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_60:.*]]:3 = fir.box_dims %[[a]], %[[VAL_59]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_61:.*]] = fir.convert %[[VAL_60]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_62:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (index) -> i64
  ! CHECK-DAG: %[[VAL_64:.*]] = addi %[[VAL_61]], %[[VAL_63]] : i64
  ! CHECK-DAG: %[[VAL_65:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_66:.*]] = subi %[[VAL_64]], %[[VAL_65]] : i64
  ! CHECK: %[[VAL_67:.*]] = fir.convert %[[VAL_66]] : (i64) -> i32
  ! CHECK: %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (i32) -> index
  ! CHECK: %[[VAL_69:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_70:.*]] = fir.convert %[[VAL_69]] : (i32) -> index
  ! CHECK: %[[VAL_71:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (i32) -> index
  ! CHECK-DAG: %[[VAL_73:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_74:.*]]:3 = fir.box_dims %[[a]], %{{.*}} : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[VAL_75:.*]] = fir.convert %[[VAL_74]]#1 : (index) -> i64
  ! CHECK-DAG: %[[VAL_76:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_77:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK-DAG: %[[VAL_78:.*]] = addi %[[VAL_75]], %[[VAL_77]] : i64
  ! CHECK-DAG: %[[VAL_79:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_80:.*]] = subi %[[VAL_78]], %[[VAL_79]] : i64
  ! CHECK: %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (i64) -> i32
  ! CHECK: %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (i32) -> index
  ! CHECK: %[[VAL_83:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_84:.*]] = fir.convert %[[VAL_83]] : (i32) -> index
  
  forall (i=1:ubound(a,1), j=1:ubound(a,2))
  ! Compute first mask
  ! CHECK-NEXT: %[[AL_0:.*]] = fir.do_loop %[[AL_1:.*]] = %[[VAL_58]] to %[[VAL_68]] step %[[VAL_70]] unordered iter_args(%[[AL_2:.*]] = %[[VAL_56]]) -> (!fir.array<?x?x?xi1>) {
  ! CHECK: %[[AL_3:.*]] = fir.convert %[[AL_1]] : (index) -> i32
  ! CHECK: fir.store %[[AL_3]] to %[[VAL_5]] : !fir.ref<i32>
  ! CHECK-NEXT: %[[AL_4:.*]] = fir.do_loop %[[AL_5:.*]] = %[[VAL_72]] to %[[VAL_82]] step %[[VAL_84]] unordered iter_args(%[[AL_6:.*]] = %[[AL_2]]) -> (!fir.array<?x?x?xi1>) {
  ! CHECK: %[[AL_7:.*]] = fir.convert %[[AL_5]] : (index) -> i32
  ! CHECK: fir.store %[[AL_7]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[AL_8:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[AL_9:.*]] = fir.convert %[[AL_8]] : (i32) -> index
  ! CHECK: %[[AL_10:.*]] = subi %[[AL_9]], %[[VAL_58]] : index
  ! CHECK: %[[AL_11:.*]] = divi_signed %[[AL_10]], %[[VAL_70]] : index
  ! CHECK: %[[AL_12:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[AL_13:.*]] = fir.convert %[[AL_12]] : (i32) -> index
  ! CHECK: %[[AL_14:.*]] = subi %[[AL_13]], %[[VAL_72]] : index
  ! CHECK: %[[AL_15:.*]] = divi_signed %[[AL_14]], %[[VAL_84]] : index
  ! CHECK: %[[NV_1:.*]] = constant 100 : i64
  ! CHECK: %[[NV_2:.*]] = fir.convert %[[NV_1]] : (i64) -> index
  ! CHECK: %[[AL_16:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[AL_17:.*]] = fir.convert %[[AL_16]] : (i32) -> i64
  ! CHECK: %[[AL_18:.*]] = constant 1 : i64
  ! CHECK: %[[AL_19:.*]] = subi %[[AL_17]], %[[AL_18]] : i64
  ! CHECK-NEXT: %[[AL_20:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[AL_21:.*]] = fir.convert %[[AL_20]] : (i32) -> i64
  ! CHECK: %[[AL_22:.*]] = constant 1 : i64
  ! CHECK: %[[AL_23:.*]] = subi %[[AL_21]], %[[AL_22]] : i64
  ! CHECK-NEXT: %[[AL_24:.*]] = fir.coordinate_of %[[b]], %[[AL_19]], %[[AL_23]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, i64, i64) -> !fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[AL_25:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[AL_26:.*]] = fir.coordinate_of %[[AL_24]], %[[AL_25]] : (!fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK: %[[AL_27:.*]] = constant 100 : index
  ! CHECK: %[[AL_28:.*]] = fir.shape %[[AL_27]] : (index) -> !fir.shape<1>
  ! CHECK: %[[AL_29:.*]] = constant 1 : index
  ! CHECK: %[[AL_30:.*]] = fir.slice %[[AL_29]], %[[AL_27]], %[[AL_29]] : (index, index, index) -> !fir.slice<1>
  ! CHECK: %[[AL_31:.*]] = fir.array_load %[[AL_26]](%[[AL_28]]) {{\[}}%[[AL_30]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[AL_32:.*]] = constant 0.000000e+00 : f32
  ! CHECK-DAG: %[[AL_33:.*]] = constant 1 : index
  ! CHECK-DAG: %[[AL_34:.*]] = constant 0 : index
  ! CHECK: %[[AL_35:.*]] = subi %[[NV_2]], %[[AL_33]] : index
  ! CHECK: %[[AL_36:.*]] = fir.do_loop %[[AL_37:.*]] = %[[AL_34]] to %[[AL_35]] step %[[AL_33]] unordered iter_args(%[[AL_38:.*]] = %[[VAL_56]]) -> (!fir.array<?x?x?xi1>) {
  ! CHECK: %[[AL_39:.*]] = fir.array_fetch %[[AL_31]], %[[AL_37]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK: %[[AL_40:.*]] = cmpf ogt, %[[AL_39]], %[[AL_32]] : f32
  ! CHECK: %[[AL_41:.*]] = fir.array_update %[[AL_38]], %[[AL_40]], %[[AL_11]], %[[AL_15]], %[[AL_37]] : (!fir.array<?x?x?xi1>, i1, index, index, index) -> !fir.array<?x?x?xi1>
  ! CHECK-NEXT: fir.result %[[AL_41]] : !fir.array<?x?x?xi1>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_36]] : !fir.array<?x?x?xi1>
  ! CHECK: }
  ! CHECK: fir.result %[[AL_4]] : !fir.array<?x?x?xi1>
  ! CHECK: }
  ! CHECK-NEXT: fir.array_merge_store %[[VAL_56]], %[[AL_0]] to %[[VAL_54]] : !fir.array<?x?x?xi1>, !fir.array<?x?x?xi1>, !fir.heap<!fir.array<?x?x?xi1>>

  ! CHECK-DAG: %[[WAL_45:.*]] = constant 1 : index
  ! CHECK-DAG: %[[WAL_46:.*]] = constant 0 : index
  ! CHECK-DAG: %[[WAL_48:.*]] = constant 1 : i32
  ! CHECK: %[[WAL_49:.*]] = fir.convert %[[WAL_48]] : (i32) -> index
  ! CHECK-DAG: %[[WAL_50:.*]] = constant 0 : index
  ! CHECK-DAG: %[[WAL_51:.*]]:3 = fir.box_dims %[[a]], %[[WAL_50]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[WAL_52:.*]] = fir.convert %[[WAL_51]]#1 : (index) -> i64
  ! CHECK-DAG: %[[WAL_53:.*]] = constant 1 : index
  ! CHECK-DAG: %[[WAL_54:.*]] = fir.convert %[[WAL_53]] : (index) -> i64
  ! CHECK-DAG: %[[WAL_55:.*]] = addi %[[WAL_52]], %[[WAL_54]] : i64
  ! CHECK-DAG: %[[WAL_56:.*]] = constant 1 : i64
  ! CHECK: %[[WAL_57:.*]] = subi %[[WAL_55]], %[[WAL_56]] : i64
  ! CHECK: %[[WAL_58:.*]] = fir.convert %[[WAL_57]] : (i64) -> i32
  ! CHECK: %[[WAL_59:.*]] = fir.convert %[[WAL_58]] : (i32) -> index
  ! CHECK: %[[WAL_60:.*]] = constant 1 : index

     where (b(j,i)%data > 0.0)
        a(i,j)%data = b(j,i)%data / 3.14
  ! CHECK-NEXT: %[[XL_0:.*]] = fir.do_loop %[[XL_1:.*]] = %[[WAL_49]] to %[[WAL_59]] step %[[WAL_60]] unordered iter_args(%[[XL_2:.*]] = %[[VAL_6]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[XL_3:.*]] = fir.convert %[[XL_1]] : (index) -> i32
  ! CHECK: fir.store %[[XL_3]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[XL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[XL_5:.*]] = fir.convert %[[XL_4]] : (i32) -> index
  ! CHECK: %[[XL_6:.*]] = subi %[[XL_5]], %[[WAL_49]] : index
  ! CHECK: %[[XL_7:.*]] = divi_signed %[[XL_6]], %[[WAL_60]] : index
  ! CHECK: %[[WAL_61:.*]] = constant 1 : i32
  ! CHECK: %[[WAL_62:.*]] = fir.convert %[[WAL_61]] : (i32) -> index
  ! CHECK-DAG: %[[WAL_63:.*]] = constant 1 : index
  ! CHECK-DAG: %[[WAL_64:.*]]:3 = fir.box_dims %[[a]], %{{.*}} : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[WAL_65:.*]] = fir.convert %[[WAL_64]]#1 : (index) -> i64
  ! CHECK-DAG: %[[WAL_66:.*]] = constant 1 : index
  ! CHECK-DAG: %[[WAL_67:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK-DAG: %[[WAL_68:.*]] = addi %[[WAL_65]], %[[WAL_67]] : i64
  ! CHECK-DAG: %[[WAL_69:.*]] = constant 1 : i64
  ! CHECK: %[[WAL_70:.*]] = subi %[[WAL_68]], %[[WAL_69]] : i64
  ! CHECK: %[[WAL_71:.*]] = fir.convert %[[WAL_70]] : (i64) -> i32
  ! CHECK: %[[WAL_72:.*]] = fir.convert %[[WAL_71]] : (i32) -> index
  ! CHECK: %[[WAL_73:.*]] = constant 1 : index
  ! CHECK-NEXT: %[[XL_8:.*]] = fir.do_loop %[[XL_9:.*]] = %[[WAL_62]] to %[[WAL_72]] step %[[WAL_73]] unordered iter_args(%[[XL_10:.*]] = %[[XL_2]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[XL_11:.*]] = fir.convert %[[XL_9]] : (index) -> i32
  ! CHECK: fir.store %[[XL_11]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[XL_12:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[XL_13:.*]] = fir.convert %[[XL_12]] : (i32) -> index
  ! CHECK: %[[XL_14:.*]] = subi %[[XL_13]], %[[WAL_62]] : index
  ! CHECK: %[[XL_15:.*]] = divi_signed %[[XL_14]], %[[WAL_73]] : index
  ! CHECK: %[[NV_1:.*]] = constant 100 : i64
  ! CHECK: %[[NV_2:.*]] = fir.convert %[[NV_1]] : (i64) -> index
  ! CHECK-NEXT: %[[XL_16:.*]] = fir.do_loop %[[XL_17:.*]] = %[[WAL_46]] to %[[NV_2]] step %[[WAL_45]] unordered iter_args(%[[XL_18:.*]] = %[[XL_10]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[XL_19:.*]] = constant 1 : index
  ! CHECK: %[[XL_20:.*]] = addi %[[XL_17]], %[[XL_19]] : index
  ! CHECK: %[[XL_21:.*]] = fir.array_coor %[[VAL_54]](%[[VAL_53]]) %[[XL_7]], %[[XL_15]], %[[XL_20]] : (!fir.heap<!fir.array<?x?x?xi1>>, !fir.shape<3>, index, index, index) -> !fir.ref<i1>
  ! CHECK: %[[XL_22:.*]] = fir.load %[[XL_21]] : !fir.ref<i1>
  ! CHECK: %[[XL_23:.*]] = fir.if %[[XL_22]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[XL_24:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[XL_25:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[XL_26:.*]] = fir.convert %[[XL_25]] : (i32) -> i64
  ! CHECK: %[[XL_27:.*]] = fir.convert %[[XL_26]] : (i64) -> index
  ! CHECK: %[[XL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[XL_29:.*]] = fir.convert %[[XL_28]] : (i32) -> i64
  ! CHECK: %[[XL_30:.*]] = fir.convert %[[XL_29]] : (i64) -> index
  ! CHECK: %[[XL_31:.*]] = fir.array_fetch %[[VAL_14]], %[[XL_30]], %[[XL_27]], %[[XL_24]], %[[XL_17]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
  ! CHECK: %[[XL_32:.*]] = divf %[[XL_31]], %[[VAL_16]] : f32
  ! CHECK: %[[XL_33:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[XL_34:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[XL_35:.*]] = fir.convert %[[XL_34]] : (i32) -> i64
  ! CHECK: %[[XL_36:.*]] = fir.convert %[[XL_35]] : (i64) -> index
  ! CHECK: %[[XL_37:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[XL_38:.*]] = fir.convert %[[XL_37]] : (i32) -> i64
  ! CHECK: %[[XL_39:.*]] = fir.convert %[[XL_38]] : (i64) -> index
  ! CHECK: %[[XL_40:.*]] = fir.array_update %[[XL_18]], %[[XL_32]], %[[XL_39]], %[[XL_36]], %[[XL_33]], %[[XL_17]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: fir.result %[[XL_40]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: } else {
  ! CHECK: fir.result %[[XL_18]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[XL_23]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[XL_16]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[XL_8]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK-NEXT: fir.array_merge_store %[[VAL_6]], %[[XL_0]] to %[[a]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
     elsewhere
  ! CHECK: %[[YAL_45:.*]] = fir.array_load %[[a]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[YAL_52:.*]] = fir.array_load %[[b]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK-DAG: %[[YAL_53:.*]] = constant 1 : index
  ! CHECK-DAG: %[[YAL_54:.*]] = constant 0 : index
  ! CHECK-DAG: %[[YAL_56:.*]] = constant 1 : i32
  ! CHECK: %[[YAL_57:.*]] = fir.convert %[[YAL_56]] : (i32) -> index
  ! CHECK-DAG: %[[YAL_58:.*]] = constant 0 : index
  ! CHECK-DAG: %[[YAL_59:.*]]:3 = fir.box_dims %[[a]], %[[YAL_58]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[YAL_60:.*]] = fir.convert %[[YAL_59]]#1 : (index) -> i64
  ! CHECK-DAG: %[[YAL_61:.*]] = constant 1 : index
  ! CHECK-DAG: %[[YAL_62:.*]] = fir.convert %[[YAL_61]] : (index) -> i64
  ! CHECK-DAG: %[[YAL_63:.*]] = addi %[[YAL_60]], %[[YAL_62]] : i64
  ! CHECK-DAG: %[[YAL_64:.*]] = constant 1 : i64
  ! CHECK: %[[YAL_65:.*]] = subi %[[YAL_63]], %[[YAL_64]] : i64
  ! CHECK: %[[YAL_66:.*]] = fir.convert %[[YAL_65]] : (i64) -> i32
  ! CHECK: %[[YAL_67:.*]] = fir.convert %[[YAL_66]] : (i32) -> index
  ! CHECK: %[[YAL_68:.*]] = constant 1 : index
        a(i,j)%data = -b(j,i)%data
  ! CHECK-NEXT: %[[ZL_0:.*]] = fir.do_loop %[[ZL_1:.*]] = %[[YAL_57]] to %[[YAL_67]] step %[[YAL_68]] unordered iter_args(%[[ZL_2:.*]] = %[[YAL_45]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[ZL_3:.*]] = fir.convert %[[ZL_1]] : (index) -> i32
  ! CHECK: fir.store %[[ZL_3]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[ZL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[ZL_5:.*]] = fir.convert %[[ZL_4]] : (i32) -> index
  ! CHECK: %[[ZL_6:.*]] = subi %[[ZL_5]], %[[YAL_57]] : index
  ! CHECK: %[[ZL_7:.*]] = divi_signed %[[ZL_6]], %[[YAL_68]] : index
  ! CHECK: %[[YAL_69:.*]] = constant 1 : i32
  ! CHECK: %[[YAL_70:.*]] = fir.convert %[[YAL_69]] : (i32) -> index
  ! CHECK-DAG: %[[YAL_71:.*]] = constant 1 : index
  ! CHECK-DAG: %[[YAL_72:.*]]:3 = fir.box_dims %[[a]], %{{.*}} : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[YAL_73:.*]] = fir.convert %[[YAL_72]]#1 : (index) -> i64
  ! CHECK-DAG: %[[YAL_74:.*]] = constant 1 : index
  ! CHECK-DAG: %[[YAL_75:.*]] = fir.convert %{{.*}} : (index) -> i64
  ! CHECK-DAG: %[[YAL_76:.*]] = addi %[[YAL_73]], %[[YAL_75]] : i64
  ! CHECK-DAG: %[[YAL_77:.*]] = constant 1 : i64
  ! CHECK: %[[YAL_78:.*]] = subi %[[YAL_76]], %[[YAL_77]] : i64
  ! CHECK: %[[YAL_79:.*]] = fir.convert %[[YAL_78]] : (i64) -> i32
  ! CHECK: %[[YAL_80:.*]] = fir.convert %[[YAL_79]] : (i32) -> index
  ! CHECK: %[[YAL_81:.*]] = constant 1 : index
  ! CHECK-NEXT: %[[ZL_8:.*]] = fir.do_loop %[[ZL_9:.*]] = %[[YAL_70]] to %[[YAL_80]] step %[[YAL_81]] unordered iter_args(%[[ZL_10:.*]] = %[[ZL_2]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[ZL_11:.*]] = fir.convert %[[ZL_9]] : (index) -> i32
  ! CHECK: fir.store %[[ZL_11]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[ZL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[ZL_13:.*]] = fir.convert %[[ZL_12]] : (i32) -> index
  ! CHECK: %[[ZL_14:.*]] = subi %[[ZL_13]], %[[YAL_70]] : index
  ! CHECK: %[[ZL_15:.*]] = divi_signed %[[ZL_14]], %[[YAL_81]] : index
  ! CHECK: %[[NV_1:.*]] = constant 100 : i64
  ! CHECK: %[[NV_2:.*]] = fir.convert %[[NV_1]] : (i64) -> index
  ! CHECK: %[[ZL_16:.*]] = fir.do_loop %[[ZL_17:.*]] = %[[YAL_54]] to %[[NV_2]] step %[[YAL_53]] unordered iter_args(%[[ZL_18:.*]] = %[[ZL_10]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[ZL_19:.*]] = constant 1 : index
  ! CHECK: %[[ZL_20:.*]] = addi %[[ZL_17]], %[[ZL_19]] : index
  ! CHECK: %[[ZL_21:.*]] = fir.array_coor %[[VAL_54]](%[[VAL_53]]) %[[ZL_7]], %[[ZL_15]], %[[ZL_20]] : (!fir.heap<!fir.array<?x?x?xi1>>, !fir.shape<3>, index, index, index) -> !fir.ref<i1>
  ! CHECK: %[[ZL_22:.*]] = fir.load %[[ZL_21]] : !fir.ref<i1>
  ! CHECK: %[[ZL_23:.*]] = fir.if %[[ZL_22]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: fir.result %[[ZL_18]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: } else {
  ! CHECK: %[[ZL_24:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[ZL_25:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[ZL_26:.*]] = fir.convert %[[ZL_25]] : (i32) -> i64
  ! CHECK: %[[ZL_27:.*]] = fir.convert %[[ZL_26]] : (i64) -> index
  ! CHECK: %[[ZL_28:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[ZL_29:.*]] = fir.convert %[[ZL_28]] : (i32) -> i64
  ! CHECK: %[[ZL_30:.*]] = fir.convert %[[ZL_29]] : (i64) -> index
  ! CHECK: %[[ZL_31:.*]] = fir.array_fetch %[[YAL_52]], %[[ZL_30]], %[[ZL_27]], %[[ZL_24]], %[[ZL_17]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
  ! CHECK: %[[ZL_32:.*]] = negf %[[ZL_31]] : f32
  ! CHECK: %[[ZL_33:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[ZL_34:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[ZL_35:.*]] = fir.convert %[[ZL_34]] : (i32) -> i64
  ! CHECK: %[[ZL_36:.*]] = fir.convert %[[ZL_35]] : (i64) -> index
  ! CHECK: %[[ZL_37:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[ZL_38:.*]] = fir.convert %[[ZL_37]] : (i32) -> i64
  ! CHECK: %[[ZL_39:.*]] = fir.convert %[[ZL_38]] : (i64) -> index
  ! CHECK: %[[ZL_40:.*]] = fir.array_update %[[ZL_18]], %[[ZL_32]], %[[ZL_39]], %[[ZL_36]], %[[ZL_33]], %[[ZL_17]] : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: fir.result %[[ZL_40]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[ZL_23]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[ZL_16]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[ZL_8]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
     end where
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[YAL_45]], %[[ZL_0]] to %[[a]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  end forall
  ! CHECK: fir.freemem %[[VAL_54]] : !fir.heap<!fir.array<?x?x?xi1>>
  ! CHECK: return
end subroutine test_nested_forall_where

! CHECK-LABEL: func @_QPtest_forall_with_slice(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.ref<i32>,
! CHECK-SAME: %[[arg1:[^:]*]]: !fir.ref<i32>)
subroutine test_forall_with_slice(i1,i2)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK: %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_2:.*]] = constant 10 : index
  ! CHECK: %[[VAL_3:.*]] = constant 10 : index
  ! CHECK: %[[VAL_4:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_sliceEa"}
  ! CHECK: %[[VAL_5:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK-DAG: %[[VAL_13:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_14:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_15:.*]] = constant 0 : index
  ! CHECK: %[[VAL_17:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK: %[[VAL_19:.*]] = constant 5 : i32
  ! CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> index
  ! CHECK: %[[VAL_21:.*]] = constant 1 : index
  interface
     pure integer function f(i)
       integer i
       intent(in) i
     end function f
  end interface
  type t
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)

  forall (i=1:5, j=1:10)
  ! CHECK: %[[V_0:.*]] = fir.do_loop %[[V_1:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] unordered iter_args(%[[V_2:.*]] = %[[VAL_6]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK: %[[V_3:.*]] = fir.convert %[[V_1]] : (index) -> i32
  ! CHECK: fir.store %[[V_3]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[V_4:.*]] = constant 1 : i32
  ! CHECK: %[[V_5:.*]] = fir.convert %[[V_4]] : (i32) -> index
  ! CHECK: %[[V_6:.*]] = constant 10 : i32
  ! CHECK: %[[V_7:.*]] = fir.convert %[[V_6]] : (i32) -> index
  ! CHECK: %[[V_8:.*]] = constant 1 : index
  ! CHECK: %[[V_9:.*]] = fir.do_loop %[[V_10:.*]] = %[[V_5]] to %[[V_7]] step %[[V_8]] unordered iter_args(%[[V_11:.*]] = %[[V_2]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK: %[[V_12:.*]] = fir.convert %[[V_10]] : (index) -> i32
  ! CHECK: fir.store %[[V_12]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[NV_1:.*]] = constant 11 : i64
  ! CHECK: %[[NV_2:.*]] = fir.convert %[[NV_1]] : (i64) -> index
  ! CHECK: %[[V_13:.*]] = fir.do_loop %[[V_14:.*]] = %[[VAL_15]] to %[[NV_2]] step %[[VAL_14]] unordered iter_args(%[[V_15:.*]] = %[[V_11]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK: %[[V_16:.*]] = fir.call @_QPf(%[[VAL_1]]) : (!fir.ref<i32>) -> i32
  ! CHECK: %[[V_17:.*]] = fir.load %[[arg1]] : !fir.ref<i32>
  ! CHECK: %[[V_19:.*]] = fir.convert %[[V_17]] : (i32) -> i64
  ! CHECK: %[[V_20:.*]] = fir.convert %[[V_19]] : (i64) -> index
  ! CHECK-NEXT: %[[V_21:.*]] = muli %[[V_20]], %[[V_14]] : index
  ! CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
  ! CHECK: %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
  ! CHECK: %[[VAL_25:.*]] = addi %[[V_21]], %[[VAL_24]] : index
  ! CHECK: %[[VAL_26:.*]] = subi %[[VAL_25]], %[[VAL_13]] : index
  ! CHECK: %[[VAL_27:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>
  ! CHECK: %[[VAL_28:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
  ! CHECK: %[[VAL_31:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK: %[[VAL_34:.*]] = fir.array_update %[[V_15]], %[[V_16]], %[[VAL_33]], %[[VAL_30]], %[[VAL_27]], %[[VAL_26]] : (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK: fir.result %[[VAL_34]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[V_13]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[V_9]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
     a(i,j)%arr(i:i1:i2) = f(i)
  end forall
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_6]], %[[V_0]] to %[[VAL_4]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>
  ! CHECK: return
end subroutine test_forall_with_slice

! CHECK-LABEL: func @_QPtest_forall_with_ranked_dimension(
subroutine test_forall_with_ranked_dimension
  ! CHECK-DAG: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK-DAG: %[[VAL_1:.*]] = constant 10 : index
  ! CHECK-DAG: %[[VAL_2:.*]] = constant 10 : index
  ! CHECK-DAG: %[[VAL_3:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_ranked_dimensionEa"}
  ! CHECK: %[[VAL_4:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_5:.*]] = fir.array_load %[[VAL_3]](%[[VAL_4]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK-DAG: %[[VAL_16:.*]] = constant 10 : i64
  ! CHECK-DAG: %[[NV_1:.*]] = constant 1 : i64
  ! CHECK-DAG: %[[NV_2:.*]] = subi %[[VAL_16]], %{{.*}} : i64
  ! CHECK-DAG: %[[NV_3:.*]] = constant 1 : i64
  ! CHECK-DAG: %[[NV_4:.*]] = addi %[[NV_2]], %{{.*}} : i64
  ! CHECK-DAG: %[[NV_5:.*]] = constant 1 : i64
  ! CHECK: %[[NV_6:.*]] = divi_signed %[[NV_4]], %{{.*}} : i64
  ! CHECK: %[[NV_7:.*]] = constant 0 : i64
  ! CHECK: %[[NV_8:.*]] = cmpi sgt, %[[NV_6]], %[[NV_7]] : i64
  ! CHECK: %[[NV_9:.*]] = select %[[NV_8]], %[[NV_6]], %[[NV_7]] : i64
  ! CHECK: %[[NV_10:.*]] = fir.convert %[[NV_9]] : (i64) -> index
  ! CHECK-DAG: %[[VAL_17:.*]] = constant 1 : index
  ! CHECK-DAG: %[[VAL_18:.*]] = constant 0 : index
  ! CHECK-DAG: %[[VAL_20:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK: %[[VAL_22:.*]] = constant 5 : i32
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> index
  ! CHECK: %[[VAL_24:.*]] = constant 1 : index
  interface
     pure integer function f(i)
       integer, intent(in) :: i
     end function f
  end interface
  type t
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)
  
  ! CHECK: %[[V_0:.*]] = fir.do_loop %[[V_1:.*]] = %[[VAL_21]] to %[[VAL_23]] step %[[VAL_24]] unordered iter_args(%[[V_2:.*]] = %[[VAL_5]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK: %[[V_3:.*]] = fir.convert %[[V_1]] : (index) -> i32
  ! CHECK: fir.store %[[V_3]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK-DAG: %[[NV_11:.*]] = constant 1 : index
  ! CHECK-DAG: %[[NV_12:.*]] = constant 1 : i64
  ! CHECK: %[[NV_13:.*]] = fir.convert %[[NV_12]] : (i64) -> index
  ! CHECK: %[[NV_14:.*]] = subi %[[NV_10]], %[[NV_11]] : index
  ! CHECK: %[[NV_15:.*]] = addi %[[NV_14]], %[[NV_13]] : index
  ! CHECK: %[[NV_16:.*]] = divi_signed %[[NV_15]], %[[NV_13]] : index
  ! CHECK: %[[V_4:.*]] = fir.do_loop %[[V_5:.*]] = %[[VAL_18]] to %[[NV_16]] step %[[VAL_17]] unordered iter_args(%[[V_6:.*]] = %[[V_2]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  forall (i=1:5)
  ! CHECK: %[[V_7:.*]] = fir.call @_QPf(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  ! CHECK-DAG: %[[V_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_11:.*]] = fir.convert %[[V_8]] : (i32) -> i64
  ! CHECK: %[[V_12:.*]] = fir.convert %[[V_11]] : (i64) -> index
  ! CHECK: %[[V_13:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>
  ! CHECK: %[[V_14:.*]] = constant 1 : i64
  ! CHECK: %[[V_15:.*]] = fir.convert %[[V_14]] : (i64) -> index
  ! CHECK: %[[V_16:.*]] = muli %[[V_15]], %[[V_5]] : index
  ! CHECK: %[[V_17:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_18:.*]] = fir.convert %[[V_17]] : (i32) -> i64
  ! CHECK: %[[V_19:.*]] = fir.convert %[[V_18]] : (i64) -> index
  ! CHECK: %[[V_20:.*]] = fir.array_update %[[V_6]], %[[V_7]], %[[V_19]], %[[V_16]], %[[V_13]], %[[V_12]] : (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK: fir.result %[[V_20]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[V_4]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK: }
     a(i,:)%arr(i) = f(i)
  end forall
  ! CHECK: fir.array_merge_store %[[VAL_5]], %[[V_0]] to %[[VAL_3]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>
  ! CHECK: return
end subroutine test_forall_with_ranked_dimension

! CHECK-LABEL: func @_QPforall_with_allocatable(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.box<!fir.array<?xf32>>) {
subroutine forall_with_allocatable(a1)
  real :: a1(:)
  real, allocatable :: arr(:)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "arr", uniq_name = "_QFforall_with_allocatableEarr"}
  ! CHECK: %[[VAL_2:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QFforall_with_allocatableEarr.addr"}
  ! CHECK: %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.lb0"}
  ! CHECK: %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.ext0"}
  ! CHECK: %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  ! CHECK: fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
  ! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: %[[VAL_9:.*]] = fir.shape_shift %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_10:.*]] = fir.array_load %[[VAL_8]](%[[VAL_9]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_11:.*]] = fir.array_load %[[arg0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_13:.*]] = constant 5 : i32
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
  ! CHECK: %[[VAL_15:.*]] = constant 15 : i32
  ! CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
  ! CHECK: %[[VAL_17:.*]] = constant 1 : index

  ! CHECK: %[[V_0:.*]] = fir.do_loop %[[V_1:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_17]] unordered iter_args(%[[V_2:.*]] = %[[VAL_10]]) -> (!fir.array<?xf32>) {
  ! CHECK: %[[V_3:.*]] = fir.convert %[[V_1]] : (index) -> i32
  ! CHECK: fir.store %[[V_3]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_5:.*]] = fir.convert %[[V_4]] : (i32) -> i64
  ! CHECK: %[[V_6:.*]] = fir.convert %[[V_5]] : (i64) -> index
  ! CHECK: %[[V_7:.*]] = fir.array_fetch %[[VAL_11]], %[[V_6]] : (!fir.array<?xf32>, index) -> f32
  ! CHECK: %[[V_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_9:.*]] = fir.convert %[[V_8]] : (i32) -> i64
  ! CHECK: %[[V_10:.*]] = fir.convert %[[V_9]] : (i64) -> index
  ! CHECK: %[[V_11:.*]] = fir.array_update %[[V_2]], %[[V_7]], %[[V_10]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK: fir.result %[[V_11]] : !fir.array<?xf32>
  forall (i=5:15)
     arr(i) = a1(i)
  end forall
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_10]], %[[V_0]] to %[[VAL_8]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
  ! CHECK: return
end subroutine forall_with_allocatable

! CHECK-LABEL: func @_QPforall_with_allocatable2(
! CHECK-SAME: %[[arg0:[^:]*]]: !fir.box<!fir.array<?xf32>>) {
subroutine forall_with_allocatable2(a1)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {bindc_name = "thing", uniq_name = "_QFforall_with_allocatable2Ething"}
  ! CHECK: %[[VAL_2:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
  ! CHECK: %[[VAL_4:.*]] = constant {{[0-9]+}} : i32
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_7:.*]] = fir.call @_FortranAInitialize(%[[VAL_5]], %[[VAL_6]], %[[VAL_4]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[VAL_8:.*]] = fir.field_index arr, !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK: %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_8]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[VAL_11:.*]] = constant 0 : index
  ! CHECK: %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_11]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[VAL_13:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[VAL_14:.*]] = fir.shape_shift %[[VAL_12]]#0, %[[VAL_12]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_15:.*]] = constant 1 : index
  ! CHECK: %[[VAL_16:.*]] = fir.slice %[[VAL_12]]#0, %[[VAL_12]]#1, %[[VAL_15]] : (index, index, index) -> !fir.slice<1>
  ! CHECK: %[[VAL_17:.*]] = fir.array_load %[[VAL_13]](%[[VAL_14]]) {{\[}}%[[VAL_16]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_18:.*]] = fir.array_load %[[arg0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[VAL_20:.*]] = constant 5 : i32
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK: %[[VAL_22:.*]] = constant 15 : i32
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> index
  ! CHECK: %[[VAL_24:.*]] = constant 1 : index
  real :: a1(:)
  type t
     integer :: i
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing

  ! CHECK: %[[V_0:.*]] = fir.do_loop %[[V_1:.*]] = %[[VAL_21]] to %[[VAL_23]] step %[[VAL_24]] unordered iter_args(%[[V_2:.*]] = %[[VAL_17]]) -> (!fir.array<?xf32>) {
  ! CHECK: %[[V_3:.*]] = fir.convert %[[V_1]] : (index) -> i32
  ! CHECK: fir.store %[[V_3]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_5:.*]] = fir.convert %[[V_4]] : (i32) -> i64
  ! CHECK: %[[V_6:.*]] = fir.convert %[[V_5]] : (i64) -> index
  ! CHECK: %[[V_7:.*]] = fir.array_fetch %[[VAL_18]], %[[V_6]] : (!fir.array<?xf32>, index) -> f32
  ! CHECK: %[[V_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK: %[[V_9:.*]] = fir.convert %[[V_8]] : (i32) -> i64
  ! CHECK: %[[V_10:.*]] = fir.convert %[[V_9]] : (i64) -> index
  ! CHECK: %[[V_11:.*]] = fir.array_update %[[V_2]], %[[V_7]], %[[V_10]] : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK: fir.result %[[V_11]] : !fir.array<?xf32>
  ! CHECK: }
  forall (i=5:15)
     thing%arr(i) = a1(i)
  end forall
  ! CHECK: fir.array_merge_store %[[VAL_17]], %[[V_0]] to %[[VAL_13]]{{\[}}%[[VAL_16]]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>, !fir.slice<1>
  ! CHECK: return
end subroutine forall_with_allocatable2
