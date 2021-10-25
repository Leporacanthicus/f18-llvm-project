! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** This FORALL construct does present a potential loop-carried dependence if
!*** implemented naively (and incorrectly). The final value of a(3) must be the
!*** value of a(2) before loopy begins execution added to b(2).
! CHECK-LABEL: func @_QPtest9(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<i32>) {
subroutine test9(a,b,n)
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
  ! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i64
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
  ! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
  ! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
  ! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_0]](%[[VAL_17]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_19:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_20:.*]] = fir.array_load %[[VAL_0]](%[[VAL_19]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_1]](%[[VAL_21]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_11]] to %[[VAL_15]] step %[[VAL_16]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_18]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_26]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
  ! CHECK:           %[[VAL_30:.*]] = fir.array_fetch %[[VAL_20]], %[[VAL_29]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK:           %[[VAL_34:.*]] = fir.array_fetch %[[VAL_22]], %[[VAL_33]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_35:.*]] = arith.addf %[[VAL_30]], %[[VAL_34]] : f32
  ! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_37:.*]] = arith.constant 1 : i32
  ! CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_36]], %[[VAL_37]] : i32
  ! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> i64
  ! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
  ! CHECK:           %[[VAL_41:.*]] = fir.array_update %[[VAL_25]], %[[VAL_35]], %[[VAL_40]] {Fortran.offsets} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_41]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_42:.*]] to %[[VAL_0]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>

  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  loopy: FORALL (i=1:n-1)
     a(i+1) = a(i) + b(i)
  END FORALL loopy
  ! CHECK: return
  ! CHECK: }
end subroutine test9

!*** Test a FORALL statement
! CHECK-LABEL: func @_QPtest_forall_stmt(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<200xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.array<200x!fir.logical<4>>>) {
subroutine test_forall_stmt(x, mask)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_3:.*]] = arith.constant 200 : index
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 100 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> !fir.array<200xf32>
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<200xf32>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
  ! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i64
  ! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_16]], %[[VAL_17]] : i64
  ! CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_18]] : (!fir.ref<!fir.array<200x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_22:.*]] = fir.if %[[VAL_21]] -> (!fir.array<200xf32>) {
  ! CHECK:             %[[VAL_23:.*]] = arith.constant 1.000000e+00 : f32
  ! CHECK:             %[[VAL_24:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
  ! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
  ! CHECK:             %[[VAL_27:.*]] = fir.array_update %[[VAL_13]], %[[VAL_23]], %[[VAL_26]] {Fortran.offsets} : (!fir.array<200xf32>, f32, index) -> !fir.array<200xf32>
  ! CHECK:             fir.result %[[VAL_27]] : !fir.array<200xf32>
  ! CHECK:           } else {
  ! CHECK:             fir.result %[[VAL_13]] : !fir.array<200xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_28:.*]] : !fir.array<200xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_29:.*]] to %[[VAL_0]] : !fir.array<200xf32>, !fir.array<200xf32>, !fir.ref<!fir.array<200xf32>>

  logical :: mask(200)
  real :: x(200)
  forall (i=1:100,mask(i)) x(i) = 1.
  ! CHECK: return
  ! CHECK: }
end subroutine test_forall_stmt

!*** Test a FORALL construct
! CHECK-LABEL: func @_QPtest_forall_construct(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>) {
subroutine test_forall_construct(a,b)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
  ! CHECK:         %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] : i64
  ! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_12]] : i64
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
  ! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_19]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_22:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (index) -> i64
  ! CHECK:         %[[VAL_24:.*]] = arith.addi %[[VAL_21]], %[[VAL_23]] : i64
  ! CHECK:         %[[VAL_25:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_26:.*]] = arith.subi %[[VAL_24]], %[[VAL_25]] : i64
  ! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> i32
  ! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
  ! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_30:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
  ! CHECK:         %[[VAL_31:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
  ! CHECK:         %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_5]] to %[[VAL_15]] step %[[VAL_16]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_30]]) -> (!fir.array<?x?xf32>) {
  ! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_35]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_36:.*]] = fir.do_loop %[[VAL_37:.*]] = %[[VAL_18]] to %[[VAL_28]] step %[[VAL_29]] unordered iter_args(%[[VAL_38:.*]] = %[[VAL_34]]) -> (!fir.array<?x?xf32>) {
  ! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_39]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_40:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> i64
  ! CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_43:.*]] = arith.subi %[[VAL_41]], %[[VAL_42]] : i64
  ! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
  ! CHECK:             %[[VAL_46:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_45]], %[[VAL_46]] : i64
  ! CHECK:             %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_43]], %[[VAL_47]] : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
  ! CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_48]] : !fir.ref<f32>
  ! CHECK:             %[[VAL_50:.*]] = arith.constant 0.000000e+00 : f32
  ! CHECK:             %[[VAL_51:.*]] = arith.cmpf ogt, %[[VAL_49]], %[[VAL_50]] : f32
  ! CHECK:             %[[VAL_52:.*]] = fir.if %[[VAL_51]] -> (!fir.array<?x?xf32>) {
  ! CHECK:               %[[VAL_53:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> i64
  ! CHECK:               %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i64) -> index
  ! CHECK:               %[[VAL_56:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i64
  ! CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:               %[[VAL_59:.*]] = fir.array_fetch %[[VAL_31]], %[[VAL_55]], %[[VAL_58]] {Fortran.offsets} : (!fir.array<?x?xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_60:.*]] = arith.constant 3.140000e+00 : f32
  ! CHECK:               %[[VAL_61:.*]] = divf %[[VAL_59]], %[[VAL_60]] : f32
  ! CHECK:               %[[VAL_62:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (i32) -> i64
  ! CHECK:               %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i64) -> index
  ! CHECK:               %[[VAL_65:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_66:.*]] = fir.convert %[[VAL_65]] : (i32) -> i64
  ! CHECK:               %[[VAL_67:.*]] = fir.convert %[[VAL_66]] : (i64) -> index
  ! CHECK:               %[[VAL_68:.*]] = fir.array_update %[[VAL_38]], %[[VAL_61]], %[[VAL_64]], %[[VAL_67]] {Fortran.offsets} : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  ! CHECK:               fir.result %[[VAL_68]] : !fir.array<?x?xf32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_38]] : !fir.array<?x?xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_69:.*]] : !fir.array<?x?xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_70:.*]] : !fir.array<?x?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_30]], %[[VAL_71:.*]] to %[[VAL_0]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.box<!fir.array<?x?xf32>>

  real :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2), b(j,i) > 0.0)
     a(i,j) = b(j,i) / 3.14
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test_forall_construct

!*** Test forall with multiple assignment statements
! CHECK-LABEL: func @_QPtest2_forall_construct(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>>) {
subroutine test2_forall_construct(a,b)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 100 : index
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 400 : index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 200 : index
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 200 : index
  ! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
  ! CHECK:         %[[VAL_12:.*]] = arith.constant 100 : i32
  ! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
  ! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
  ! CHECK:         %[[VAL_17:.*]] = arith.constant 200 : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_20:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_0]](%[[VAL_20]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_23:.*]] = fir.array_load %[[VAL_1]](%[[VAL_22]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_24:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_25:.*]] = fir.array_load %[[VAL_1]](%[[VAL_24]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_26:.*]] = fir.do_loop %[[VAL_27:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_14]] unordered iter_args(%[[VAL_28:.*]] = %[[VAL_21]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_29]] to %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_30:.*]] = fir.do_loop %[[VAL_31:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] unordered iter_args(%[[VAL_32:.*]] = %[[VAL_28]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_31]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_33]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
  ! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
  ! CHECK:             %[[VAL_37:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> i64
  ! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
  ! CHECK:             %[[VAL_40:.*]] = fir.array_fetch %[[VAL_23]], %[[VAL_36]], %[[VAL_39]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i32
  ! CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_41]], %[[VAL_42]] : i32
  ! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
  ! CHECK:             %[[VAL_46:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> i64
  ! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
  ! CHECK:             %[[VAL_49:.*]] = fir.array_fetch %[[VAL_25]], %[[VAL_45]], %[[VAL_48]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_50:.*]] = arith.addf %[[VAL_40]], %[[VAL_49]] : f32
  ! CHECK:             %[[VAL_51:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> i64
  ! CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i64) -> index
  ! CHECK:             %[[VAL_54:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i32) -> i64
  ! CHECK:             %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i64) -> index
  ! CHECK:             %[[VAL_57:.*]] = fir.array_update %[[VAL_32]], %[[VAL_50]], %[[VAL_53]], %[[VAL_56]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:             fir.result %[[VAL_57]] : !fir.array<100x400xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_58:.*]] : !fir.array<100x400xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_59:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK:         %[[VAL_60:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_61:.*]] = fir.array_load %[[VAL_0]](%[[VAL_60]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_62:.*]] = fir.shape %[[VAL_8]], %[[VAL_9]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_63:.*]] = fir.array_load %[[VAL_1]](%[[VAL_62]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_64:.*]] = fir.do_loop %[[VAL_65:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_14]] unordered iter_args(%[[VAL_66:.*]] = %[[VAL_61]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:           %[[VAL_67:.*]] = fir.convert %[[VAL_65]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_67]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_68:.*]] = fir.do_loop %[[VAL_69:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] unordered iter_args(%[[VAL_70:.*]] = %[[VAL_66]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:             %[[VAL_71:.*]] = fir.convert %[[VAL_69]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_71]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_72:.*]] = arith.constant 1.000000e+00 : f32
  ! CHECK:             %[[VAL_73:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_74:.*]] = fir.convert %[[VAL_73]] : (i32) -> i64
  ! CHECK:             %[[VAL_75:.*]] = fir.convert %[[VAL_74]] : (i64) -> index
  ! CHECK:             %[[VAL_76:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (i32) -> i64
  ! CHECK:             %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : (i64) -> index
  ! CHECK:             %[[VAL_79:.*]] = fir.array_fetch %[[VAL_63]], %[[VAL_75]], %[[VAL_78]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_80:.*]] = divf %[[VAL_72]], %[[VAL_79]] : f32
  ! CHECK:             %[[VAL_81:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (i32) -> i64
  ! CHECK:             %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (i64) -> index
  ! CHECK:             %[[VAL_84:.*]] = arith.constant 200 : i32
  ! CHECK:             %[[VAL_85:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_86:.*]] = arith.addi %[[VAL_84]], %[[VAL_85]] : i32
  ! CHECK:             %[[VAL_87:.*]] = fir.convert %[[VAL_86]] : (i32) -> i64
  ! CHECK:             %[[VAL_88:.*]] = fir.convert %[[VAL_87]] : (i64) -> index
  ! CHECK:             %[[VAL_89:.*]] = fir.array_update %[[VAL_70]], %[[VAL_80]], %[[VAL_83]], %[[VAL_88]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:             fir.result %[[VAL_89]] : !fir.array<100x400xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_90:.*]] : !fir.array<100x400xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_61]], %[[VAL_91:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>

  real :: a(100,400), b(200,200)
  forall (i=1:100, j=1:200)
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test2_forall_construct

!*** Test forall with multiple assignment statements and mask
! CHECK-LABEL: func @_QPtest3_forall_construct(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>>,
! CHECK-SAME: %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>>,
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<!fir.array<100x200x!fir.logical<4>>>) {
subroutine test3_forall_construct(a,b, mask)
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 100 : index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 400 : index
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 200 : index
  ! CHECK:         %[[VAL_10:.*]] = arith.constant 200 : index
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 100 : i32
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
  ! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
  ! CHECK:         %[[VAL_18:.*]] = arith.constant 200 : i32
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK:         %[[VAL_20:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_0]](%[[VAL_21]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_1]](%[[VAL_23]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_26:.*]] = fir.array_load %[[VAL_1]](%[[VAL_25]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_27:.*]] = fir.do_loop %[[VAL_28:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_15]] unordered iter_args(%[[VAL_29:.*]] = %[[VAL_22]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_30]] to %[[VAL_6]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_20]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_29]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_32]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_34]] to %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_35:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> i64
  ! CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_38:.*]] = arith.subi %[[VAL_36]], %[[VAL_37]] : i64
  ! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
  ! CHECK:             %[[VAL_41:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_40]], %[[VAL_41]] : i64
  ! CHECK:             %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_38]], %[[VAL_42]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_46:.*]] = fir.if %[[VAL_45]] -> (!fir.array<100x400xf32>) {
  ! CHECK:               %[[VAL_47:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
  ! CHECK:               %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i64) -> index
  ! CHECK:               %[[VAL_50:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i32) -> i64
  ! CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i64) -> index
  ! CHECK:               %[[VAL_53:.*]] = fir.array_fetch %[[VAL_24]], %[[VAL_49]], %[[VAL_52]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_54:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_55:.*]] = arith.constant 1 : i32
  ! CHECK:               %[[VAL_56:.*]] = arith.addi %[[VAL_54]], %[[VAL_55]] : i32
  ! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i64
  ! CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:               %[[VAL_59:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
  ! CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i64) -> index
  ! CHECK:               %[[VAL_62:.*]] = fir.array_fetch %[[VAL_26]], %[[VAL_58]], %[[VAL_61]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_63:.*]] = arith.addf %[[VAL_53]], %[[VAL_62]] : f32
  ! CHECK:               %[[VAL_64:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i32) -> i64
  ! CHECK:               %[[VAL_66:.*]] = fir.convert %[[VAL_65]] : (i64) -> index
  ! CHECK:               %[[VAL_67:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (i32) -> i64
  ! CHECK:               %[[VAL_69:.*]] = fir.convert %[[VAL_68]] : (i64) -> index
  ! CHECK:               %[[VAL_70:.*]] = fir.array_update %[[VAL_33]], %[[VAL_63]], %[[VAL_66]], %[[VAL_69]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:               fir.result %[[VAL_70]] : !fir.array<100x400xf32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_33]] : !fir.array<100x400xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_71:.*]] : !fir.array<100x400xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_72:.*]] : !fir.array<100x400xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_22]], %[[VAL_73:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK:         %[[VAL_74:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_75:.*]] = fir.array_load %[[VAL_0]](%[[VAL_74]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_76:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_77:.*]] = fir.array_load %[[VAL_1]](%[[VAL_76]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_78:.*]] = fir.do_loop %[[VAL_79:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_15]] unordered iter_args(%[[VAL_80:.*]] = %[[VAL_75]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:           %[[VAL_81:.*]] = fir.convert %[[VAL_79]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_81]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_82:.*]] = fir.do_loop %[[VAL_83:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_20]] unordered iter_args(%[[VAL_84:.*]] = %[[VAL_80]]) -> (!fir.array<100x400xf32>) {
  ! CHECK:             %[[VAL_85:.*]] = fir.convert %[[VAL_83]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_85]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_86:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_87:.*]] = fir.convert %[[VAL_86]] : (i32) -> i64
  ! CHECK:             %[[VAL_88:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_89:.*]] = arith.subi %[[VAL_87]], %[[VAL_88]] : i64
  ! CHECK:             %[[VAL_90:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_91:.*]] = fir.convert %[[VAL_90]] : (i32) -> i64
  ! CHECK:             %[[VAL_92:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_93:.*]] = arith.subi %[[VAL_91]], %[[VAL_92]] : i64
  ! CHECK:             %[[VAL_94:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_89]], %[[VAL_93]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_95:.*]] = fir.load %[[VAL_94]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_96:.*]] = fir.convert %[[VAL_95]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_97:.*]] = fir.if %[[VAL_96]] -> (!fir.array<100x400xf32>) {
  ! CHECK:               %[[VAL_98:.*]] = arith.constant 1.000000e+00 : f32
  ! CHECK:               %[[VAL_99:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_100:.*]] = fir.convert %[[VAL_99]] : (i32) -> i64
  ! CHECK:               %[[VAL_101:.*]] = fir.convert %[[VAL_100]] : (i64) -> index
  ! CHECK:               %[[VAL_102:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_103:.*]] = fir.convert %[[VAL_102]] : (i32) -> i64
  ! CHECK:               %[[VAL_104:.*]] = fir.convert %[[VAL_103]] : (i64) -> index
  ! CHECK:               %[[VAL_105:.*]] = fir.array_fetch %[[VAL_77]], %[[VAL_101]], %[[VAL_104]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_106:.*]] = divf %[[VAL_98]], %[[VAL_105]] : f32
  ! CHECK:               %[[VAL_107:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_108:.*]] = fir.convert %[[VAL_107]] : (i32) -> i64
  ! CHECK:               %[[VAL_109:.*]] = fir.convert %[[VAL_108]] : (i64) -> index
  ! CHECK:               %[[VAL_110:.*]] = arith.constant 200 : i32
  ! CHECK:               %[[VAL_111:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_112:.*]] = arith.addi %[[VAL_110]], %[[VAL_111]] : i32
  ! CHECK:               %[[VAL_113:.*]] = fir.convert %[[VAL_112]] : (i32) -> i64
  ! CHECK:               %[[VAL_114:.*]] = fir.convert %[[VAL_113]] : (i64) -> index
  ! CHECK:               %[[VAL_115:.*]] = fir.array_update %[[VAL_84]], %[[VAL_106]], %[[VAL_109]], %[[VAL_114]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:               fir.result %[[VAL_115]] : !fir.array<100x400xf32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_84]] : !fir.array<100x400xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_116:.*]] : !fir.array<100x400xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_117:.*]] : !fir.array<100x400xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_75]], %[[VAL_118:.*]] to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>

  real :: a(100,400), b(200,200)
  logical :: mask(100,200)
  forall (i=1:100, j=1:200, mask(i,j))
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test3_forall_construct

!*** Test a FORALL construct with an array assignment
!    This is similar to the following embedded WHERE construct test, but the
!    elements are assigned unconditionally.
! CHECK-LABEL: func @_QPtest_forall_with_array_assignment(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>) {
subroutine test_forall_with_array_assignment(aa,bb)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_19:.*]] = arith.constant 64 : i64
  ! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
  ! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_22:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_20]], %[[VAL_21]] : index
  ! CHECK:           %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_17]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_28:.*]] = arith.constant 1 : i32
  ! CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_27]], %[[VAL_28]] : i32
  ! CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
  ! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
  ! CHECK:             %[[VAL_32:.*]] = fir.field_index block2, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK:             %[[VAL_33:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_25]], %[[VAL_33]] : index
  ! CHECK:             %[[VAL_35:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_31]], %[[VAL_32]], %[[VAL_34]] {Fortran.offsets} : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, index, !fir.field, index) -> i64
  ! CHECK:             %[[VAL_36:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
  ! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
  ! CHECK:             %[[VAL_39:.*]] = fir.field_index block1, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK:             %[[VAL_40:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_25]], %[[VAL_40]] : index
  ! CHECK:             %[[VAL_42:.*]] = fir.array_update %[[VAL_26]], %[[VAL_35]], %[[VAL_38]], %[[VAL_39]], %[[VAL_41]] {Fortran.offsets} : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, i64, index, !fir.field, index) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:             fir.result %[[VAL_42]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_43:.*]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_44:.*]] to %[[VAL_0]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>

  type t
     integer(kind=8) :: block1(64)
     integer(kind=8) :: block2(64)
  end type t
  type(t) :: aa(10), bb(10)

  forall (i=1:10:2)
     aa(i)%block1 = bb(i+1)%block2
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test_forall_with_array_assignment

!*** Test a FORALL construct with a nested WHERE construct.
!    This has both an explicit and implicit iteration space. The WHERE construct
!    makes the assignments conditional and the where mask evaluation must happen
!    prior to evaluating the array assignment statement.
! CHECK-LABEL: func @_QPtest_nested_forall_where(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) {
subroutine test_nested_forall_where(a,b)  
  ! CHECK: %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK: %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK: %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK: %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK: %[[VAL_6:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK: %[[VAL_7:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK: %[[VAL_8:.*]] = fir.alloca tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>
  ! CHECK: %[[VAL_9:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_10:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_9]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<i64>
  ! CHECK: fir.store %[[VAL_10]] to %[[VAL_11]] : !fir.ref<i64>
  ! CHECK: %[[VAL_12:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_13:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi8>>
  ! CHECK: %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_12]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: fir.store %[[VAL_13]] to %[[VAL_14]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_15:.*]] = arith.constant 2 : i32
  ! CHECK: %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi64>>
  ! CHECK: %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_15]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK: fir.store %[[VAL_16]] to %[[VAL_17]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK: %[[VAL_18:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK: %[[VAL_20:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_20]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_21]]#1 : (index) -> i64
  ! CHECK: %[[VAL_23:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
  ! CHECK: %[[VAL_25:.*]] = arith.addi %[[VAL_22]], %[[VAL_24]] : i64
  ! CHECK: %[[VAL_26:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_26]] : i64
  ! CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> i32
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> index
  ! CHECK: %[[VAL_30:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_31:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
  ! CHECK: %[[VAL_33:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_34:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_33]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_34]]#1 : (index) -> i64
  ! CHECK: %[[VAL_36:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (index) -> i64
  ! CHECK: %[[VAL_38:.*]] = arith.addi %[[VAL_35]], %[[VAL_37]] : i64
  ! CHECK: %[[VAL_39:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_39]] : i64
  ! CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i64) -> i32
  ! CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
  ! CHECK: %[[VAL_43:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_44:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[VAL_45:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %[[VAL_19]] to %[[VAL_29]] step %[[VAL_30]] unordered iter_args(%[[VAL_48:.*]] = %[[VAL_44]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_49]] to %[[VAL_7]] : !fir.ref<i32>
  ! CHECK: %[[VAL_50:.*]] = fir.do_loop %[[VAL_51:.*]] = %[[VAL_32]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_52:.*]] = %[[VAL_48]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_53:.*]] = fir.convert %[[VAL_51]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_53]] to %[[VAL_6]] : !fir.ref<i32>
  ! CHECK: %[[VAL_54:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_55:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_56:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
  ! CHECK: %[[VAL_57:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
  ! CHECK: %[[VAL_58:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK: %[[VAL_59:.*]] = arith.subi %[[VAL_57]], %[[VAL_56]] : i64
  ! CHECK: %[[VAL_60:.*]] = arith.addi %[[VAL_59]], %[[VAL_58]] : i64
  ! CHECK: %[[VAL_61:.*]] = arith.divsi %[[VAL_60]], %[[VAL_58]] : i64
  ! CHECK: %[[VAL_62:.*]] = arith.cmpi sgt, %[[VAL_61]], %[[VAL_55]] : i64
  ! CHECK: %[[VAL_63:.*]] = select %[[VAL_62]], %[[VAL_61]], %[[VAL_55]] : i64
  ! CHECK: %[[VAL_64:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_65:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK: %[[VAL_66:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
  ! CHECK: %[[VAL_67:.*]] = fir.convert %[[VAL_43]] : (index) -> i64
  ! CHECK: %[[VAL_68:.*]] = arith.subi %[[VAL_66]], %[[VAL_65]] : i64
  ! CHECK: %[[VAL_69:.*]] = arith.addi %[[VAL_68]], %[[VAL_67]] : i64
  ! CHECK: %[[VAL_70:.*]] = arith.divsi %[[VAL_69]], %[[VAL_67]] : i64
  ! CHECK: %[[VAL_71:.*]] = arith.cmpi sgt, %[[VAL_70]], %[[VAL_64]] : i64
  ! CHECK: %[[VAL_72:.*]] = select %[[VAL_71]], %[[VAL_70]], %[[VAL_64]] : i64
  ! CHECK: %[[VAL_73:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_74:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_73]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_75:.*]] = fir.load %[[VAL_74]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK: %[[VAL_77:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_78:.*]] = arith.cmpi eq, %[[VAL_76]], %[[VAL_77]] : i64
  ! CHECK: fir.if %[[VAL_78]] {
  ! CHECK: %[[VAL_79:.*]] = arith.constant true
  ! CHECK: %[[VAL_80:.*]] = arith.constant 2 : i64
  ! CHECK: %[[VAL_81:.*]] = fir.allocmem !fir.array<2xi64>
  ! CHECK: %[[VAL_82:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_83:.*]] = fir.coordinate_of %[[VAL_81]], %[[VAL_82]] : (!fir.heap<!fir.array<2xi64>>, i32) -> !fir.ref<i64>
  ! CHECK: fir.store %[[VAL_63]] to %[[VAL_83]] : !fir.ref<i64>
  ! CHECK: %[[VAL_84:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_85:.*]] = fir.coordinate_of %[[VAL_81]], %[[VAL_84]] : (!fir.heap<!fir.array<2xi64>>, i32) -> !fir.ref<i64>
  ! CHECK: fir.store %[[VAL_72]] to %[[VAL_85]] : !fir.ref<i64>
  ! CHECK: %[[VAL_86:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK: %[[VAL_87:.*]] = fir.convert %[[VAL_81]] : (!fir.heap<!fir.array<2xi64>>) -> !fir.ref<i64>
  ! CHECK: %[[VAL_88:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_86]], %[[VAL_79]], %[[VAL_80]], %[[VAL_54]], %[[VAL_87]]) : (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
  ! CHECK: }
  ! CHECK: %[[VAL_89:.*]] = arith.subi %[[VAL_47]], %[[VAL_19]] : index
  ! CHECK: %[[VAL_90:.*]] = arith.divsi %[[VAL_89]], %[[VAL_30]] : index
  ! CHECK: %[[VAL_91:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_92:.*]] = arith.addi %[[VAL_90]], %[[VAL_91]] : index
  ! CHECK: %[[VAL_93:.*]] = arith.subi %[[VAL_51]], %[[VAL_32]] : index
  ! CHECK: %[[VAL_94:.*]] = arith.divsi %[[VAL_93]], %[[VAL_43]] : index
  ! CHECK: %[[VAL_95:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_96:.*]] = arith.addi %[[VAL_94]], %[[VAL_95]] : index
  ! CHECK: %[[VAL_97:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_98:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_97]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_99:.*]] = fir.load %[[VAL_98]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_100:.*]] = fir.convert %[[VAL_99]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK: %[[VAL_101:.*]] = fir.shape %[[VAL_63]], %[[VAL_72]] : (i64, i64) -> !fir.shape<2>
  ! CHECK: %[[VAL_102:.*]] = fir.array_coor %[[VAL_100]](%[[VAL_101]]) %[[VAL_92]], %[[VAL_96]] : (!fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<2>, index, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[VAL_103:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
  ! CHECK: %[[VAL_104:.*]] = fir.convert %[[VAL_103]] : (i32) -> i64
  ! CHECK: %[[VAL_105:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_106:.*]] = arith.subi %[[VAL_104]], %[[VAL_105]] : i64
  ! CHECK: %[[VAL_107:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
  ! CHECK: %[[VAL_108:.*]] = fir.convert %[[VAL_107]] : (i32) -> i64
  ! CHECK: %[[VAL_109:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_110:.*]] = arith.subi %[[VAL_108]], %[[VAL_109]] : i64
  ! CHECK: %[[VAL_111:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_106]], %[[VAL_110]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, i64, i64) -> !fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[VAL_112:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[VAL_113:.*]] = fir.coordinate_of %[[VAL_111]], %[[VAL_112]] : (!fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK: %[[VAL_114:.*]] = arith.constant 100 : index
  ! CHECK: %[[VAL_115:.*]] = fir.shape %[[VAL_114]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_116:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_117:.*]] = fir.slice %[[VAL_116]], %[[VAL_114]], %[[VAL_116]] : (index, index, index) -> !fir.slice<1>
  ! CHECK: %[[VAL_118:.*]] = fir.array_load %[[VAL_113]](%[[VAL_115]]) {{\[}}%[[VAL_117]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK: %[[VAL_119:.*]] = arith.constant 0.000000e+00 : f32
  ! CHECK: %[[VAL_120:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_121:.*]] = arith.subi %[[VAL_114]], %[[VAL_116]] : index
  ! CHECK: %[[VAL_122:.*]] = arith.addi %[[VAL_121]], %[[VAL_116]] : index
  ! CHECK: %[[VAL_123:.*]] = arith.divsi %[[VAL_122]], %[[VAL_116]] : index
  ! CHECK: %[[VAL_124:.*]] = arith.cmpi sgt, %[[VAL_123]], %[[VAL_120]] : index
  ! CHECK: %[[VAL_125:.*]] = select %[[VAL_124]], %[[VAL_123]], %[[VAL_120]] : index
  ! CHECK: %[[VAL_126:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_127:.*]] = fir.coordinate_of %[[VAL_102]], %[[VAL_126]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_128:.*]] = fir.load %[[VAL_127]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_129:.*]] = fir.shape %[[VAL_125]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_130:.*]] = fir.array_load %[[VAL_128]](%[[VAL_129]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
  ! CHECK: %[[VAL_131:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_132:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_133:.*]] = fir.coordinate_of %[[VAL_102]], %[[VAL_132]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_134:.*]] = fir.load %[[VAL_133]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_135:.*]] = fir.convert %[[VAL_134]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK: %[[VAL_136:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_137:.*]] = arith.cmpi eq, %[[VAL_135]], %[[VAL_136]] : i64
  ! CHECK: fir.if %[[VAL_137]] {
  ! CHECK: %[[VAL_138:.*]] = arith.constant false
  ! CHECK: %[[VAL_139:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_140:.*]] = fir.allocmem !fir.array<1xi64>
  ! CHECK: %[[VAL_141:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_142:.*]] = fir.coordinate_of %[[VAL_140]], %[[VAL_141]] : (!fir.heap<!fir.array<1xi64>>, i32) -> !fir.ref<i64>
  ! CHECK: %[[VAL_143:.*]] = fir.convert %[[VAL_125]] : (index) -> i64
  ! CHECK: fir.store %[[VAL_143]] to %[[VAL_142]] : !fir.ref<i64>
  ! CHECK: %[[VAL_144:.*]] = fir.convert %[[VAL_102]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK: %[[VAL_145:.*]] = fir.convert %[[VAL_140]] : (!fir.heap<!fir.array<1xi64>>) -> !fir.ref<i64>
  ! CHECK: %[[VAL_146:.*]] = fir.call @_FortranARaggedArrayAllocate(%[[VAL_144]], %[[VAL_138]], %[[VAL_139]], %[[VAL_131]], %[[VAL_145]]) : (!fir.llvm_ptr<i8>, i1, i64, i64, !fir.ref<i64>) -> !fir.llvm_ptr<i8>
  ! CHECK: }
  ! CHECK: %[[VAL_147:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_148:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_149:.*]] = arith.subi %[[VAL_125]], %[[VAL_147]] : index
  ! CHECK: %[[VAL_150:.*]] = fir.do_loop %[[VAL_151:.*]] = %[[VAL_148]] to %[[VAL_149]] step %[[VAL_147]] unordered iter_args(%[[VAL_152:.*]] = %[[VAL_130]]) -> (!fir.array<?xi8>) {
  ! CHECK: %[[VAL_153:.*]] = fir.array_fetch %[[VAL_118]], %[[VAL_151]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK: %[[VAL_154:.*]] = arith.cmpf ogt, %[[VAL_153]], %[[VAL_119]] : f32
  ! CHECK: %[[VAL_155:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_156:.*]] = fir.coordinate_of %[[VAL_102]], %[[VAL_155]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_157:.*]] = fir.load %[[VAL_156]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_158:.*]] = fir.shape %[[VAL_125]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_159:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_160:.*]] = arith.addi %[[VAL_151]], %[[VAL_159]] : index
  ! CHECK: %[[VAL_161:.*]] = fir.array_coor %[[VAL_157]](%[[VAL_158]]) %[[VAL_160]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK: %[[VAL_162:.*]] = fir.convert %[[VAL_154]] : (i1) -> i8
  ! CHECK: fir.store %[[VAL_162]] to %[[VAL_161]] : !fir.ref<i8>
  ! CHECK: fir.result %[[VAL_152]] : !fir.array<?xi8>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_52]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_163:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: %[[VAL_164:.*]] = fir.do_loop %[[VAL_165:.*]] = %[[VAL_19]] to %[[VAL_29]] step %[[VAL_30]] unordered iter_args(%[[VAL_166:.*]] = %[[VAL_44]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_167:.*]] = fir.convert %[[VAL_165]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_167]] to %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[VAL_168:.*]] = fir.do_loop %[[VAL_169:.*]] = %[[VAL_32]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_170:.*]] = %[[VAL_166]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_171:.*]] = fir.convert %[[VAL_169]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_171]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[VAL_172:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_173:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
  ! CHECK: %[[VAL_174:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
  ! CHECK: %[[VAL_175:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK: %[[VAL_176:.*]] = arith.subi %[[VAL_174]], %[[VAL_173]] : i64
  ! CHECK: %[[VAL_177:.*]] = arith.addi %[[VAL_176]], %[[VAL_175]] : i64
  ! CHECK: %[[VAL_178:.*]] = arith.divsi %[[VAL_177]], %[[VAL_175]] : i64
  ! CHECK: %[[VAL_179:.*]] = arith.cmpi sgt, %[[VAL_178]], %[[VAL_172]] : i64
  ! CHECK: %[[VAL_180:.*]] = select %[[VAL_179]], %[[VAL_178]], %[[VAL_172]] : i64
  ! CHECK: %[[VAL_181:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_182:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK: %[[VAL_183:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
  ! CHECK: %[[VAL_184:.*]] = fir.convert %[[VAL_43]] : (index) -> i64
  ! CHECK: %[[VAL_185:.*]] = arith.subi %[[VAL_183]], %[[VAL_182]] : i64
  ! CHECK: %[[VAL_186:.*]] = arith.addi %[[VAL_185]], %[[VAL_184]] : i64
  ! CHECK: %[[VAL_187:.*]] = arith.divsi %[[VAL_186]], %[[VAL_184]] : i64
  ! CHECK: %[[VAL_188:.*]] = arith.cmpi sgt, %[[VAL_187]], %[[VAL_181]] : i64
  ! CHECK: %[[VAL_189:.*]] = select %[[VAL_188]], %[[VAL_187]], %[[VAL_181]] : i64
  ! CHECK: %[[VAL_190:.*]] = arith.subi %[[VAL_165]], %[[VAL_19]] : index
  ! CHECK: %[[VAL_191:.*]] = arith.divsi %[[VAL_190]], %[[VAL_30]] : index
  ! CHECK: %[[VAL_192:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_193:.*]] = arith.addi %[[VAL_191]], %[[VAL_192]] : index
  ! CHECK: %[[VAL_194:.*]] = arith.subi %[[VAL_169]], %[[VAL_32]] : index
  ! CHECK: %[[VAL_195:.*]] = arith.divsi %[[VAL_194]], %[[VAL_43]] : index
  ! CHECK: %[[VAL_196:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_197:.*]] = arith.addi %[[VAL_195]], %[[VAL_196]] : index
  ! CHECK: %[[VAL_198:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_199:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_198]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_200:.*]] = fir.load %[[VAL_199]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_201:.*]] = fir.convert %[[VAL_200]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK: %[[VAL_202:.*]] = fir.shape %[[VAL_180]], %[[VAL_189]] : (i64, i64) -> !fir.shape<2>
  ! CHECK: %[[VAL_203:.*]] = fir.array_coor %[[VAL_201]](%[[VAL_202]]) %[[VAL_193]], %[[VAL_197]] : (!fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<2>, index, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[VAL_204:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_205:.*]] = fir.coordinate_of %[[VAL_203]], %[[VAL_204]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_206:.*]] = fir.load %[[VAL_205]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_207:.*]] = fir.convert %[[VAL_206]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK: %[[VAL_208:.*]] = arith.constant 2 : i32
  ! CHECK: %[[VAL_209:.*]] = fir.coordinate_of %[[VAL_203]], %[[VAL_208]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK: %[[VAL_210:.*]] = fir.load %[[VAL_209]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK: %[[VAL_211:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_212:.*]] = fir.coordinate_of %[[VAL_210]], %[[VAL_211]] : (!fir.heap<!fir.array<?xi64>>, i32) -> !fir.ref<i64>
  ! CHECK: %[[VAL_213:.*]] = fir.load %[[VAL_212]] : !fir.ref<i64>
  ! CHECK: %[[VAL_214:.*]] = fir.convert %[[VAL_213]] : (i64) -> index
  ! CHECK: %[[VAL_215:.*]] = fir.shape %[[VAL_214]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_216:.*]] = arith.constant 100 : i64
  ! CHECK: %[[VAL_217:.*]] = fir.convert %[[VAL_216]] : (i64) -> index
  ! CHECK: %[[VAL_218:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_219:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_220:.*]] = arith.subi %[[VAL_217]], %[[VAL_218]] : index
  ! CHECK: %[[VAL_221:.*]] = fir.do_loop %[[VAL_222:.*]] = %[[VAL_219]] to %[[VAL_220]] step %[[VAL_218]] unordered iter_args(%[[VAL_223:.*]] = %[[VAL_170]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_224:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_225:.*]] = arith.addi %[[VAL_222]], %[[VAL_224]] : index
  ! CHECK: %[[VAL_226:.*]] = fir.array_coor %[[VAL_207]](%[[VAL_215]]) %[[VAL_225]] : (!fir.ref<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK: %[[VAL_227:.*]] = fir.load %[[VAL_226]] : !fir.ref<i8>
  ! CHECK: %[[VAL_228:.*]] = fir.convert %[[VAL_227]] : (i8) -> i1
  ! CHECK: %[[VAL_229:.*]] = fir.if %[[VAL_228]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_230:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[VAL_231:.*]] = fir.convert %[[VAL_230]] : (i32) -> i64
  ! CHECK: %[[VAL_232:.*]] = fir.convert %[[VAL_231]] : (i64) -> index
  ! CHECK: %[[VAL_233:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[VAL_234:.*]] = fir.convert %[[VAL_233]] : (i32) -> i64
  ! CHECK: %[[VAL_235:.*]] = fir.convert %[[VAL_234]] : (i64) -> index
  ! CHECK: %[[VAL_236:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[VAL_237:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_238:.*]] = arith.addi %[[VAL_222]], %[[VAL_237]] : index
  ! CHECK: %[[VAL_239:.*]] = fir.array_fetch %[[VAL_45]], %[[VAL_232]], %[[VAL_235]], %[[VAL_236]], %[[VAL_238]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
  ! CHECK: %[[VAL_240:.*]] = arith.constant 3.140000e+00 : f32
  ! CHECK: %[[VAL_241:.*]] = divf %[[VAL_239]], %[[VAL_240]] : f32
  ! CHECK: %[[VAL_242:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK: %[[VAL_243:.*]] = fir.convert %[[VAL_242]] : (i32) -> i64
  ! CHECK: %[[VAL_244:.*]] = fir.convert %[[VAL_243]] : (i64) -> index
  ! CHECK: %[[VAL_245:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK: %[[VAL_246:.*]] = fir.convert %[[VAL_245]] : (i32) -> i64
  ! CHECK: %[[VAL_247:.*]] = fir.convert %[[VAL_246]] : (i64) -> index
  ! CHECK: %[[VAL_248:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[VAL_249:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_250:.*]] = arith.addi %[[VAL_222]], %[[VAL_249]] : index
  ! CHECK: %[[VAL_251:.*]] = fir.array_update %[[VAL_223]], %[[VAL_241]], %[[VAL_244]], %[[VAL_247]], %[[VAL_248]], %[[VAL_250]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: fir.result %[[VAL_251]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: } else {
  ! CHECK: fir.result %[[VAL_223]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_252:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_253:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_254:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_44]], %[[VAL_255:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  ! CHECK: %[[VAL_256:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[VAL_257:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: %[[VAL_258:.*]] = fir.do_loop %[[VAL_259:.*]] = %[[VAL_19]] to %[[VAL_29]] step %[[VAL_30]] unordered iter_args(%[[VAL_260:.*]] = %[[VAL_256]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_261:.*]] = fir.convert %[[VAL_259]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_261]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_262:.*]] = fir.do_loop %[[VAL_263:.*]] = %[[VAL_32]] to %[[VAL_42]] step %[[VAL_43]] unordered iter_args(%[[VAL_264:.*]] = %[[VAL_260]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_265:.*]] = fir.convert %[[VAL_263]] : (index) -> i32
  ! CHECK: fir.store %[[VAL_265]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_266:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_267:.*]] = fir.convert %[[VAL_19]] : (index) -> i64
  ! CHECK: %[[VAL_268:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
  ! CHECK: %[[VAL_269:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
  ! CHECK: %[[VAL_270:.*]] = arith.subi %[[VAL_268]], %[[VAL_267]] : i64
  ! CHECK: %[[VAL_271:.*]] = arith.addi %[[VAL_270]], %[[VAL_269]] : i64
  ! CHECK: %[[VAL_272:.*]] = arith.divsi %[[VAL_271]], %[[VAL_269]] : i64
  ! CHECK: %[[VAL_273:.*]] = arith.cmpi sgt, %[[VAL_272]], %[[VAL_266]] : i64
  ! CHECK: %[[VAL_274:.*]] = select %[[VAL_273]], %[[VAL_272]], %[[VAL_266]] : i64
  ! CHECK: %[[VAL_275:.*]] = arith.constant 0 : i64
  ! CHECK: %[[VAL_276:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
  ! CHECK: %[[VAL_277:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
  ! CHECK: %[[VAL_278:.*]] = fir.convert %[[VAL_43]] : (index) -> i64
  ! CHECK: %[[VAL_279:.*]] = arith.subi %[[VAL_277]], %[[VAL_276]] : i64
  ! CHECK: %[[VAL_280:.*]] = arith.addi %[[VAL_279]], %[[VAL_278]] : i64
  ! CHECK: %[[VAL_281:.*]] = arith.divsi %[[VAL_280]], %[[VAL_278]] : i64
  ! CHECK: %[[VAL_282:.*]] = arith.cmpi sgt, %[[VAL_281]], %[[VAL_275]] : i64
  ! CHECK: %[[VAL_283:.*]] = select %[[VAL_282]], %[[VAL_281]], %[[VAL_275]] : i64
  ! CHECK: %[[VAL_284:.*]] = arith.subi %[[VAL_259]], %[[VAL_19]] : index
  ! CHECK: %[[VAL_285:.*]] = arith.divsi %[[VAL_284]], %[[VAL_30]] : index
  ! CHECK: %[[VAL_286:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_287:.*]] = arith.addi %[[VAL_285]], %[[VAL_286]] : index
  ! CHECK: %[[VAL_288:.*]] = arith.subi %[[VAL_263]], %[[VAL_32]] : index
  ! CHECK: %[[VAL_289:.*]] = arith.divsi %[[VAL_288]], %[[VAL_43]] : index
  ! CHECK: %[[VAL_290:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_291:.*]] = arith.addi %[[VAL_289]], %[[VAL_290]] : index
  ! CHECK: %[[VAL_292:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_293:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_292]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_294:.*]] = fir.load %[[VAL_293]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_295:.*]] = fir.convert %[[VAL_294]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>
  ! CHECK: %[[VAL_296:.*]] = fir.shape %[[VAL_274]], %[[VAL_283]] : (i64, i64) -> !fir.shape<2>
  ! CHECK: %[[VAL_297:.*]] = fir.array_coor %[[VAL_295]](%[[VAL_296]]) %[[VAL_287]], %[[VAL_291]] : (!fir.ref<!fir.array<?x?xtuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>>, !fir.shape<2>, index, index) -> !fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>
  ! CHECK: %[[VAL_298:.*]] = arith.constant 1 : i32
  ! CHECK: %[[VAL_299:.*]] = fir.coordinate_of %[[VAL_297]], %[[VAL_298]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_300:.*]] = fir.load %[[VAL_299]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK: %[[VAL_301:.*]] = fir.convert %[[VAL_300]] : (!fir.heap<!fir.array<?xi8>>) -> !fir.ref<!fir.array<?xi8>>
  ! CHECK: %[[VAL_302:.*]] = arith.constant 2 : i32
  ! CHECK: %[[VAL_303:.*]] = fir.coordinate_of %[[VAL_297]], %[[VAL_302]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>, i32) -> !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK: %[[VAL_304:.*]] = fir.load %[[VAL_303]] : !fir.ref<!fir.heap<!fir.array<?xi64>>>
  ! CHECK: %[[VAL_305:.*]] = arith.constant 0 : i32
  ! CHECK: %[[VAL_306:.*]] = fir.coordinate_of %[[VAL_304]], %[[VAL_305]] : (!fir.heap<!fir.array<?xi64>>, i32) -> !fir.ref<i64>
  ! CHECK: %[[VAL_307:.*]] = fir.load %[[VAL_306]] : !fir.ref<i64>
  ! CHECK: %[[VAL_308:.*]] = fir.convert %[[VAL_307]] : (i64) -> index
  ! CHECK: %[[VAL_309:.*]] = fir.shape %[[VAL_308]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_310:.*]] = arith.constant 100 : i64
  ! CHECK: %[[VAL_311:.*]] = fir.convert %[[VAL_310]] : (i64) -> index
  ! CHECK: %[[VAL_312:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_313:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_314:.*]] = arith.subi %[[VAL_311]], %[[VAL_312]] : index
  ! CHECK: %[[VAL_315:.*]] = fir.do_loop %[[VAL_316:.*]] = %[[VAL_313]] to %[[VAL_314]] step %[[VAL_312]] unordered iter_args(%[[VAL_317:.*]] = %[[VAL_264]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: %[[VAL_318:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_319:.*]] = arith.addi %[[VAL_316]], %[[VAL_318]] : index
  ! CHECK: %[[VAL_320:.*]] = fir.array_coor %[[VAL_301]](%[[VAL_309]]) %[[VAL_319]] : (!fir.ref<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK: %[[VAL_321:.*]] = fir.load %[[VAL_320]] : !fir.ref<i8>
  ! CHECK: %[[VAL_322:.*]] = fir.convert %[[VAL_321]] : (i8) -> i1
  ! CHECK: %[[VAL_323:.*]] = fir.if %[[VAL_322]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK: fir.result %[[VAL_317]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: } else {
  ! CHECK: %[[VAL_324:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_325:.*]] = fir.convert %[[VAL_324]] : (i32) -> i64
  ! CHECK: %[[VAL_326:.*]] = fir.convert %[[VAL_325]] : (i64) -> index
  ! CHECK: %[[VAL_327:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_328:.*]] = fir.convert %[[VAL_327]] : (i32) -> i64
  ! CHECK: %[[VAL_329:.*]] = fir.convert %[[VAL_328]] : (i64) -> index
  ! CHECK: %[[VAL_330:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[VAL_331:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_332:.*]] = arith.addi %[[VAL_316]], %[[VAL_331]] : index
  ! CHECK: %[[VAL_333:.*]] = fir.array_fetch %[[VAL_257]], %[[VAL_326]], %[[VAL_329]], %[[VAL_330]], %[[VAL_332]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
  ! CHECK: %[[VAL_334:.*]] = arith.negf %[[VAL_333]] : f32
  ! CHECK: %[[VAL_335:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_336:.*]] = fir.convert %[[VAL_335]] : (i32) -> i64
  ! CHECK: %[[VAL_337:.*]] = fir.convert %[[VAL_336]] : (i64) -> index
  ! CHECK: %[[VAL_338:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_339:.*]] = fir.convert %[[VAL_338]] : (i32) -> i64
  ! CHECK: %[[VAL_340:.*]] = fir.convert %[[VAL_339]] : (i64) -> index
  ! CHECK: %[[VAL_341:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK: %[[VAL_342:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_343:.*]] = arith.addi %[[VAL_316]], %[[VAL_342]] : index
  ! CHECK: %[[VAL_344:.*]] = fir.array_update %[[VAL_317]], %[[VAL_334]], %[[VAL_337]], %[[VAL_340]], %[[VAL_341]], %[[VAL_343]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: fir.result %[[VAL_344]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_345:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_346:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_347:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_256]], %[[VAL_348:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  ! CHECK: %[[VAL_349:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<tuple<i64, !fir.heap<!fir.array<?xi8>>, !fir.heap<!fir.array<?xi64>>>>) -> !fir.llvm_ptr<i8>
  ! CHECK: %[[VAL_350:.*]] = fir.call @_FortranARaggedArrayDeallocate(%[[VAL_349]]) : (!fir.llvm_ptr<i8>) -> none
  type t
     real data(100)
  end type t
  type(t) :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2))
     where (b(j,i)%data > 0.0)
        a(i,j)%data = b(j,i)%data / 3.14
     elsewhere
        a(i,j)%data = -b(j,i)%data
     end where
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test_nested_forall_where

! CHECK-LABEL: func @_QPtest_forall_with_slice(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>) {
subroutine test_forall_with_slice(i1,i2)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_sliceEa"}
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 5 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
  ! CHECK:         %[[VAL_14:.*]] = arith.constant 10 : i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
  ! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_6]](%[[VAL_17]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] unordered iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_22]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_16]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_21]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_26]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
  ! CHECK:             %[[VAL_29:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
  ! CHECK:             %[[VAL_31:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK:             %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK:             %[[VAL_34:.*]] = arith.constant 0 : index
  ! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
  ! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
  ! CHECK:             %[[VAL_37:.*]] = arith.subi %[[VAL_36]], %[[VAL_35]] : index
  ! CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_33]] : index
  ! CHECK:             %[[VAL_39:.*]] = arith.divsi %[[VAL_38]], %[[VAL_33]] : index
  ! CHECK:             %[[VAL_40:.*]] = arith.cmpi sgt, %[[VAL_39]], %[[VAL_34]] : index
  ! CHECK:             %[[VAL_41:.*]] = select %[[VAL_40]], %[[VAL_39]], %[[VAL_34]] : index
  ! CHECK:             %[[VAL_42:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_43:.*]] = arith.constant 0 : index
  ! CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_41]], %[[VAL_42]] : index
  ! CHECK:             %[[VAL_45:.*]] = fir.do_loop %[[VAL_46:.*]] = %[[VAL_43]] to %[[VAL_44]] step %[[VAL_42]] unordered iter_args(%[[VAL_47:.*]] = %[[VAL_25]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:               %[[VAL_48:.*]] = fir.call @_QPf(%[[VAL_3]]) : (!fir.ref<i32>) -> i32
  ! CHECK:               %[[VAL_49:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> i64
  ! CHECK:               %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i64) -> index
  ! CHECK:               %[[VAL_52:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i32) -> i64
  ! CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i64) -> index
  ! CHECK:               %[[VAL_55:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>
  ! CHECK:               %[[VAL_56:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i64
  ! CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:               %[[VAL_59:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
  ! CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i64) -> index
  ! CHECK:               %[[VAL_62:.*]] = arith.muli %[[VAL_46]], %[[VAL_61]] : index
  ! CHECK:               %[[VAL_63:.*]] = arith.addi %[[VAL_58]], %[[VAL_62]] : index
  ! CHECK:               %[[VAL_64:.*]] = fir.array_update %[[VAL_47]], %[[VAL_48]], %[[VAL_51]], %[[VAL_54]], %[[VAL_55]], %[[VAL_63]] {Fortran.offsets} : (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:               fir.result %[[VAL_64]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_65:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_66:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_67:.*]] to %[[VAL_6]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>

  interface
     pure integer function f(i)
       integer i
       intent(in) i
     end function f
  end interface
  type t
     !integer :: arr(5:15)
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)

  forall (i=1:5, j=1:10)
     a(i,j)%arr(i:i1:i2) = f(i)
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test_forall_with_slice

! CHECK-LABEL: func @_QPtest_forall_with_ranked_dimension() {
subroutine test_forall_with_ranked_dimension
  ! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_ranked_dimensionEa"}
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 5 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_2]] : index
  ! CHECK:           %[[VAL_17:.*]] = arith.subi %[[VAL_16]], %[[VAL_15]] : index
  ! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : i64
  ! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_17]], %[[VAL_15]] : index
  ! CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_23:.*]] = arith.divsi %[[VAL_22]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_20]] : index
  ! CHECK:           %[[VAL_25:.*]] = select %[[VAL_24]], %[[VAL_23]], %[[VAL_20]] : index
  ! CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
  ! CHECK:           %[[VAL_27:.*]] = arith.constant 0 : index
  ! CHECK:           %[[VAL_28:.*]] = arith.subi %[[VAL_25]], %[[VAL_26]] : index
  ! CHECK:           %[[VAL_29:.*]] = fir.do_loop %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_28]] step %[[VAL_26]] unordered iter_args(%[[VAL_31:.*]] = %[[VAL_13]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:             %[[VAL_32:.*]] = fir.call @_QPf(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  ! CHECK:             %[[VAL_33:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
  ! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
  ! CHECK:             %[[VAL_36:.*]] = arith.constant 1 : index
  ! CHECK:             %[[VAL_37:.*]] = arith.constant 1 : i64
  ! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
  ! CHECK:             %[[VAL_39:.*]] = arith.muli %[[VAL_30]], %[[VAL_38]] : index
  ! CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_36]], %[[VAL_39]] : index
  ! CHECK:             %[[VAL_41:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>
  ! CHECK:             %[[VAL_42:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_43:.*]] = arith.constant 4 : i32
  ! CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_42]], %[[VAL_43]] : i32
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
  ! CHECK:             %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i64) -> index
  ! CHECK:             %[[VAL_47:.*]] = fir.array_update %[[VAL_31]], %[[VAL_32]], %[[VAL_35]], %[[VAL_40]], %[[VAL_41]], %[[VAL_46]] {Fortran.offsets} : (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:             fir.result %[[VAL_47]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_48:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_49:.*]] to %[[VAL_3]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>

  interface
     pure integer function f(i)
       integer, intent(in) :: i
     end function f
  end interface
  type t
     !integer :: arr(5:15)
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)
  
  forall (i=1:5)
     a(i,:)%arr(i+4) = f(i)
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine test_forall_with_ranked_dimension

! CHECK-LABEL: func @_QPforall_with_allocatable(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine forall_with_allocatable(a1)
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "arr", uniq_name = "_QFforall_with_allocatableEarr"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QFforall_with_allocatableEarr.addr"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.lb0"}
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.ext0"}
  ! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  ! CHECK:         fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 5 : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 15 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
  ! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
  ! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:         %[[VAL_15:.*]] = fir.shape_shift %[[VAL_12]], %[[VAL_13]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_14]](%[[VAL_15]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_16]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_21]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
  ! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
  ! CHECK:           %[[VAL_25:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_24]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
  ! CHECK:           %[[VAL_29:.*]] = fir.array_update %[[VAL_20]], %[[VAL_25]], %[[VAL_28]] {Fortran.offsets} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_29]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_16]], %[[VAL_30:.*]] to %[[VAL_14]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
  
  real :: a1(:)
  real, allocatable :: arr(:)
  forall (i=5:15)
     arr(i) = a1(i)
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine forall_with_allocatable

! CHECK-LABEL: func @_QPforall_with_allocatable2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine forall_with_allocatable2(a1)
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {bindc_name = "thing", uniq_name = "_QFforall_with_allocatable2Ething"}
  ! CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_4:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_5:.*]] = arith.constant {{.*}} : i32
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAInitialize(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 5 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 15 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_14:.*]] = fir.field_index arr, !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_14]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_17:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_18:.*]]:3 = fir.box_dims %[[VAL_16]], %[[VAL_17]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_19:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK:         %[[VAL_20:.*]] = fir.shape_shift %[[VAL_18]]#0, %[[VAL_18]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_19]](%[[VAL_20]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_22:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_13]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_21]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_26]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
  ! CHECK:           %[[VAL_30:.*]] = fir.array_fetch %[[VAL_22]], %[[VAL_29]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK:           %[[VAL_34:.*]] = fir.array_update %[[VAL_25]], %[[VAL_30]], %[[VAL_33]] {Fortran.offsets} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_34]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_35:.*]] to %[[VAL_19]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
  
  real :: a1(:)
  type t
     integer :: i
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing
  forall (i=5:15)
     thing%arr(i) = a1(i)
  end forall
  ! CHECK: return
  ! CHECK: }
end subroutine forall_with_allocatable2
