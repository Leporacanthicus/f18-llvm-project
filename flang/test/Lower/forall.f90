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
  ! CHECK:         %[[VAL_10:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
  ! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_13:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_14:.*]] = subi %[[VAL_12]], %[[VAL_13]] : i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
  ! CHECK:         %[[VAL_16:.*]] = constant 1 : index
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
  ! CHECK:           %[[VAL_35:.*]] = addf %[[VAL_30]], %[[VAL_34]] : f32
  ! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_37:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_38:.*]] = addi %[[VAL_36]], %[[VAL_37]] : i32
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
  ! CHECK:         %[[VAL_3:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = constant 100 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> !fir.array<200xf32>
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<200xf32>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
  ! CHECK:           %[[VAL_17:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_18:.*]] = subi %[[VAL_16]], %[[VAL_17]] : i64
  ! CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_18]] : (!fir.ref<!fir.array<200x!fir.logical<4>>>, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<!fir.logical<4>>
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.logical<4>) -> i1
  ! CHECK:           %[[VAL_22:.*]] = fir.if %[[VAL_21]] -> (!fir.array<200xf32>) {
  ! CHECK:             %[[VAL_23:.*]] = constant 1.000000e+00 : f32
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
  ! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_9:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
  ! CHECK:         %[[VAL_11:.*]] = addi %[[VAL_8]], %[[VAL_10]] : i64
  ! CHECK:         %[[VAL_12:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_13:.*]] = subi %[[VAL_11]], %[[VAL_12]] : i64
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
  ! CHECK:         %[[VAL_16:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_17:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK:         %[[VAL_19:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_19]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (index) -> i64
  ! CHECK:         %[[VAL_24:.*]] = addi %[[VAL_21]], %[[VAL_23]] : i64
  ! CHECK:         %[[VAL_25:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_26:.*]] = subi %[[VAL_24]], %[[VAL_25]] : i64
  ! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> i32
  ! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
  ! CHECK:         %[[VAL_29:.*]] = constant 1 : index
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
  ! CHECK:             %[[VAL_42:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_43:.*]] = subi %[[VAL_41]], %[[VAL_42]] : i64
  ! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
  ! CHECK:             %[[VAL_46:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_47:.*]] = subi %[[VAL_45]], %[[VAL_46]] : i64
  ! CHECK:             %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_43]], %[[VAL_47]] : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
  ! CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_48]] : !fir.ref<f32>
  ! CHECK:             %[[VAL_50:.*]] = constant 0.000000e+00 : f32
  ! CHECK:             %[[VAL_51:.*]] = cmpf ogt, %[[VAL_49]], %[[VAL_50]] : f32
  ! CHECK:             %[[VAL_52:.*]] = fir.if %[[VAL_51]] -> (!fir.array<?x?xf32>) {
  ! CHECK:               %[[VAL_53:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> i64
  ! CHECK:               %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i64) -> index
  ! CHECK:               %[[VAL_56:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i64
  ! CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:               %[[VAL_59:.*]] = fir.array_fetch %[[VAL_31]], %[[VAL_55]], %[[VAL_58]] {Fortran.offsets} : (!fir.array<?x?xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_60:.*]] = constant 3.140000e+00 : f32
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
  ! CHECK:         %[[VAL_6:.*]] = constant 100 : index
  ! CHECK:         %[[VAL_7:.*]] = constant 400 : index
  ! CHECK:         %[[VAL_8:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_9:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_10:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
  ! CHECK:         %[[VAL_12:.*]] = constant 100 : i32
  ! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
  ! CHECK:         %[[VAL_14:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_15:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
  ! CHECK:         %[[VAL_17:.*]] = constant 200 : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK:         %[[VAL_19:.*]] = constant 1 : index
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
  ! CHECK:             %[[VAL_42:.*]] = constant 1 : i32
  ! CHECK:             %[[VAL_43:.*]] = addi %[[VAL_41]], %[[VAL_42]] : i32
  ! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
  ! CHECK:             %[[VAL_46:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> i64
  ! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
  ! CHECK:             %[[VAL_49:.*]] = fir.array_fetch %[[VAL_25]], %[[VAL_45]], %[[VAL_48]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_50:.*]] = addf %[[VAL_40]], %[[VAL_49]] : f32
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
  ! CHECK:             %[[VAL_72:.*]] = constant 1.000000e+00 : f32
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
  ! CHECK:             %[[VAL_84:.*]] = constant 200 : i32
  ! CHECK:             %[[VAL_85:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_86:.*]] = addi %[[VAL_84]], %[[VAL_85]] : i32
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
  ! CHECK:         %[[VAL_7:.*]] = constant 100 : index
  ! CHECK:         %[[VAL_8:.*]] = constant 400 : index
  ! CHECK:         %[[VAL_9:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_10:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_11:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = constant 100 : i32
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
  ! CHECK:         %[[VAL_15:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_16:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
  ! CHECK:         %[[VAL_18:.*]] = constant 200 : i32
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK:         %[[VAL_20:.*]] = constant 1 : index
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
  ! CHECK:             %[[VAL_37:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_38:.*]] = subi %[[VAL_36]], %[[VAL_37]] : i64
  ! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
  ! CHECK:             %[[VAL_41:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_42:.*]] = subi %[[VAL_40]], %[[VAL_41]] : i64
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
  ! CHECK:               %[[VAL_55:.*]] = constant 1 : i32
  ! CHECK:               %[[VAL_56:.*]] = addi %[[VAL_54]], %[[VAL_55]] : i32
  ! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i64
  ! CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:               %[[VAL_59:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
  ! CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i64) -> index
  ! CHECK:               %[[VAL_62:.*]] = fir.array_fetch %[[VAL_26]], %[[VAL_58]], %[[VAL_61]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_63:.*]] = addf %[[VAL_53]], %[[VAL_62]] : f32
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
  ! CHECK:             %[[VAL_88:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_89:.*]] = subi %[[VAL_87]], %[[VAL_88]] : i64
  ! CHECK:             %[[VAL_90:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_91:.*]] = fir.convert %[[VAL_90]] : (i32) -> i64
  ! CHECK:             %[[VAL_92:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_93:.*]] = subi %[[VAL_91]], %[[VAL_92]] : i64
  ! CHECK:             %[[VAL_94:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_89]], %[[VAL_93]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_95:.*]] = fir.load %[[VAL_94]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_96:.*]] = fir.convert %[[VAL_95]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_97:.*]] = fir.if %[[VAL_96]] -> (!fir.array<100x400xf32>) {
  ! CHECK:               %[[VAL_98:.*]] = constant 1.000000e+00 : f32
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
  ! CHECK:               %[[VAL_110:.*]] = constant 200 : i32
  ! CHECK:               %[[VAL_111:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_112:.*]] = addi %[[VAL_110]], %[[VAL_111]] : i32
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
  ! CHECK:         %[[VAL_3:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_4:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
  ! CHECK:         %[[VAL_7:.*]] = constant 10 : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = constant 2 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_10]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_19:.*]] = constant 64 : i64
  ! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
  ! CHECK:           %[[VAL_21:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_22:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_23:.*]] = subi %[[VAL_20]], %[[VAL_21]] : index
  ! CHECK:           %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_12]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_28:.*]] = constant 1 : i32
  ! CHECK:             %[[VAL_29:.*]] = addi %[[VAL_27]], %[[VAL_28]] : i32
  ! CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
  ! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
  ! CHECK:             %[[VAL_32:.*]] = fir.field_index block2, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK:             %[[VAL_33:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_34:.*]] = addi %[[VAL_25]], %[[VAL_33]] : index
  ! CHECK:             %[[VAL_35:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_31]], %[[VAL_32]], %[[VAL_34]] {Fortran.offsets} : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, index, !fir.field, index) -> i64
  ! CHECK:             %[[VAL_36:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
  ! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
  ! CHECK:             %[[VAL_39:.*]] = fir.field_index block1, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK:             %[[VAL_40:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_41:.*]] = addi %[[VAL_25]], %[[VAL_40]] : index
  ! CHECK:             %[[VAL_42:.*]] = fir.array_update %[[VAL_17]], %[[VAL_35]], %[[VAL_38]], %[[VAL_39]], %[[VAL_41]] {Fortran.offsets} : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, i64, index, !fir.field, index) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
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
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "j"}
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.heap<index>
  ! CHECK:         %[[VAL_7:.*]] = fir.alloca !fir.heap<i8>
  ! CHECK:         %[[VAL_8:.*]] = fir.zero_bits !fir.heap<i8>
  ! CHECK:         fir.store %[[VAL_8]] to %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_9:.*]] = fir.zero_bits !fir.heap<index>
  ! CHECK:         fir.store %[[VAL_9]] to %[[VAL_6]] : !fir.ref<!fir.heap<index>>
  ! CHECK:         %[[VAL_10:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
  ! CHECK:         %[[VAL_12:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_12]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_15:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (index) -> i64
  ! CHECK:         %[[VAL_17:.*]] = addi %[[VAL_14]], %[[VAL_16]] : i64
  ! CHECK:         %[[VAL_18:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_19:.*]] = subi %[[VAL_17]], %[[VAL_18]] : i64
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> i32
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK:         %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> index
  ! CHECK:         %[[VAL_25:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_25]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_26]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_28:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
  ! CHECK:         %[[VAL_30:.*]] = addi %[[VAL_27]], %[[VAL_29]] : i64
  ! CHECK:         %[[VAL_31:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_32:.*]] = subi %[[VAL_30]], %[[VAL_31]] : i64
  ! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> i32
  ! CHECK:         %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
  ! CHECK:         %[[VAL_35:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_36:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_37:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_38:.*]] = fir.do_loop %[[VAL_39:.*]] = %[[VAL_11]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_40:.*]] = %[[VAL_36]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_39]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_41]] to %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_42:.*]] = fir.do_loop %[[VAL_43:.*]] = %[[VAL_24]] to %[[VAL_34]] step %[[VAL_35]] unordered iter_args(%[[VAL_44:.*]] = %[[VAL_40]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_43]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_45]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_46:.*]] = constant 100 : i64
  ! CHECK:             %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i64) -> index
  ! CHECK:             %[[VAL_48:.*]] = constant 3.140000e+00 : f32
  ! CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> i64
  ! CHECK:             %[[VAL_51:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_52:.*]] = subi %[[VAL_50]], %[[VAL_51]] : i64
  ! CHECK:             %[[VAL_53:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> i64
  ! CHECK:             %[[VAL_55:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_56:.*]] = subi %[[VAL_54]], %[[VAL_55]] : i64
  ! CHECK:             %[[VAL_57:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_52]], %[[VAL_56]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, i64, i64) -> !fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:             %[[VAL_58:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:             %[[VAL_59:.*]] = fir.coordinate_of %[[VAL_57]], %[[VAL_58]] : (!fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:             %[[VAL_60:.*]] = constant 100 : index
  ! CHECK:             %[[VAL_61:.*]] = fir.shape %[[VAL_60]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_62:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_63:.*]] = fir.slice %[[VAL_62]], %[[VAL_60]], %[[VAL_62]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:             %[[VAL_64:.*]] = fir.array_load %[[VAL_59]](%[[VAL_61]]) {{\[}}%[[VAL_63]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK:             %[[VAL_65:.*]] = constant 0.000000e+00 : f32
  ! CHECK:             %[[VAL_66:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_67:.*]] = subi %[[VAL_60]], %[[VAL_62]] : index
  ! CHECK:             %[[VAL_68:.*]] = addi %[[VAL_67]], %[[VAL_62]] : index
  ! CHECK:             %[[VAL_69:.*]] = divi_signed %[[VAL_68]], %[[VAL_62]] : index
  ! CHECK:             %[[VAL_70:.*]] = cmpi sgt, %[[VAL_69]], %[[VAL_66]] : index
  ! CHECK:             %[[VAL_71:.*]] = select %[[VAL_70]], %[[VAL_69]], %[[VAL_66]] : index
  ! CHECK:             %[[VAL_72:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:             %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:             %[[VAL_74:.*]] = fir.shape %[[VAL_71]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_75:.*]] = fir.array_load %[[VAL_73]](%[[VAL_74]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
  ! CHECK:             %[[VAL_76:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:             %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:             %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:             %[[VAL_79:.*]] = constant 0 : i64
  ! CHECK:             %[[VAL_80:.*]] = cmpi eq, %[[VAL_78]], %[[VAL_79]] : i64
  ! CHECK:             fir.if %[[VAL_80]] {
  ! CHECK:               %[[VAL_81:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_71]] {uniq_name = ".lazy.mask"}
  ! CHECK:               %[[VAL_82:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:               fir.store %[[VAL_81]] to %[[VAL_82]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:               %[[VAL_83:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
  ! CHECK:               %[[VAL_84:.*]] = constant 0 : index
  ! CHECK:               %[[VAL_85:.*]] = fir.coordinate_of %[[VAL_83]], %[[VAL_84]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
  ! CHECK:               fir.store %[[VAL_71]] to %[[VAL_85]] : !fir.ref<index>
  ! CHECK:               %[[VAL_86:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
  ! CHECK:               fir.store %[[VAL_83]] to %[[VAL_86]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
  ! CHECK:             }
  ! CHECK:             %[[VAL_87:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_88:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_89:.*]] = subi %[[VAL_71]], %[[VAL_87]] : index
  ! CHECK:             %[[VAL_90:.*]] = fir.do_loop %[[VAL_91:.*]] = %[[VAL_88]] to %[[VAL_89]] step %[[VAL_87]] unordered iter_args(%[[VAL_92:.*]] = %[[VAL_75]]) -> (!fir.array<?xi8>) {
  ! CHECK:               %[[VAL_93:.*]] = fir.array_fetch %[[VAL_64]], %[[VAL_91]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:               %[[VAL_94:.*]] = cmpf ogt, %[[VAL_93]], %[[VAL_65]] : f32
  ! CHECK:               %[[VAL_95:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:               %[[VAL_96:.*]] = fir.convert %[[VAL_95]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:               %[[VAL_97:.*]] = fir.shape %[[VAL_71]] : (index) -> !fir.shape<1>
  ! CHECK:               %[[VAL_98:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_99:.*]] = addi %[[VAL_91]], %[[VAL_98]] : index
  ! CHECK:               %[[VAL_100:.*]] = fir.array_coor %[[VAL_96]](%[[VAL_97]]) %[[VAL_99]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:               %[[VAL_101:.*]] = fir.convert %[[VAL_94]] : (i1) -> i8
  ! CHECK:               fir.store %[[VAL_101]] to %[[VAL_100]] : !fir.ref<i8>
  ! CHECK:               fir.result %[[VAL_92]] : !fir.array<?xi8>
  ! CHECK:             }
  ! CHECK:             %[[VAL_102:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:             %[[VAL_103:.*]] = fir.convert %[[VAL_102]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:             fir.array_merge_store %[[VAL_75]], %[[VAL_104:.*]] to %[[VAL_103]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
  ! CHECK:             %[[VAL_105:.*]] = fir.shape %[[VAL_71]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_106:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_107:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_108:.*]] = subi %[[VAL_47]], %[[VAL_106]] : index
  ! CHECK:             %[[VAL_109:.*]] = fir.do_loop %[[VAL_110:.*]] = %[[VAL_107]] to %[[VAL_108]] step %[[VAL_106]] unordered iter_args(%[[VAL_111:.*]] = %[[VAL_36]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:               %[[VAL_112:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_113:.*]] = addi %[[VAL_110]], %[[VAL_112]] : index
  ! CHECK:               %[[VAL_114:.*]] = fir.array_coor %[[VAL_103]](%[[VAL_105]]) %[[VAL_113]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:               %[[VAL_115:.*]] = fir.load %[[VAL_114]] : !fir.ref<i8>
  ! CHECK:               %[[VAL_116:.*]] = fir.convert %[[VAL_115]] : (i8) -> i1
  ! CHECK:               %[[VAL_117:.*]] = fir.if %[[VAL_116]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:                 %[[VAL_118:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_119:.*]] = fir.convert %[[VAL_118]] : (i32) -> i64
  ! CHECK:                 %[[VAL_120:.*]] = fir.convert %[[VAL_119]] : (i64) -> index
  ! CHECK:                 %[[VAL_121:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_122:.*]] = fir.convert %[[VAL_121]] : (i32) -> i64
  ! CHECK:                 %[[VAL_123:.*]] = fir.convert %[[VAL_122]] : (i64) -> index
  ! CHECK:                 %[[VAL_124:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:                 %[[VAL_125:.*]] = constant 1 : index
  ! CHECK:                 %[[VAL_126:.*]] = addi %[[VAL_110]], %[[VAL_125]] : index
  ! CHECK:                 %[[VAL_127:.*]] = fir.array_fetch %[[VAL_37]], %[[VAL_120]], %[[VAL_123]], %[[VAL_124]], %[[VAL_126]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
  ! CHECK:                 %[[VAL_128:.*]] = divf %[[VAL_127]], %[[VAL_48]] : f32
  ! CHECK:                 %[[VAL_129:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_130:.*]] = fir.convert %[[VAL_129]] : (i32) -> i64
  ! CHECK:                 %[[VAL_131:.*]] = fir.convert %[[VAL_130]] : (i64) -> index
  ! CHECK:                 %[[VAL_132:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_133:.*]] = fir.convert %[[VAL_132]] : (i32) -> i64
  ! CHECK:                 %[[VAL_134:.*]] = fir.convert %[[VAL_133]] : (i64) -> index
  ! CHECK:                 %[[VAL_135:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:                 %[[VAL_136:.*]] = constant 1 : index
  ! CHECK:                 %[[VAL_137:.*]] = addi %[[VAL_110]], %[[VAL_136]] : index
  ! CHECK:                 %[[VAL_138:.*]] = fir.array_update %[[VAL_44]], %[[VAL_128]], %[[VAL_131]], %[[VAL_134]], %[[VAL_135]], %[[VAL_137]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:                 fir.result %[[VAL_138]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:               } else {
  ! CHECK:                 fir.result %[[VAL_111]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:               }
  ! CHECK:               fir.result %[[VAL_139:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_140:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_141:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_36]], %[[VAL_142:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  ! CHECK:         %[[VAL_143:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_144:.*]] = fir.convert %[[VAL_143]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:         %[[VAL_145:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.array<1xindex>>
  ! CHECK:         %[[VAL_146:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_147:.*]] = fir.coordinate_of %[[VAL_145]], %[[VAL_146]] : (!fir.ref<!fir.array<1xindex>>, index) -> !fir.ref<index>
  ! CHECK:         %[[VAL_148:.*]] = fir.load %[[VAL_147]] : !fir.ref<index>
  ! CHECK:         %[[VAL_149:.*]] = fir.shape %[[VAL_148]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_150:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_151:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_152:.*]] = fir.do_loop %[[VAL_153:.*]] = %[[VAL_11]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_154:.*]] = %[[VAL_150]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:           %[[VAL_155:.*]] = fir.convert %[[VAL_153]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_155]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_156:.*]] = fir.do_loop %[[VAL_157:.*]] = %[[VAL_24]] to %[[VAL_34]] step %[[VAL_35]] unordered iter_args(%[[VAL_158:.*]] = %[[VAL_154]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:             %[[VAL_159:.*]] = fir.convert %[[VAL_157]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_159]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_160:.*]] = constant 100 : i64
  ! CHECK:             %[[VAL_161:.*]] = fir.convert %[[VAL_160]] : (i64) -> index
  ! CHECK:             %[[VAL_162:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_163:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_164:.*]] = subi %[[VAL_161]], %[[VAL_162]] : index
  ! CHECK:             %[[VAL_165:.*]] = fir.do_loop %[[VAL_166:.*]] = %[[VAL_163]] to %[[VAL_164]] step %[[VAL_162]] unordered iter_args(%[[VAL_167:.*]] = %[[VAL_150]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:               %[[VAL_168:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_169:.*]] = addi %[[VAL_166]], %[[VAL_168]] : index
  ! CHECK:               %[[VAL_170:.*]] = fir.array_coor %[[VAL_144]](%[[VAL_149]]) %[[VAL_169]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:               %[[VAL_171:.*]] = fir.load %[[VAL_170]] : !fir.ref<i8>
  ! CHECK:               %[[VAL_172:.*]] = fir.convert %[[VAL_171]] : (i8) -> i1
  ! CHECK:               %[[VAL_173:.*]] = fir.if %[[VAL_172]] -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:                 fir.result %[[VAL_167]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:               } else {
  ! CHECK:                 %[[VAL_174:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_175:.*]] = fir.convert %[[VAL_174]] : (i32) -> i64
  ! CHECK:                 %[[VAL_176:.*]] = fir.convert %[[VAL_175]] : (i64) -> index
  ! CHECK:                 %[[VAL_177:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_178:.*]] = fir.convert %[[VAL_177]] : (i32) -> i64
  ! CHECK:                 %[[VAL_179:.*]] = fir.convert %[[VAL_178]] : (i64) -> index
  ! CHECK:                 %[[VAL_180:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:                 %[[VAL_181:.*]] = constant 1 : index
  ! CHECK:                 %[[VAL_182:.*]] = addi %[[VAL_166]], %[[VAL_181]] : index
  ! CHECK:                 %[[VAL_183:.*]] = fir.array_fetch %[[VAL_151]], %[[VAL_176]], %[[VAL_179]], %[[VAL_180]], %[[VAL_182]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field, index) -> f32
  ! CHECK:                 %[[VAL_184:.*]] = negf %[[VAL_183]] : f32
  ! CHECK:                 %[[VAL_185:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_186:.*]] = fir.convert %[[VAL_185]] : (i32) -> i64
  ! CHECK:                 %[[VAL_187:.*]] = fir.convert %[[VAL_186]] : (i64) -> index
  ! CHECK:                 %[[VAL_188:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:                 %[[VAL_189:.*]] = fir.convert %[[VAL_188]] : (i32) -> i64
  ! CHECK:                 %[[VAL_190:.*]] = fir.convert %[[VAL_189]] : (i64) -> index
  ! CHECK:                 %[[VAL_191:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:                 %[[VAL_192:.*]] = constant 1 : index
  ! CHECK:                 %[[VAL_193:.*]] = addi %[[VAL_166]], %[[VAL_192]] : index
  ! CHECK:                 %[[VAL_194:.*]] = fir.array_update %[[VAL_158]], %[[VAL_184]], %[[VAL_187]], %[[VAL_190]], %[[VAL_191]], %[[VAL_193]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, f32, index, index, !fir.field, index) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:                 fir.result %[[VAL_194]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:               }
  ! CHECK:               fir.result %[[VAL_195:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_196:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_197:.*]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_150]], %[[VAL_198:.*]] to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  ! CHECK:         %[[VAL_199:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_200:.*]] = fir.convert %[[VAL_199]] : (!fir.heap<i8>) -> i64
  ! CHECK:         %[[VAL_201:.*]] = constant 0 : i64
  ! CHECK:         %[[VAL_202:.*]] = cmpi ne, %[[VAL_200]], %[[VAL_201]] : i64
  ! CHECK:         fir.if %[[VAL_202]] {
  ! CHECK:           fir.freemem %[[VAL_199]] : !fir.heap<i8>
  ! CHECK:         }

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
  ! CHECK:         %[[VAL_4:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_sliceEa"}
  ! CHECK:         %[[VAL_7:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_12:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
  ! CHECK:         %[[VAL_14:.*]] = constant 10 : i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
  ! CHECK:         %[[VAL_16:.*]] = constant 1 : index
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
  ! CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
  ! CHECK:             %[[VAL_30:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
  ! CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
  ! CHECK:             %[[VAL_33:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
  ! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
  ! CHECK:             %[[VAL_36:.*]] = subi %[[VAL_32]], %[[VAL_29]] : index
  ! CHECK:             %[[VAL_37:.*]] = addi %[[VAL_36]], %[[VAL_35]] : index
  ! CHECK:             %[[VAL_38:.*]] = divi_signed %[[VAL_37]], %[[VAL_35]] : index
  ! CHECK:             %[[VAL_39:.*]] = fir.call @_QPf(%[[VAL_3]]) : (!fir.ref<i32>) -> i32
  ! CHECK:             %[[VAL_40:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_41:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_42:.*]] = subi %[[VAL_38]], %[[VAL_40]] : index
  ! CHECK:             %[[VAL_43:.*]] = fir.do_loop %[[VAL_44:.*]] = %[[VAL_41]] to %[[VAL_42]] step %[[VAL_40]] unordered iter_args(%[[VAL_45:.*]] = %[[VAL_18]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:               %[[VAL_46:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i32) -> i64
  ! CHECK:               %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
  ! CHECK:               %[[VAL_49:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> i64
  ! CHECK:               %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i64) -> index
  ! CHECK:               %[[VAL_52:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>
  ! CHECK:               %[[VAL_53:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> i64
  ! CHECK:               %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i64) -> index
  ! CHECK:               %[[VAL_56:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i64
  ! CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:               %[[VAL_59:.*]] = muli %[[VAL_44]], %[[VAL_58]] : index
  ! CHECK:               %[[VAL_60:.*]] = addi %[[VAL_55]], %[[VAL_59]] : index
  ! CHECK:               %[[VAL_61:.*]] = fir.array_update %[[VAL_25]], %[[VAL_39]], %[[VAL_48]], %[[VAL_51]], %[[VAL_52]], %[[VAL_60]] {Fortran.offsets} : (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:               fir.result %[[VAL_61]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_62:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_63:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_64:.*]] to %[[VAL_6]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>


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
  ! CHECK:         %[[VAL_1:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_ranked_dimensionEa"}
  ! CHECK:         %[[VAL_4:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
  ! CHECK:         %[[VAL_6:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_16:.*]] = addi %[[VAL_15]], %[[VAL_2]] : index
  ! CHECK:           %[[VAL_17:.*]] = subi %[[VAL_16]], %[[VAL_15]] : index
  ! CHECK:           %[[VAL_18:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:           %[[VAL_20:.*]] = subi %[[VAL_17]], %[[VAL_15]] : index
  ! CHECK:           %[[VAL_21:.*]] = addi %[[VAL_20]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_22:.*]] = divi_signed %[[VAL_21]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_23:.*]] = fir.call @_QPf(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  ! CHECK:           %[[VAL_24:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_25:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_26:.*]] = subi %[[VAL_1]], %[[VAL_24]] : index
  ! CHECK:           %[[VAL_27:.*]] = subi %[[VAL_22]], %[[VAL_24]] : index
  ! CHECK:           %[[VAL_28:.*]] = fir.do_loop %[[VAL_29:.*]] = %[[VAL_25]] to %[[VAL_27]] step %[[VAL_24]] unordered iter_args(%[[VAL_30:.*]] = %[[VAL_10]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:             %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_25]] to %[[VAL_26]] step %[[VAL_24]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_30]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:               %[[VAL_34:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> i64
  ! CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> index
  ! CHECK:               %[[VAL_37:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_38:.*]] = constant 1 : i64
  ! CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i64) -> index
  ! CHECK:               %[[VAL_40:.*]] = muli %[[VAL_32]], %[[VAL_39]] : index
  ! CHECK:               %[[VAL_41:.*]] = addi %[[VAL_37]], %[[VAL_40]] : index
  ! CHECK:               %[[VAL_42:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>
  ! CHECK:               %[[VAL_43:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_44:.*]] = constant 4 : i32
  ! CHECK:               %[[VAL_45:.*]] = addi %[[VAL_43]], %[[VAL_44]] : i32
  ! CHECK:               %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i32) -> i64
  ! CHECK:               %[[VAL_47:.*]] = fir.convert %[[VAL_46]] : (i64) -> index
  ! CHECK:               %[[VAL_48:.*]] = fir.array_update %[[VAL_13]], %[[VAL_23]], %[[VAL_36]], %[[VAL_41]], %[[VAL_42]], %[[VAL_47]] {Fortran.offsets} : (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, i32, index, index, !fir.field, index) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:               fir.result %[[VAL_48]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_49:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_50:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_51:.*]] to %[[VAL_3]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>

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
  ! CHECK:         %[[VAL_7:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
  ! CHECK:         %[[VAL_9:.*]] = constant 15 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = constant 1 : index
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
  ! CHECK:         %[[VAL_5:.*]] = constant {{.*}} : i32
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAInitialize(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_9:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = constant 15 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_14:.*]] = fir.field_index arr, !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_14]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_17:.*]] = constant 0 : index
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
