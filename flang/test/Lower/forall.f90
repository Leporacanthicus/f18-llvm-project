! Test forall lowering

! RUN: bbc -emit-fir %s -o - | FileCheck %s

!*** This FORALL construct does present a potential loop-carried dependence if
!*** implemented naively (and incorrectly). The final value of a(3) must be the
!*** value of a(2) before loopy begins execution added to b(2).
! CHECK-LABEL: func @_QPtest9(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<i32>) {
subroutine test9(a,b,n)
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> i64
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
  ! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i64
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
  ! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_0]](%[[VAL_12]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_1]](%[[VAL_14]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_16:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
  ! CHECK:         %[[VAL_18:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_19:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_20:.*]] = subi %[[VAL_18]], %[[VAL_19]] : i32
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK:         %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_17]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_11]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_26]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
  ! CHECK:           %[[VAL_30:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_29]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK:           %[[VAL_34:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_33]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_35:.*]] = addf %[[VAL_30]], %[[VAL_34]] : f32
  ! CHECK:           %[[VAL_36:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_37:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_38:.*]] = addi %[[VAL_36]], %[[VAL_37]] : i32
  ! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> i64
  ! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
  ! CHECK:           %[[VAL_41:.*]] = fir.array_update %[[VAL_25]], %[[VAL_35]], %[[VAL_40]] {Fortran.offsets} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_41]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_42:.*]] to %[[VAL_0]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>
  ! CHECK:         return
  ! CHECK:       }
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  loopy: FORALL (i=1:n-1)
     a(i+1) = a(i) + b(i)
  END FORALL loopy
end subroutine test9

!*** Test a FORALL statement
! CHECK-LABEL: func @_QPtest_forall_stmt(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<200xf32>>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.array<200x!fir.logical<4>>>) {
subroutine test_forall_stmt(x, mask)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_3:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_0]](%[[VAL_4]]) : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> !fir.array<200xf32>
  ! CHECK:         %[[VAL_6:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 100 : i32
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
  ! CHECK:         %[[VAL_10:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_5]]) -> (!fir.array<200xf32>) {
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
  ! CHECK:         fir.array_merge_store %[[VAL_5]], %[[VAL_29:.*]] to %[[VAL_0]] : !fir.array<200xf32>, !fir.array<200xf32>, !fir.ref<!fir.array<200xf32>>
  ! CHECK:         return
  ! CHECK:       }
  logical :: mask(200)
  real :: x(200)
  forall (i=1:100,mask(i)) x(i) = 1.
end subroutine test_forall_stmt

!*** Test a FORALL construct
! CHECK-LABEL: func @_QPtest_forall_construct(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>) {
subroutine test_forall_construct(a,b)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.array<?x?xf32>
  ! CHECK:         %[[VAL_6:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_8]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_11:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (index) -> i64
  ! CHECK:         %[[VAL_13:.*]] = addi %[[VAL_10]], %[[VAL_12]] : i64
  ! CHECK:         %[[VAL_14:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_15:.*]] = subi %[[VAL_13]], %[[VAL_14]] : i64
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> i32
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> index
  ! CHECK:         %[[VAL_18:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %[[VAL_7]] to %[[VAL_17]] step %[[VAL_18]] unordered iter_args(%[[VAL_21:.*]] = %[[VAL_4]]) -> (!fir.array<?x?xf32>) {
  ! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_20]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_22]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_23:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> index
  ! CHECK:           %[[VAL_25:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_25]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]]#1 : (index) -> i64
  ! CHECK:           %[[VAL_28:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (index) -> i64
  ! CHECK:           %[[VAL_30:.*]] = addi %[[VAL_27]], %[[VAL_29]] : i64
  ! CHECK:           %[[VAL_31:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_32:.*]] = subi %[[VAL_30]], %[[VAL_31]] : i64
  ! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> i32
  ! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
  ! CHECK:           %[[VAL_35:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_36:.*]] = fir.do_loop %[[VAL_37:.*]] = %[[VAL_24]] to %[[VAL_34]] step %[[VAL_35]] unordered iter_args(%[[VAL_38:.*]] = %[[VAL_21]]) -> (!fir.array<?x?xf32>) {
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
  ! CHECK:               %[[VAL_59:.*]] = fir.array_fetch %[[VAL_5]], %[[VAL_55]], %[[VAL_58]] {Fortran.offsets} : (!fir.array<?x?xf32>, index, index) -> f32
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
  ! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_71:.*]] to %[[VAL_0]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  ! CHECK:         return
  ! CHECK:       }
  real :: a(:,:), b(:,:)
  forall (i=1:ubound(a,1), j=1:ubound(a,2), b(j,i) > 0.0)
     a(i,j) = b(j,i) / 3.14
  end forall
end subroutine test_forall_construct

!*** Test forall with multiple assignment statements
! CHECK-LABEL: func @_QPtest2_forall_construct(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>>) {
subroutine test2_forall_construct(a,b)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = constant 100 : index
  ! CHECK:         %[[VAL_5:.*]] = constant 400 : index
  ! CHECK:         %[[VAL_6:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_7:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]](%[[VAL_8]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]](%[[VAL_10]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_1]](%[[VAL_12]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_1]](%[[VAL_14]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_6]], %[[VAL_7]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_1]](%[[VAL_16]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_18:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK:         %[[VAL_20:.*]] = constant 100 : i32
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK:         %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_23:.*]]:2 = fir.do_loop %[[VAL_24:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_9]], %[[VAL_26:.*]] = %[[VAL_11]]) -> (!fir.array<100x400xf32>, !fir.array<100x400xf32>) {
  ! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_27]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_28:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> index
  ! CHECK:           %[[VAL_30:.*]] = constant 200 : i32
  ! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> index
  ! CHECK:           %[[VAL_32:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_33:.*]]:2 = fir.do_loop %[[VAL_34:.*]] = %[[VAL_29]] to %[[VAL_31]] step %[[VAL_32]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_25]], %[[VAL_36:.*]] = %[[VAL_26]]) -> (!fir.array<100x400xf32>, !fir.array<100x400xf32>) {
  ! CHECK:             %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_37]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> i64
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i64) -> index
  ! CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> i64
  ! CHECK:             %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (i64) -> index
  ! CHECK:             %[[VAL_44:.*]] = fir.array_fetch %[[VAL_13]], %[[VAL_40]], %[[VAL_43]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_45:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_46:.*]] = constant 1 : i32
  ! CHECK:             %[[VAL_47:.*]] = addi %[[VAL_45]], %[[VAL_46]] : i32
  ! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
  ! CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i64) -> index
  ! CHECK:             %[[VAL_50:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (i32) -> i64
  ! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i64) -> index
  ! CHECK:             %[[VAL_53:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_49]], %[[VAL_52]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_54:.*]] = addf %[[VAL_44]], %[[VAL_53]] : f32
  ! CHECK:             %[[VAL_55:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i32) -> i64
  ! CHECK:             %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i64) -> index
  ! CHECK:             %[[VAL_58:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_59:.*]] = fir.convert %[[VAL_58]] : (i32) -> i64
  ! CHECK:             %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i64) -> index
  ! CHECK:             %[[VAL_61:.*]] = fir.array_update %[[VAL_35]], %[[VAL_54]], %[[VAL_57]], %[[VAL_60]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:             %[[VAL_62:.*]] = constant 1.000000e+00 : f32
  ! CHECK:             %[[VAL_63:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i32) -> i64
  ! CHECK:             %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i64) -> index
  ! CHECK:             %[[VAL_66:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_67:.*]] = fir.convert %[[VAL_66]] : (i32) -> i64
  ! CHECK:             %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (i64) -> index
  ! CHECK:             %[[VAL_69:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_65]], %[[VAL_68]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:             %[[VAL_70:.*]] = divf %[[VAL_62]], %[[VAL_69]] : f32
  ! CHECK:             %[[VAL_71:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (i32) -> i64
  ! CHECK:             %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (i64) -> index
  ! CHECK:             %[[VAL_74:.*]] = constant 200 : i32
  ! CHECK:             %[[VAL_75:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_76:.*]] = addi %[[VAL_74]], %[[VAL_75]] : i32
  ! CHECK:             %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (i32) -> i64
  ! CHECK:             %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : (i64) -> index
  ! CHECK:             %[[VAL_79:.*]] = fir.array_update %[[VAL_36]], %[[VAL_70]], %[[VAL_73]], %[[VAL_78]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:             fir.result %[[VAL_61]], %[[VAL_79]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_80:.*]]#0, %[[VAL_80]]#1 : !fir.array<100x400xf32>, !fir.array<100x400xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_81:.*]]#0 to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_81]]#1 to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK:         return
  ! CHECK:       }
  real :: a(100,400), b(200,200)
  forall (i=1:100, j=1:200)
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
end subroutine test2_forall_construct

!*** Test forall with multiple assignment statements and mask
! CHECK-LABEL: func @_QPtest3_forall_construct(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<100x400xf32>>,
! CHECK-SAME: %[[VAL_1:.*]]: !fir.ref<!fir.array<200x200xf32>>,
! CHECK-SAME: %[[VAL_2:.*]]: !fir.ref<!fir.array<100x200x!fir.logical<4>>>) {
subroutine test3_forall_construct(a,b, mask)
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_5:.*]] = constant 100 : index
  ! CHECK:         %[[VAL_6:.*]] = constant 400 : index
  ! CHECK:         %[[VAL_7:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_8:.*]] = constant 200 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_5]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_0]](%[[VAL_9]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_5]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<100x400xf32>>, !fir.shape<2>) -> !fir.array<100x400xf32>
  ! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_1]](%[[VAL_13]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_15:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_1]](%[[VAL_15]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_1]](%[[VAL_17]]) : (!fir.ref<!fir.array<200x200xf32>>, !fir.shape<2>) -> !fir.array<200x200xf32>
  ! CHECK:         %[[VAL_19:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> index
  ! CHECK:         %[[VAL_21:.*]] = constant 100 : i32
  ! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> index
  ! CHECK:         %[[VAL_23:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_24:.*]]:2 = fir.do_loop %[[VAL_25:.*]] = %[[VAL_20]] to %[[VAL_22]] step %[[VAL_23]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_10]], %[[VAL_27:.*]] = %[[VAL_12]]) -> (!fir.array<100x400xf32>, !fir.array<100x400xf32>) {
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_28]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_29:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
  ! CHECK:           %[[VAL_31:.*]] = constant 200 : i32
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
  ! CHECK:           %[[VAL_33:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_34:.*]]:2 = fir.do_loop %[[VAL_35:.*]] = %[[VAL_30]] to %[[VAL_32]] step %[[VAL_33]] unordered iter_args(%[[VAL_36:.*]] = %[[VAL_26]], %[[VAL_37:.*]] = %[[VAL_27]]) -> (!fir.array<100x400xf32>, !fir.array<100x400xf32>) {
  ! CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_35]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_38]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
  ! CHECK:             %[[VAL_41:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_42:.*]] = subi %[[VAL_40]], %[[VAL_41]] : i64
  ! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
  ! CHECK:             %[[VAL_45:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_46:.*]] = subi %[[VAL_44]], %[[VAL_45]] : i64
  ! CHECK:             %[[VAL_47:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_42]], %[[VAL_46]] : (!fir.ref<!fir.array<100x200x!fir.logical<4>>>, i64, i64) -> !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_48:.*]] = fir.load %[[VAL_47]] : !fir.ref<!fir.logical<4>>
  ! CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (!fir.logical<4>) -> i1
  ! CHECK:             %[[VAL_50:.*]]:2 = fir.if %[[VAL_49]] -> (!fir.array<100x400xf32>, !fir.array<100x400xf32>) {
  ! CHECK:               %[[VAL_51:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> i64
  ! CHECK:               %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i64) -> index
  ! CHECK:               %[[VAL_54:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (i32) -> i64
  ! CHECK:               %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i64) -> index
  ! CHECK:               %[[VAL_57:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_53]], %[[VAL_56]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_58:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_59:.*]] = constant 1 : i32
  ! CHECK:               %[[VAL_60:.*]] = addi %[[VAL_58]], %[[VAL_59]] : i32
  ! CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i32) -> i64
  ! CHECK:               %[[VAL_62:.*]] = fir.convert %[[VAL_61]] : (i64) -> index
  ! CHECK:               %[[VAL_63:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i32) -> i64
  ! CHECK:               %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i64) -> index
  ! CHECK:               %[[VAL_66:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_62]], %[[VAL_65]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_67:.*]] = addf %[[VAL_57]], %[[VAL_66]] : f32
  ! CHECK:               %[[VAL_68:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_69:.*]] = fir.convert %[[VAL_68]] : (i32) -> i64
  ! CHECK:               %[[VAL_70:.*]] = fir.convert %[[VAL_69]] : (i64) -> index
  ! CHECK:               %[[VAL_71:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (i32) -> i64
  ! CHECK:               %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (i64) -> index
  ! CHECK:               %[[VAL_74:.*]] = fir.array_update %[[VAL_36]], %[[VAL_67]], %[[VAL_70]], %[[VAL_73]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:               %[[VAL_75:.*]] = constant 1.000000e+00 : f32
  ! CHECK:               %[[VAL_76:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (i32) -> i64
  ! CHECK:               %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : (i64) -> index
  ! CHECK:               %[[VAL_79:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (i32) -> i64
  ! CHECK:               %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (i64) -> index
  ! CHECK:               %[[VAL_82:.*]] = fir.array_fetch %[[VAL_18]], %[[VAL_78]], %[[VAL_81]] {Fortran.offsets} : (!fir.array<200x200xf32>, index, index) -> f32
  ! CHECK:               %[[VAL_83:.*]] = divf %[[VAL_75]], %[[VAL_82]] : f32
  ! CHECK:               %[[VAL_84:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (i32) -> i64
  ! CHECK:               %[[VAL_86:.*]] = fir.convert %[[VAL_85]] : (i64) -> index
  ! CHECK:               %[[VAL_87:.*]] = constant 200 : i32
  ! CHECK:               %[[VAL_88:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:               %[[VAL_89:.*]] = addi %[[VAL_87]], %[[VAL_88]] : i32
  ! CHECK:               %[[VAL_90:.*]] = fir.convert %[[VAL_89]] : (i32) -> i64
  ! CHECK:               %[[VAL_91:.*]] = fir.convert %[[VAL_90]] : (i64) -> index
  ! CHECK:               %[[VAL_92:.*]] = fir.array_update %[[VAL_37]], %[[VAL_83]], %[[VAL_86]], %[[VAL_91]] {Fortran.offsets} : (!fir.array<100x400xf32>, f32, index, index) -> !fir.array<100x400xf32>
  ! CHECK:               fir.result %[[VAL_74]], %[[VAL_92]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_36]], %[[VAL_37]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_93:.*]]#0, %[[VAL_93]]#1 : !fir.array<100x400xf32>, !fir.array<100x400xf32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_94:.*]]#0, %[[VAL_94]]#1 : !fir.array<100x400xf32>, !fir.array<100x400xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_95:.*]]#0 to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_95]]#1 to %[[VAL_0]] : !fir.array<100x400xf32>, !fir.array<100x400xf32>, !fir.ref<!fir.array<100x400xf32>>
  ! CHECK:         return
  ! CHECK:       }
  real :: a(100,400), b(200,200)
  logical :: mask(100,200)
  forall (i=1:100, j=1:200, mask(i,j))
     a(i,j) = b(i,j) + b(i+1,j)
     a(i,200+j) = 1.0 / b(j, i)
  end forall
end subroutine test3_forall_construct

!*** Test a FORALL construct with an array assignment
!    This is similar to the following embedded WHERE construct test, but the
!    elements are assigned unconditionally.
  ! CHECK-LABEL: func @_QPtest_forall_with_array_assignment(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>) {
subroutine test_forall_with_array_assignment(aa,bb)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_3:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_4:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]](%[[VAL_5]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_1]](%[[VAL_7]]) : (!fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>, !fir.shape<1>) -> !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         %[[VAL_9:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = constant 10 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = constant 2 : i32
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
  ! CHECK:         %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_14]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_6]]) -> (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>) {
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_18]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
  ! CHECK:           %[[VAL_22:.*]] = fir.field_index block1, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK:           %[[VAL_23:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_21]], %[[VAL_22]] {Fortran.offsets} : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, index, !fir.field) -> !fir.ref<!fir.array<64xi64>>
  ! CHECK:           %[[VAL_24:.*]] = constant 64 : index
  ! CHECK:           %[[VAL_25:.*]] = fir.shape %[[VAL_24]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_26:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_27:.*]] = fir.slice %[[VAL_26]], %[[VAL_24]], %[[VAL_26]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:           %[[VAL_28:.*]] = fir.array_load %[[VAL_23]](%[[VAL_25]]) {{\[}}%[[VAL_27]]] : (!fir.ref<!fir.array<64xi64>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<64xi64>
  ! CHECK:           %[[VAL_29:.*]] = constant 64 : i64
  ! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i64) -> index
  ! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_32:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_33:.*]] = addi %[[VAL_31]], %[[VAL_32]] : i32
  ! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
  ! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
  ! CHECK:           %[[VAL_36:.*]] = fir.field_index block2, !fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>
  ! CHECK:           %[[VAL_37:.*]] = fir.array_fetch %[[VAL_8]], %[[VAL_35]], %[[VAL_36]] {Fortran.offsets} : (!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, index, !fir.field) -> !fir.ref<!fir.array<64xi64>>
  ! CHECK:           %[[VAL_38:.*]] = constant 64 : index
  ! CHECK:           %[[VAL_39:.*]] = fir.shape %[[VAL_38]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_40:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_41:.*]] = fir.slice %[[VAL_40]], %[[VAL_38]], %[[VAL_40]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:           %[[VAL_42:.*]] = fir.array_load %[[VAL_37]](%[[VAL_39]]) {{\[}}%[[VAL_41]]] : (!fir.ref<!fir.array<64xi64>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<64xi64>
  ! CHECK:           %[[VAL_43:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_44:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_45:.*]] = subi %[[VAL_30]], %[[VAL_43]] : index
  ! CHECK:           %[[VAL_46:.*]] = fir.do_loop %[[VAL_47:.*]] = %[[VAL_44]] to %[[VAL_45]] step %[[VAL_43]] unordered iter_args(%[[VAL_48:.*]] = %[[VAL_28]]) -> (!fir.array<64xi64>) {
  ! CHECK:             %[[VAL_49:.*]] = fir.array_fetch %[[VAL_42]], %[[VAL_47]] : (!fir.array<64xi64>, index) -> i64
  ! CHECK:             %[[VAL_50:.*]] = fir.array_update %[[VAL_48]], %[[VAL_49]], %[[VAL_47]] : (!fir.array<64xi64>, i64, index) -> !fir.array<64xi64>
  ! CHECK:             fir.result %[[VAL_50]] : !fir.array<64xi64>
  ! CHECK:           }
  ! CHECK:           fir.array_merge_store %[[VAL_28]], %[[VAL_51:.*]] to %[[VAL_23]]{{\[}}%[[VAL_27]]] : !fir.array<64xi64>, !fir.array<64xi64>, !fir.ref<!fir.array<64xi64>>, !fir.slice<1>
  ! CHECK:           fir.result %[[VAL_17]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_6]], %[[VAL_52:.*]] to %[[VAL_0]] : !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>, !fir.ref<!fir.array<10x!fir.type<_QFtest_forall_with_array_assignmentTt{block1:!fir.array<64xi64>,block2:!fir.array<64xi64>}>>>
  ! CHECK:         return
  ! CHECK:       }
  type t
     integer(kind=8) :: block1(64)
     integer(kind=8) :: block2(64)
  end type t
  type(t) :: aa(10), bb(10)

  forall (i=1:10:2)
     aa(i)%block1 = bb(i+1)%block2
  end forall
end subroutine test_forall_with_array_assignment

!*** Test a FORALL construct with a nested WHERE construct.
!    This has both an explicit and implicit iteration space. The WHERE construct
!    makes the assignments conditional and the where mask evaluation must happen
!    prior to evaluating the array assignment statement.
! CHECK-LABEL: func @_QPtest_nested_forall_where(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) {
subroutine test_nested_forall_where(a,b)  
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_1]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>) -> !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         %[[VAL_8:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
  ! CHECK:         %[[VAL_10:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_10]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_13:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
  ! CHECK:         %[[VAL_15:.*]] = addi %[[VAL_12]], %[[VAL_14]] : i64
  ! CHECK:         %[[VAL_16:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_17:.*]] = subi %[[VAL_15]], %[[VAL_16]] : i64
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> i32
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK:         %[[VAL_20:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_21:.*]]:2 = fir.do_loop %[[VAL_22:.*]] = %[[VAL_9]] to %[[VAL_19]] step %[[VAL_20]] unordered iter_args(%[[VAL_23:.*]] = %[[VAL_4]], %[[VAL_24:.*]] = %[[VAL_5]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_22]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_25]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_26:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> index
  ! CHECK:           %[[VAL_28:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_29:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_28]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, index) -> (index, index, index)
  ! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]]#1 : (index) -> i64
  ! CHECK:           %[[VAL_31:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (index) -> i64
  ! CHECK:           %[[VAL_33:.*]] = addi %[[VAL_30]], %[[VAL_32]] : i64
  ! CHECK:           %[[VAL_34:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_35:.*]] = subi %[[VAL_33]], %[[VAL_34]] : i64
  ! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i64) -> i32
  ! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
  ! CHECK:           %[[VAL_38:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_39:.*]]:2 = fir.do_loop %[[VAL_40:.*]] = %[[VAL_27]] to %[[VAL_37]] step %[[VAL_38]] unordered iter_args(%[[VAL_41:.*]] = %[[VAL_23]], %[[VAL_42:.*]] = %[[VAL_24]]) -> (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>) {
  ! CHECK:             %[[VAL_43:.*]] = fir.convert %[[VAL_40]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_43]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_44:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i32) -> i64
  ! CHECK:             %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i64) -> index
  ! CHECK:             %[[VAL_47:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i32) -> i64
  ! CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i64) -> index
  ! CHECK:             %[[VAL_50:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:             %[[VAL_51:.*]] = fir.array_fetch %[[VAL_4]], %[[VAL_46]], %[[VAL_49]], %[[VAL_50]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:             %[[VAL_52:.*]] = constant 100 : index
  ! CHECK:             %[[VAL_53:.*]] = fir.shape %[[VAL_52]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_54:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_55:.*]] = fir.slice %[[VAL_54]], %[[VAL_52]], %[[VAL_54]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:             %[[VAL_56:.*]] = fir.array_load %[[VAL_51]](%[[VAL_53]]) {{\[}}%[[VAL_55]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK:             %[[VAL_57:.*]] = constant 100 : i64
  ! CHECK:             %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i64) -> index
  ! CHECK:             %[[VAL_59:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i32) -> i64
  ! CHECK:             %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i64) -> index
  ! CHECK:             %[[VAL_62:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (i32) -> i64
  ! CHECK:             %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i64) -> index
  ! CHECK:             %[[VAL_65:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:             %[[VAL_66:.*]] = fir.array_fetch %[[VAL_6]], %[[VAL_61]], %[[VAL_64]], %[[VAL_65]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:             %[[VAL_67:.*]] = constant 100 : index
  ! CHECK:             %[[VAL_68:.*]] = fir.shape %[[VAL_67]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_69:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_70:.*]] = fir.slice %[[VAL_69]], %[[VAL_67]], %[[VAL_69]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:             %[[VAL_71:.*]] = fir.array_load %[[VAL_66]](%[[VAL_68]]) {{\[}}%[[VAL_70]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK:             %[[VAL_72:.*]] = constant 3.140000e+00 : f32
  ! CHECK:             %[[VAL_73:.*]] = constant 100 : i64
  ! CHECK:             %[[VAL_74:.*]] = fir.convert %[[VAL_73]] : (i64) -> index
  ! CHECK:             %[[VAL_75:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (i32) -> i64
  ! CHECK:             %[[VAL_77:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_78:.*]] = subi %[[VAL_76]], %[[VAL_77]] : i64
  ! CHECK:             %[[VAL_79:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (i32) -> i64
  ! CHECK:             %[[VAL_81:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_82:.*]] = subi %[[VAL_80]], %[[VAL_81]] : i64
  ! CHECK:             %[[VAL_83:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_78]], %[[VAL_82]] : (!fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>, i64, i64) -> !fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:             %[[VAL_84:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:             %[[VAL_85:.*]] = fir.coordinate_of %[[VAL_83]], %[[VAL_84]] : (!fir.ref<!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:             %[[VAL_86:.*]] = constant 100 : index
  ! CHECK:             %[[VAL_87:.*]] = fir.shape %[[VAL_86]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_88:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_89:.*]] = fir.slice %[[VAL_88]], %[[VAL_86]], %[[VAL_88]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:             %[[VAL_90:.*]] = fir.array_load %[[VAL_85]](%[[VAL_87]]) {{\[}}%[[VAL_89]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK:             %[[VAL_91:.*]] = constant 0.000000e+00 : f32
  ! CHECK:             %[[VAL_92:.*]] = fir.allocmem !fir.array<100x!fir.logical<4>>
  ! CHECK:             %[[VAL_93:.*]] = fir.shape %[[VAL_74]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_94:.*]] = fir.array_load %[[VAL_92]](%[[VAL_93]]) : (!fir.heap<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<100x!fir.logical<4>>
  ! CHECK:             %[[VAL_95:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_96:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_97:.*]] = subi %[[VAL_74]], %[[VAL_95]] : index
  ! CHECK:             %[[VAL_98:.*]] = fir.do_loop %[[VAL_99:.*]] = %[[VAL_96]] to %[[VAL_97]] step %[[VAL_95]] unordered iter_args(%[[VAL_100:.*]] = %[[VAL_94]]) -> (!fir.array<100x!fir.logical<4>>) {
  ! CHECK:               %[[VAL_101:.*]] = fir.array_fetch %[[VAL_90]], %[[VAL_99]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:               %[[VAL_102:.*]] = cmpf ogt, %[[VAL_101]], %[[VAL_91]] : f32
  ! CHECK:               %[[VAL_103:.*]] = fir.convert %[[VAL_102]] : (i1) -> !fir.logical<4>
  ! CHECK:               %[[VAL_104:.*]] = fir.array_update %[[VAL_100]], %[[VAL_103]], %[[VAL_99]] : (!fir.array<100x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<100x!fir.logical<4>>
  ! CHECK:               fir.result %[[VAL_104]] : !fir.array<100x!fir.logical<4>>
  ! CHECK:             }
  ! CHECK:             fir.array_merge_store %[[VAL_94]], %[[VAL_105:.*]] to %[[VAL_92]] : !fir.array<100x!fir.logical<4>>, !fir.array<100x!fir.logical<4>>, !fir.heap<!fir.array<100x!fir.logical<4>>>
  ! CHECK:             %[[VAL_106:.*]] = fir.shape %[[VAL_74]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_107:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_108:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_109:.*]] = subi %[[VAL_58]], %[[VAL_107]] : index
  ! CHECK:             %[[VAL_110:.*]] = fir.do_loop %[[VAL_111:.*]] = %[[VAL_108]] to %[[VAL_109]] step %[[VAL_107]] unordered iter_args(%[[VAL_112:.*]] = %[[VAL_56]]) -> (!fir.array<100xf32>) {
  ! CHECK:               %[[VAL_113:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_114:.*]] = addi %[[VAL_111]], %[[VAL_113]] : index
  ! CHECK:               %[[VAL_115:.*]] = fir.array_coor %[[VAL_92]](%[[VAL_106]]) %[[VAL_114]] : (!fir.heap<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:               %[[VAL_116:.*]] = fir.load %[[VAL_115]] : !fir.ref<!fir.logical<4>>
  ! CHECK:               %[[VAL_117:.*]] = fir.convert %[[VAL_116]] : (!fir.logical<4>) -> i1
  ! CHECK:               %[[VAL_118:.*]] = fir.if %[[VAL_117]] -> (!fir.array<100xf32>) {
  ! CHECK:                 %[[VAL_119:.*]] = fir.array_fetch %[[VAL_71]], %[[VAL_111]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:                 %[[VAL_120:.*]] = divf %[[VAL_119]], %[[VAL_72]] : f32
  ! CHECK:                 %[[VAL_121:.*]] = fir.array_update %[[VAL_112]], %[[VAL_120]], %[[VAL_111]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK:                 fir.result %[[VAL_121]] : !fir.array<100xf32>
  ! CHECK:               } else {
  ! CHECK:                 fir.result %[[VAL_112]] : !fir.array<100xf32>
  ! CHECK:               }
  ! CHECK:               fir.result %[[VAL_122:.*]] : !fir.array<100xf32>
  ! CHECK:             }
  ! CHECK:             fir.array_merge_store %[[VAL_56]], %[[VAL_123:.*]] to %[[VAL_51]]{{\[}}%[[VAL_55]]] : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>, !fir.slice<1>
  ! CHECK:             %[[VAL_124:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_125:.*]] = fir.convert %[[VAL_124]] : (i32) -> i64
  ! CHECK:             %[[VAL_126:.*]] = fir.convert %[[VAL_125]] : (i64) -> index
  ! CHECK:             %[[VAL_127:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_128:.*]] = fir.convert %[[VAL_127]] : (i32) -> i64
  ! CHECK:             %[[VAL_129:.*]] = fir.convert %[[VAL_128]] : (i64) -> index
  ! CHECK:             %[[VAL_130:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:             %[[VAL_131:.*]] = fir.array_fetch %[[VAL_5]], %[[VAL_126]], %[[VAL_129]], %[[VAL_130]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:             %[[VAL_132:.*]] = constant 100 : index
  ! CHECK:             %[[VAL_133:.*]] = fir.shape %[[VAL_132]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_134:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_135:.*]] = fir.slice %[[VAL_134]], %[[VAL_132]], %[[VAL_134]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:             %[[VAL_136:.*]] = fir.array_load %[[VAL_131]](%[[VAL_133]]) {{\[}}%[[VAL_135]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK:             %[[VAL_137:.*]] = constant 100 : i64
  ! CHECK:             %[[VAL_138:.*]] = fir.convert %[[VAL_137]] : (i64) -> index
  ! CHECK:             %[[VAL_139:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_140:.*]] = fir.convert %[[VAL_139]] : (i32) -> i64
  ! CHECK:             %[[VAL_141:.*]] = fir.convert %[[VAL_140]] : (i64) -> index
  ! CHECK:             %[[VAL_142:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_143:.*]] = fir.convert %[[VAL_142]] : (i32) -> i64
  ! CHECK:             %[[VAL_144:.*]] = fir.convert %[[VAL_143]] : (i64) -> index
  ! CHECK:             %[[VAL_145:.*]] = fir.field_index data, !fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>
  ! CHECK:             %[[VAL_146:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_141]], %[[VAL_144]], %[[VAL_145]] {Fortran.offsets} : (!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, index, index, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  ! CHECK:             %[[VAL_147:.*]] = constant 100 : index
  ! CHECK:             %[[VAL_148:.*]] = fir.shape %[[VAL_147]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_149:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_150:.*]] = fir.slice %[[VAL_149]], %[[VAL_147]], %[[VAL_149]] : (index, index, index) -> !fir.slice<1>
  ! CHECK:             %[[VAL_151:.*]] = fir.array_load %[[VAL_146]](%[[VAL_148]]) {{\[}}%[[VAL_150]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK:             %[[VAL_152:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_153:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_154:.*]] = subi %[[VAL_138]], %[[VAL_152]] : index
  ! CHECK:             %[[VAL_155:.*]] = fir.do_loop %[[VAL_156:.*]] = %[[VAL_153]] to %[[VAL_154]] step %[[VAL_152]] unordered iter_args(%[[VAL_157:.*]] = %[[VAL_136]]) -> (!fir.array<100xf32>) {
  ! CHECK:               %[[VAL_158:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_159:.*]] = addi %[[VAL_156]], %[[VAL_158]] : index
  ! CHECK:               %[[VAL_160:.*]] = fir.array_coor %[[VAL_92]](%[[VAL_106]]) %[[VAL_159]] : (!fir.heap<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK:               %[[VAL_161:.*]] = fir.load %[[VAL_160]] : !fir.ref<!fir.logical<4>>
  ! CHECK:               %[[VAL_162:.*]] = fir.convert %[[VAL_161]] : (!fir.logical<4>) -> i1
  ! CHECK:               %[[VAL_163:.*]] = fir.if %[[VAL_162]] -> (!fir.array<100xf32>) {
  ! CHECK:                 fir.result %[[VAL_157]] : !fir.array<100xf32>
  ! CHECK:               } else {
  ! CHECK:                 %[[VAL_164:.*]] = fir.array_fetch %[[VAL_151]], %[[VAL_156]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:                 %[[VAL_165:.*]] = negf %[[VAL_164]] : f32
  ! CHECK:                 %[[VAL_166:.*]] = fir.array_update %[[VAL_157]], %[[VAL_165]], %[[VAL_156]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK:                 fir.result %[[VAL_166]] : !fir.array<100xf32>
  ! CHECK:               }
  ! CHECK:               fir.result %[[VAL_167:.*]] : !fir.array<100xf32>
  ! CHECK:             }
  ! CHECK:             fir.array_merge_store %[[VAL_136]], %[[VAL_168:.*]] to %[[VAL_131]]{{\[}}%[[VAL_135]]] : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>, !fir.slice<1>
  ! CHECK:             fir.freemem %[[VAL_92]] : !fir.heap<!fir.array<100x!fir.logical<4>>>
  ! CHECK:             fir.result %[[VAL_41]], %[[VAL_42]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_169:.*]]#0, %[[VAL_169]]#1 : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_170:.*]]#0 to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  ! CHECK:         fir.array_merge_store %[[VAL_5]], %[[VAL_170]]#1 to %[[VAL_0]] : !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>, !fir.box<!fir.array<?x?x!fir.type<_QFtest_nested_forall_whereTt{data:!fir.array<100xf32>}>>>
  ! CHECK:         return
  ! CHECK:       }
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
end subroutine test_nested_forall_where

! CHECK-LABEL: func @_QPtest_forall_with_slice(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>) {
subroutine test_forall_with_slice(i1,i2)
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "j"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_4:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_5:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_sliceEa"}
  ! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_6]](%[[VAL_7]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         %[[VAL_9:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK:         %[[VAL_11:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
  ! CHECK:         %[[VAL_13:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_14:.*]] = fir.do_loop %[[VAL_15:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_13]] unordered iter_args(%[[VAL_16:.*]] = %[[VAL_8]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_17]] to %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_18:.*]] = constant 1 : i32
  ! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK:           %[[VAL_20:.*]] = constant 10 : i32
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK:           %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_16]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:             fir.store %[[VAL_26]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
  ! CHECK:             %[[VAL_29:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_30:.*]] = subi %[[VAL_28]], %[[VAL_29]] : i64
  ! CHECK:             %[[VAL_31:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK:             %[[VAL_33:.*]] = constant 1 : i64
  ! CHECK:             %[[VAL_34:.*]] = subi %[[VAL_32]], %[[VAL_33]] : i64
  ! CHECK:             %[[VAL_35:.*]] = fir.coordinate_of %[[VAL_6]], %[[VAL_30]], %[[VAL_34]] : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>, i64, i64) -> !fir.ref<!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:             %[[VAL_36:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>
  ! CHECK:             %[[VAL_37:.*]] = fir.coordinate_of %[[VAL_35]], %[[VAL_36]] : (!fir.ref<!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.field) -> !fir.ref<!fir.array<11xi32>>
  ! CHECK:             %[[VAL_38:.*]] = constant 11 : index
  ! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (i32) -> i64
  ! CHECK:             %[[VAL_41:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> i64
  ! CHECK:             %[[VAL_43:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> i64
  ! CHECK:             %[[VAL_45:.*]] = fir.shape %[[VAL_38]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_46:.*]] = fir.slice %[[VAL_40]], %[[VAL_42]], %[[VAL_44]] : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK:             %[[VAL_47:.*]] = fir.array_load %[[VAL_37]](%[[VAL_45]]) {{\[}}%[[VAL_46]]] : (!fir.ref<!fir.array<11xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<11xi32>
  ! CHECK:             %[[VAL_48:.*]] = fir.call @_QPf(%[[VAL_3]]) : (!fir.ref<i32>) -> i32
  ! CHECK:             %[[VAL_49:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_50:.*]] = fir.convert %[[VAL_40]] : (i64) -> index
  ! CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_42]] : (i64) -> index
  ! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_44]] : (i64) -> index
  ! CHECK:             %[[VAL_53:.*]] = subi %[[VAL_51]], %[[VAL_50]] : index
  ! CHECK:             %[[VAL_54:.*]] = addi %[[VAL_53]], %[[VAL_52]] : index
  ! CHECK:             %[[VAL_55:.*]] = divi_signed %[[VAL_54]], %[[VAL_52]] : index
  ! CHECK:             %[[VAL_56:.*]] = cmpi sgt, %[[VAL_55]], %[[VAL_49]] : index
  ! CHECK:             %[[VAL_57:.*]] = select %[[VAL_56]], %[[VAL_55]], %[[VAL_49]] : index
  ! CHECK:             %[[VAL_58:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_59:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_60:.*]] = subi %[[VAL_57]], %[[VAL_58]] : index
  ! CHECK:             %[[VAL_61:.*]] = fir.do_loop %[[VAL_62:.*]] = %[[VAL_59]] to %[[VAL_60]] step %[[VAL_58]] unordered iter_args(%[[VAL_63:.*]] = %[[VAL_47]]) -> (!fir.array<11xi32>) {
  ! CHECK:               %[[VAL_64:.*]] = fir.array_update %[[VAL_63]], %[[VAL_48]], %[[VAL_62]] : (!fir.array<11xi32>, i32, index) -> !fir.array<11xi32>
  ! CHECK:               fir.result %[[VAL_64]] : !fir.array<11xi32>
  ! CHECK:             }
  ! CHECK:             fir.array_merge_store %[[VAL_47]], %[[VAL_65:.*]] to %[[VAL_37]]{{\[}}%[[VAL_46]]] : !fir.array<11xi32>, !fir.array<11xi32>, !fir.ref<!fir.array<11xi32>>, !fir.slice<1>
  ! CHECK:             fir.result %[[VAL_25]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_66:.*]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_67:.*]] to %[[VAL_6]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>>
  ! CHECK:         return
  ! CHECK:       }
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
end subroutine test_forall_with_slice

! CHECK-LABEL: func @_QPtest_forall_with_ranked_dimension() {
subroutine test_forall_with_ranked_dimension
  ! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_1:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_2:.*]] = constant 10 : index
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>> {bindc_name = "a", uniq_name = "_QFtest_forall_with_ranked_dimensionEa"}
  ! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_3]](%[[VAL_4]]) : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>) -> !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         %[[VAL_6:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
  ! CHECK:         %[[VAL_10:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_5]]) -> (!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>) {
  ! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_16:.*]] = constant 4 : i32
  ! CHECK:           %[[VAL_17:.*]] = addi %[[VAL_15]], %[[VAL_16]] : i32
  ! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
  ! CHECK:           %[[VAL_19:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
  ! CHECK:           %[[VAL_22:.*]] = fir.undefined index
  ! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
  ! CHECK:           %[[VAL_24:.*]] = subi %[[VAL_23]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_25:.*]] = addi %[[VAL_19]], %[[VAL_2]] : index
  ! CHECK:           %[[VAL_26:.*]] = subi %[[VAL_25]], %[[VAL_19]] : index
  ! CHECK:           %[[VAL_27:.*]] = constant 1 : i64
  ! CHECK:           %[[VAL_28:.*]] = fir.field_index arr, !fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>
  ! CHECK:           %[[VAL_29:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
  ! CHECK:           %[[VAL_30:.*]] = fir.slice %[[VAL_21]], %[[VAL_22]], %[[VAL_22]], %[[VAL_19]], %[[VAL_26]], %[[VAL_27]] path %[[VAL_28]], %[[VAL_18]] : (i64, index, index, index, index, i64, !fir.field, i64) -> !fir.slice<2>
  ! CHECK:           %[[VAL_31:.*]] = fir.array_load %[[VAL_3]](%[[VAL_29]]) {{\[}}%[[VAL_30]]] : (!fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x10xi32>
  ! CHECK:           %[[VAL_32:.*]] = constant 10 : i64
  ! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK:           %[[VAL_34:.*]] = fir.call @_QPf(%[[VAL_0]]) : (!fir.ref<i32>) -> i32
  ! CHECK:           %[[VAL_35:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_36:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_37:.*]] = subi %[[VAL_33]], %[[VAL_35]] : index
  ! CHECK:           %[[VAL_38:.*]] = fir.do_loop %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_37]] step %[[VAL_35]] unordered iter_args(%[[VAL_40:.*]] = %[[VAL_31]]) -> (!fir.array<10x10xi32>) {
  ! CHECK:             %[[VAL_41:.*]] = fir.array_update %[[VAL_40]], %[[VAL_34]], %[[VAL_24]], %[[VAL_39]] : (!fir.array<10x10xi32>, i32, index, index) -> !fir.array<10x10xi32>
  ! CHECK:             fir.result %[[VAL_41]] : !fir.array<10x10xi32>
  ! CHECK:           }
  ! CHECK:           fir.array_merge_store %[[VAL_31]], %[[VAL_42:.*]] to %[[VAL_3]]{{\[}}%[[VAL_30]]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>, !fir.slice<2>
  ! CHECK:           fir.result %[[VAL_13]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_5]], %[[VAL_43:.*]] to %[[VAL_3]] : !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>, !fir.ref<!fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>>
  ! CHECK:         return
  ! CHECK:       }
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
end subroutine test_forall_with_ranked_dimension

! CHECK-LABEL: func @_QPforall_with_allocatable(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine forall_with_allocatable(a1)
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "arr", uniq_name = "_QFforall_with_allocatableEarr"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QFforall_with_allocatableEarr.addr"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.lb0"}
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFforall_with_allocatableEarr.ext0"}
  ! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  ! CHECK:         fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
  ! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
  ! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:         %[[VAL_10:.*]] = fir.shape_shift %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_13:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
  ! CHECK:         %[[VAL_15:.*]] = constant 15 : i32
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
  ! CHECK:         %[[VAL_17:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_17]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_11]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_21]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i32) -> i64
  ! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i64) -> index
  ! CHECK:           %[[VAL_25:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_24]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
  ! CHECK:           %[[VAL_29:.*]] = fir.array_update %[[VAL_20]], %[[VAL_25]], %[[VAL_28]] {Fortran.offsets} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_29]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_30:.*]] to %[[VAL_9]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
  ! CHECK:         return
  ! CHECK:       }
  real :: a1(:)
  real, allocatable :: arr(:)
  forall (i=5:15)
     arr(i) = a1(i)
  end forall
end subroutine forall_with_allocatable

! CHECK-LABEL: func @_QPforall_with_allocatable2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>) {
subroutine forall_with_allocatable2(a1)
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, uniq_name = "i"}
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {bindc_name = "thing", uniq_name = "_QFforall_with_allocatable2Ething"}
  ! CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_4:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_5:.*]] = constant {{.*}} : i32
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAInitialize(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_9:.*]] = fir.field_index arr, !fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
  ! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_9]] : (!fir.ref<!fir.type<_QFforall_with_allocatable2Tt{i:i32,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_12:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_11]], %[[VAL_12]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_14:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK:         %[[VAL_15:.*]] = fir.shape_shift %[[VAL_13]]#0, %[[VAL_13]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_14]](%[[VAL_15]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK:         %[[VAL_18:.*]] = constant 5 : i32
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
  ! CHECK:         %[[VAL_20:.*]] = constant 15 : i32
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
  ! CHECK:         %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_22]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_16]]) -> (!fir.array<?xf32>) {
  ! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_24]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_26]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
  ! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
  ! CHECK:           %[[VAL_30:.*]] = fir.array_fetch %[[VAL_17]], %[[VAL_29]] {Fortran.offsets} : (!fir.array<?xf32>, index) -> f32
  ! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
  ! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
  ! CHECK:           %[[VAL_34:.*]] = fir.array_update %[[VAL_25]], %[[VAL_30]], %[[VAL_33]] {Fortran.offsets} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK:           fir.result %[[VAL_34]] : !fir.array<?xf32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_16]], %[[VAL_35:.*]] to %[[VAL_14]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.heap<!fir.array<?xf32>>
  ! CHECK:         return
  ! CHECK:       }
  real :: a1(:)
  type t
     integer :: i
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing
  forall (i=5:15)
     thing%arr(i) = a1(i)
  end forall
end subroutine forall_with_allocatable2
