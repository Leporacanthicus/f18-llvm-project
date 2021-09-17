! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: mvbits_test
function mvbits_test(from, frompos, len, to, topos)
  ! CHECK: %[[result:.*]] = fir.alloca i32 {bindc_name = "mvbits_test"
  ! CHECK-DAG: %[[from:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[frompos:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[len:.*]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK-DAG: %[[to:.*]] = fir.load %arg3 : !fir.ref<i32>
  ! CHECK-DAG: %[[topos:.*]] = fir.load %arg4 : !fir.ref<i32>
  integer :: from, frompos, len, to, topos
  integer :: mvbits_test
  ! CHECK: %[[VAL_11:.*]] = constant 0 : i32
  ! CHECK: %[[VAL_12:.*]] = constant -1 : i32
  ! CHECK: %[[VAL_13:.*]] = constant 32 : i32
  ! CHECK: %[[VAL_14:.*]] = subi %[[VAL_13]], %[[len]] : i32
  ! CHECK: %[[VAL_15:.*]] = shift_right_unsigned %[[VAL_12]], %[[VAL_14]] : i32
  ! CHECK: %[[VAL_16:.*]] = shift_left %[[VAL_15]], %[[topos]] : i32
  ! CHECK: %[[VAL_17:.*]] = xor %[[VAL_16]], %[[VAL_12]] : i32
  ! CHECK: %[[VAL_18:.*]] = and %[[VAL_17]], %[[to]] : i32
  ! CHECK: %[[VAL_19:.*]] = shift_right_unsigned %[[from]], %[[frompos]] : i32
  ! CHECK: %[[VAL_20:.*]] = and %[[VAL_19]], %[[VAL_15]] : i32
  ! CHECK: %[[VAL_21:.*]] = shift_left %[[VAL_20]], %[[topos]] : i32
  ! CHECK: %[[VAL_22:.*]] = or %[[VAL_18]], %[[VAL_21]] : i32
  ! CHECK: %[[VAL_23:.*]] = cmpi eq, %[[len]], %[[VAL_11]] : i32
  ! CHECK: %[[VAL_24:.*]] = select %[[VAL_23]], %[[to]], %[[VAL_22]] : i32
  ! CHECK: fir.store %[[VAL_24]] to %arg3 : !fir.ref<i32>
  ! CHECK: %[[VAL_25:.*]] = fir.load %arg3 : !fir.ref<i32>
  ! CHECK: fir.store %[[VAL_25]] to %[[result]] : !fir.ref<i32>
  call mvbits(from, frompos, len, to, topos)
  ! CHECK: %[[VAL_26:.*]] = fir.load %[[result]] : !fir.ref<i32>
  ! CHECK: return %[[VAL_26]] : i32
  mvbits_test = to
end

! CHECK-LABEL: func @_QPmvbits_array_test(
! CHECK-SAME:                             %[[VAL_0:[^:]+]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME:                             %[[VAL_1:[^:]+]]: !fir.ref<i32>,
! CHECK-SAME:                             %[[VAL_2:[^:]+]]: !fir.ref<i32>,
! CHECK-SAME:                             %[[VAL_3:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME:                             %[[VAL_4:.*]]: !fir.ref<i32>) {
subroutine mvbits_array_test(from, frompos, len, to, topos)
  integer :: from(:), frompos, len, to(:), topos

  call mvbits(from, frompos, len, to, topos)
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
  ! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_9:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_9]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_11:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_12:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_13:.*]] = subi %[[VAL_10]]#1, %[[VAL_11]] : index
  ! CHECK:         fir.do_loop %[[VAL_14:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_11]] {
  ! CHECK:           %[[VAL_15:.*]] = fir.array_fetch %[[VAL_5]], %[[VAL_14]] : (!fir.array<?xi32>, index) -> i32
  ! CHECK:           %[[VAL_16:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_17:.*]] = addi %[[VAL_14]], %[[VAL_16]] : index
  ! CHECK:           %[[VAL_18:.*]] = fir.array_coor %[[VAL_3]] %[[VAL_17]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  ! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_20:.*]] = constant 0 : i32
  ! CHECK:           %[[VAL_21:.*]] = constant -1 : i32
  ! CHECK:           %[[VAL_22:.*]] = constant 32 : i32
  ! CHECK:           %[[VAL_23:.*]] = subi %[[VAL_22]], %[[VAL_7]] : i32
  ! CHECK:           %[[VAL_24:.*]] = shift_right_unsigned %[[VAL_21]], %[[VAL_23]] : i32
  ! CHECK:           %[[VAL_25:.*]] = shift_left %[[VAL_24]], %[[VAL_8]] : i32
  ! CHECK:           %[[VAL_26:.*]] = xor %[[VAL_25]], %[[VAL_21]] : i32
  ! CHECK:           %[[VAL_27:.*]] = and %[[VAL_26]], %[[VAL_19]] : i32
  ! CHECK:           %[[VAL_28:.*]] = shift_right_unsigned %[[VAL_15]], %[[VAL_6]] : i32
  ! CHECK:           %[[VAL_29:.*]] = and %[[VAL_28]], %[[VAL_24]] : i32
  ! CHECK:           %[[VAL_30:.*]] = shift_left %[[VAL_29]], %[[VAL_8]] : i32
  ! CHECK:           %[[VAL_31:.*]] = or %[[VAL_27]], %[[VAL_30]] : i32
  ! CHECK:           %[[VAL_32:.*]] = cmpi eq, %[[VAL_7]], %[[VAL_20]] : i32
  ! CHECK:           %[[VAL_33:.*]] = select %[[VAL_32]], %[[VAL_19]], %[[VAL_31]] : i32
  ! CHECK:           fir.store %[[VAL_33]] to %[[VAL_18]] : !fir.ref<i32>
  ! CHECK:         }
  ! CHECK:         return
  ! CHECK:       }
end subroutine
