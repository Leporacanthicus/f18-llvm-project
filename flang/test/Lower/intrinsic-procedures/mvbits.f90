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

