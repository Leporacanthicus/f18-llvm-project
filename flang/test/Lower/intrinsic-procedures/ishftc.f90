! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ishftc_test
function ishftc_test(i, j, k)
  ! CHECK-DAG: %[[result:.*]] = fir.alloca i32 {bindc_name = "ishftc_test"
  ! CHECK-DAG: %[[i:.*]] = fir.load %arg0 : !fir.ref<i32>
  ! CHECK-DAG: %[[j:.*]] = fir.load %arg1 : !fir.ref<i32>
  ! CHECK-DAG: %[[k:.*]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK-DAG: %[[VAL_7:.*]] = arith.constant 32 : i32
  ! CHECK-DAG: %[[VAL_8:.*]] = arith.constant 0 : i32
  ! CHECK-DAG: %[[VAL_9:.*]] = arith.constant -1 : i32
  ! CHECK-DAG: %[[VAL_10:.*]] = arith.constant 31 : i32
  ! CHECK: %[[VAL_11:.*]] = shift_right_signed %[[j]], %[[VAL_10]] : i32
  ! CHECK: %[[VAL_12:.*]] = xor %[[j]], %[[VAL_11]] : i32
  ! CHECK: %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_11]] : i32
  ! CHECK: %[[VAL_14:.*]] = arith.subi %[[k]], %[[VAL_13]] : i32
  ! CHECK: %[[VAL_15:.*]] = arith.cmpi eq, %[[j]], %[[VAL_8]] : i32
  ! CHECK: %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_13]], %[[k]] : i32
  ! CHECK: %[[VAL_17:.*]] = or %[[VAL_15]], %[[VAL_16]] : i1
  ! CHECK: %[[VAL_18:.*]] = arith.cmpi sgt, %[[j]], %[[VAL_8]] : i32
  ! CHECK: %[[VAL_19:.*]] = select %[[VAL_18]], %[[VAL_13]], %[[VAL_14]] : i32
  ! CHECK: %[[VAL_20:.*]] = select %[[VAL_18]], %[[VAL_14]], %[[VAL_13]] : i32
  ! CHECK: %[[VAL_21:.*]] = arith.cmpi ne, %[[k]], %[[VAL_7]] : i32
  ! CHECK: %[[VAL_22:.*]] = shift_right_unsigned %[[i]], %[[k]] : i32
  ! CHECK: %[[VAL_23:.*]] = shift_left %[[VAL_22]], %[[k]] : i32
  ! CHECK: %[[VAL_24:.*]] = select %[[VAL_21]], %[[VAL_23]], %[[VAL_8]] : i32
  ! CHECK: %[[VAL_25:.*]] = arith.subi %[[VAL_7]], %[[VAL_19]] : i32
  ! CHECK: %[[VAL_26:.*]] = shift_right_unsigned %[[VAL_9]], %[[VAL_25]] : i32
  ! CHECK: %[[VAL_27:.*]] = shift_right_unsigned %[[i]], %[[VAL_20]] : i32
  ! CHECK: %[[VAL_28:.*]] = and %[[VAL_27]], %[[VAL_26]] : i32
  ! CHECK: %[[VAL_29:.*]] = arith.subi %[[VAL_7]], %[[VAL_20]] : i32
  ! CHECK: %[[VAL_30:.*]] = shift_right_unsigned %[[VAL_9]], %[[VAL_29]] : i32
  ! CHECK: %[[VAL_31:.*]] = and %[[i]], %[[VAL_30]] : i32
  ! CHECK: %[[VAL_32:.*]] = shift_left %[[VAL_31]], %[[VAL_19]] : i32
  ! CHECK: %[[VAL_33:.*]] = or %[[VAL_24]], %[[VAL_28]] : i32
  ! CHECK: %[[VAL_34:.*]] = or %[[VAL_33]], %[[VAL_32]] : i32
  ! CHECK: %[[VAL_35:.*]] = select %[[VAL_17]], %[[i]], %[[VAL_34]] : i32
  ! CHECK: fir.store %[[VAL_35]] to %[[result]] : !fir.ref<i32>
  ! CHECK: %[[VAL_36:.*]] = fir.load %[[result]] : !fir.ref<i32>
  ! CHECK: return %[[VAL_36]] : i32
  ishftc_test = ishftc(i, j, k)
end
