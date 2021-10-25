! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: modulo_testr
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<f64>, %[[arg1:.*]]: !fir.ref<f64>, %[[arg2:.*]]: !fir.ref<f64>)
subroutine modulo_testr(r, a, p)
  real(8) :: r, a, p
  ! CHECK-DAG: %[[a:.*]] = fir.load %[[arg1]] : !fir.ref<f64>
  ! CHECK-DAG: %[[p:.*]] = fir.load %[[arg2]] : !fir.ref<f64>
  ! CHECK-DAG: %[[rem:.*]] = remf %[[a]], %[[p]] : f64
  ! CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
  ! CHECK-DAG: %[[remNotZero:.*]] = cmpf une, %[[rem]], %[[zero]] : f64
  ! CHECK-DAG: %[[aNeg:.*]] = cmpf olt, %[[a]], %[[zero]] : f64
  ! CHECK-DAG: %[[pNeg:.*]] = cmpf olt, %[[p]], %[[zero]] : f64
  ! CHECK-DAG: %[[signDifferent:.*]] = xor %[[aNeg]], %[[pNeg]] : i1
  ! CHECK-DAG: %[[mustAddP:.*]] = and %[[remNotZero]], %[[signDifferent]] : i1
  ! CHECK-DAG: %[[remPlusP:.*]] = arith.addf %[[rem]], %[[p]] : f64
  ! CHECK: %[[res:.*]] = select %[[mustAddP]], %[[remPlusP]], %[[rem]] : f64
  ! CHECK: fir.store %[[res]] to %[[arg0]] : !fir.ref<f64>
  r = modulo(a, p)
end subroutine

! CHECK-LABEL: modulo_testi
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<i64>, %[[arg1:.*]]: !fir.ref<i64>, %[[arg2:.*]]: !fir.ref<i64>)
subroutine modulo_testi(r, a, p)
  integer(8) :: r, a, p
  ! CHECK-DAG: %[[a:.*]] = fir.load %[[arg1]] : !fir.ref<i64>
  ! CHECK-DAG: %[[p:.*]] = fir.load %[[arg2]] : !fir.ref<i64>
  ! CHECK-DAG: %[[rem:.*]] = remi_signed %[[a]], %[[p]] : i64
  ! CHECK-DAG: %[[argXor:.*]] = xor %[[a]], %[[p]] : i64
  ! CHECK-DAG: %[[signDifferent:.*]] = arith.cmpi slt, %[[argXor]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[remNotZero:.*]] = arith.cmpi ne, %[[rem]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[mustAddP:.*]] = and %[[remNotZero]], %[[signDifferent]] : i1
  ! CHECK-DAG: %[[remPlusP:.*]] = arith.addi %[[rem]], %[[p]] : i64
  ! CHECK: %[[res:.*]] = select %[[mustAddP]], %[[remPlusP]], %[[rem]] : i64
  ! CHECK: fir.store %[[res]] to %[[arg0]] : !fir.ref<i64>
  r = modulo(a, p)
end subroutine

