! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: sign_testi
subroutine sign_testi(a, b, c)
  integer a, b, c
  ! CHECK: arith.shrsi
  ! CHECK: arith.xori
  ! CHECK: arith.subi
  ! CHECK-DAG: arith.subi
  ! CHECK-DAG: arith.cmpi slt
  ! CHECK: select
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr
subroutine sign_testr(a, b, c)
  real a, b, c
  ! CHECK-DAG: fir.call {{.*}}fabs
  ! CHECK-DAG: arith.negf
  ! CHECK-DAG: arith.cmpf olt
  ! CHECK: select
  c = sign(a, b)
end subroutine

