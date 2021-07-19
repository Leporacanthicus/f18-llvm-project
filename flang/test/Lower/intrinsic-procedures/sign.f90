! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: sign_testi
subroutine sign_testi(a, b, c)
  integer a, b, c
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  ! CHECK-DAG: subi
  ! CHECK-DAG: cmpi slt
  ! CHECK: select
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr
subroutine sign_testr(a, b, c)
  real a, b, c
  ! CHECK-DAG: fir.call {{.*}}fabs
  ! CHECK-DAG: negf
  ! CHECK-DAG: cmpf olt
  ! CHECK: select
  c = sign(a, b)
end subroutine

