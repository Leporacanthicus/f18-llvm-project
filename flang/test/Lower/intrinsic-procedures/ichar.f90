! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ichar_test
subroutine ichar_test(c)
  character(1) :: c
  character :: str(10)
  ! CHECK-DAG: %[[unbox:.*]]:2 = fir.unboxchar
  ! CHECK-DAG: %[[J:.*]] = fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Ej"}
  ! CHECK-DAG: %[[STR:.*]] = fir.alloca !fir.array{{.*}} {{{.*}}uniq_name = "{{.*}}Estr"}
  ! CHECK: %[[BOX:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.load %[[BOX]] : !fir.ref<!fir.char<1>>
  ! CHECK: fir.extract_value %[[CHAR]], [0 : index] :
  print *, ichar(c)
  ! CHECK: fir.call @{{.*}}EndIoStatement

  ! CHECK-DAG: %{{.*}} = fir.load %[[J]] : !fir.ref<i32>
  ! CHECK: %[[ptr:.*]] = fir.coordinate_of %[[STR]], %
  ! CHECK: %[[VAL:.*]] = fir.load %[[ptr]] : !fir.ref<!fir.char<1>>
  ! CHECK: fir.extract_value %[[VAL]], [0 : index] :
  print *, ichar(str(J))
  ! CHECK: fir.call @{{.*}}EndIoStatement
end subroutine

