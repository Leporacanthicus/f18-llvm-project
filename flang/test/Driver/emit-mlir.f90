! RUN: %flang_fc1 -emit-mlir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK: module attributes {
! CHECK-LABEL: func @_QQmain() {
! CHECK-NEXT:  return
! CHECK-NEXT: }
! CHECK-NEXT: }

end program
