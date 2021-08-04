! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: pack_test
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME: %[[arg2:[^:]+]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[arg3:[^:]+]]: !fir.box<!fir.array<?xi32>>
subroutine pack_test(a,m,v,r)
  integer :: a(:)
  logical :: m(:)
  integer :: v(:)
  integer :: r(:)
! CHECK:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {uniq_name = ""}
! CHECK:  %[[a5:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[a6:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[a7:.*]] = fir.convert %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK:  %[[a8:.*]] = fir.convert %[[arg2]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  r = pack(a,m,v)
! CHECK: %{{.*}} = fir.call @_FortranAPack(%[[a5]], %[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:  %[[a11:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[a13:.*]] = fir.box_addr %[[a11]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:  fir.freemem %[[a13]] : !fir.heap<!fir.array<?xi32>>
end subroutine
