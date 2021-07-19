! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: trans_test
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<i32>, %[[arg1:.*]]: !fir.ref<f32>)
subroutine trans_test(store, word)
  integer :: store
  real :: word
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {uniq_name = ""}
! CHECK-DAG:  %[[a1:.*]] = fir.embox %[[arg1]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK-DAG:  %[[a2:.*]] = fir.embox %[[arg0]] : (!fir.ref<i32>) -> !fir.box<i32>

! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[a1]] : (!fir.box<f32>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a2]] : (!fir.box<i32>) -> !fir.box<none>
  store = transfer(word, store)
! CHECK:  %{{.*}} = fir.call @_FortranATransfer(%[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[a11:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-DAG:  %[[a12:.*]] = fir.box_addr %[[a11]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK-DAG:  fir.freemem %[[a12]] : !fir.heap<i32>
end subroutine

! CHECK-LABEL: trans_test2
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<!fir.array<3xi32>>, %[[arg1:.*]]: !fir.ref<f32>)
subroutine trans_test2(store, word)
  integer :: store(3)
  real :: word
! CHECK-DAG:  %[[c3_i32:.*]] = constant 3 : i32
! CHECK-DAG:  %[[c3:.*]] = constant 3 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {uniq_name = ""}
! CHECK-DAG:  %[[a1:.*]] = fir.shape %[[c3]] : (index) -> !fir.shape<1>
! CHECK-DAG:  %[[a2:.*]] = fir.embox %[[arg1]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK-DAG:  %[[a3:.*]] = fir.embox %[[arg0]](%{{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  store = transfer(word, store, 3)
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a2]] : (!fir.box<f32>) -> !fir.box<none>
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a3]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a12:.*]] = fir.convert %[[c3_i32]] : (i32) -> i64
! CHECK:  %{{.*}} = fir.call @_FortranATransferSize(%[[a8]], %[[a9]], %[[a10]], %{{.*}}, %{{.*}}, %[[a12]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32, i64) -> none
! CHECK-DAG:  %[[a14:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:  %[[a16:.*]] = fir.box_addr %[[a14]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK-DAG:  fir.freemem %[[a16]] : !fir.heap<!fir.array<?xi32>>
end subroutine

! CHECK-LABEL: trans_test3
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>) -> i32
integer function trans_test3(p)
  type obj
    integer :: x
  end type
  type (obj) :: t
  integer :: p
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>> {uniq_name = ""}
! CHECK-DAG:  %[[a1:.*]] = fir.alloca !fir.type<_QFtrans_test3Tobj{x:i32}> {bindc_name = "t", uniq_name = "_QFtrans_test3Et"}
! CHECK-DAG:  %[[a3:.*]] = fir.embox %[[arg0]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK-DAG:  %[[a4:.*]] = fir.embox %[[a1]] : (!fir.ref<!fir.type<_QFtrans_test3Tobj{x:i32}>>) -> !fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a9:.*]] = fir.convert %[[a3]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a4]] : (!fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>) -> !fir.box<none>
  t = transfer(p, t)
! CHECK:  %{{.*}} = fir.call @_FortranATransfer(%[[a8]], %[[a9]], %[[a10]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[a13:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>>
! CHECK-DAG: %[[a14:.*]] = fir.box_addr %[[a13]] : (!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>) -> !fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>
! CHECK-DAG:  fir.freemem %[[a14]] : !fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  trans_test3 = t%x
end function
