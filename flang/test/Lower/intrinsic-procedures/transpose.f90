! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: transpose_test
! CHECK-SAME: (%[[source:.*]]: !fir.ref<!fir.array<2x3xf32>>)
subroutine transpose_test(mat)
! CHECK-DAG:  %[[resultDescr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {uniq_name = ""}
   real :: mat(2,3)
   call bar_transpose_test(transpose(mat))
! CHECK-DAG:  %[[sourceBox:.*]] = fir.embox %[[source]]({{.*}}) : (!fir.ref<!fir.array<2x3xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<2x3xf32>>
! CHECK-DAG:  %[[zeroArray:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>
! CHECK-DAG:  %[[c0:.*]] = constant 0 : index
! CHECK-DAG:  %[[shapeResult:.*]] = fir.shape %[[c0]], %[[c0]] : (index, index) -> !fir.shape<2>
! CHECK-DAG:  %[[resultBox:.*]] = fir.embox %[[zeroArray]](%[[shapeResult]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK-DAG:  fir.store %[[resultBox]] to %[[resultDescr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK-DAG:  %[[resultOpaque:.*]] = fir.convert %[[resultDescr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[sourceOpaque:.*]] = fir.convert %[[sourceBox]] : (!fir.box<!fir.array<2x3xf32>>) -> !fir.box<none>
! CHECK-DAG:  %{{.*}} = fir.call @_FortranATranspose(%[[resultOpaque]], %[[sourceOpaque]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[tmp1:.*]] = fir.load %[[resultDescr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK-DAG:  %[[tmp2:.*]] = fir.box_addr %[[tmp1]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK-DAG:  %[[tmp3:.*]] = fir.convert %[[tmp2]] : (!fir.heap<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<3x2xf32>>
! CHECK-DAG:  fir.call @_QPbar_transpose_test(%[[tmp3]]) : (!fir.ref<!fir.array<3x2xf32>>) -> ()
! CHECK-DAG:  fir.freemem %[[tmp2]] : !fir.heap<!fir.array<?x?xf32>
end subroutine

