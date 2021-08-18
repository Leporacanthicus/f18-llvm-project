! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: eoshift_test1
subroutine eoshift_test1(arr, shift)
  logical, dimension(3) :: arr, res
  integer :: shift
! CHECK: %[[boundAlloc:.*]] = fir.alloca !fir.logical<4> {uniq_name = ""}
! CHECK: %[[resBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>> {uniq_name = ""}
! CHECK: %[[res:.*]] = fir.alloca !fir.array<3x!fir.logical<4>> {bindc_name = "res", uniq_name = "_QFeoshift_test1Eres"}
! CHECK: %[[resLoad:.*]] = fir.array_load %[[res]]({{.*}}) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
! CHECK: %[[arr:.*]] = fir.embox %arg0({{.*}}) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.logical<4>>>
! CHECK: %[[bits:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK: %[[init:.*]] = fir.embox %[[bits]]({{.*}}) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK: fir.store %[[init]] to %[[resBox]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK: %[[zero:.*]] = constant 0 : index
! CHECK: %[[false:.*]] = fir.convert %[[zero]] : (index) -> !fir.logical<4>
! CHECK: fir.store %[[false]] to %[[boundAlloc]] : !fir.ref<!fir.logical<4>>
! CHECK: %[[boundBox:.*]] = fir.embox %[[boundAlloc]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
! CHECK: %[[shift:.*]] = fir.load %arg1 : !fir.ref<i32>

  res = eoshift(arr, shift)

! CHECK: %[[resIRBox:.*]] = fir.convert %[[resBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[arrBox:.*]] = fir.convert %[[arr]] : (!fir.box<!fir.array<3x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[shiftBox:.*]] = fir.convert %[[shift]] : (i32) -> i64
! CHECK: %[[boundBoxNone:.*]] = fir.convert %[[boundBox]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
! CHECK: %[[tmp:.*]] = fir.call @_FortranAEoshiftVector(%[[resIRBox]], %[[arrBox]], %[[shiftBox]], %[[boundBoxNone]], {{.*}}, {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i64, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK: fir.array_merge_store %[[resLoad]], {{.*}} to %[[res]] : !fir.array<3x!fir.logical<4>>, !fir.array<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
end subroutine eoshift_test1

! CHECK-LABEL: eoshift_test2
subroutine eoshift_test2(arr, shift, bound, dim)
  integer, dimension(3,3) :: arr, res
  integer, dimension(3) :: shift
  integer :: bound, dim
! CHECK: %[[boundAlloc:.*]] = fir.alloca i32 {uniq_name = ""}
! CHECK: %[[resBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {uniq_name = ""}
! CHECK: %[[res:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "res", uniq_name = "_QFeoshift_test2Eres"}
!CHECK: %[[resLoad:.*]] = fir.array_load %[[res]]({{.*}}) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.array<3x3xi32>
! CHECK: %[[bound:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK: %[[dim:.*]] = fir.load %arg3 : !fir.ref<i32>
  
  res = eoshift(arr, shift, bound, dim)

! CHECK: %[[arr:.*]] = fir.embox %arg0({{.*}}) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
! CHECK:  fir.store %[[bound]] to %[[boundAlloc]] : !fir.ref<i32>
! CHECK: %[[boundBox:.*]] = fir.embox %[[boundAlloc]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[shiftBox:.*]] = fir.embox %arg1({{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
! CHECK: %[[resIRBox:.*]] = fir.convert %[[resBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[arrBox:.*]] = fir.convert %[[arr]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.box<none>
! CHECK: %[[shiftBoxNone:.*]] = fir.convert %[[shiftBox]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
! CHECK: %[[boundBoxNone:.*]] = fir.convert %[[boundBox]] : (!fir.box<i32>) -> !fir.box<none>

! CHECK: %[[tmp:.*]] = fir.call @_FortranAEoshift(%[[resIRBox]], %[[arrBox]], %[[shiftBoxNone]], %[[boundBoxNone]], %[[dim]], {{.*}}, {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
! CHECK: fir.array_merge_store %[[resLoad]], {{.*}} to %[[res]] : !fir.array<3x3xi32>, !fir.array<3x3xi32>, !fir.ref<!fir.array<3x3xi32>>
end subroutine eoshift_test2

! CHECK-LABEL: eoshift_test3
subroutine eoshift_test3(arr, shift, dim)
  character(4), dimension(3,3) :: arr, res
  integer :: shift, dim

! CHECK: %[[resBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>> {uniq_name = ""}
! CHECK: %[[arr:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK: %[[array:.*]] = fir.convert %[[arr]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x3x!fir.char<1,4>>>
! CHECK: %[[res:.*]] = fir.alloca !fir.array<3x3x!fir.char<1,4>> {bindc_name = "res", uniq_name = "_QFeoshift_test3Eres"}
! CHECK: %[[resLoad:.*]] = fir.array_load %[[res]]({{.*}}) : (!fir.ref<!fir.array<3x3x!fir.char<1,4>>>, !fir.shape<2>) -> !fir.array<3x3x!fir.char<1,4>>
! CHECK: %[[dim:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK: %[[arrayBox:.*]] = fir.embox %[[array]]({{.*}}) : (!fir.ref<!fir.array<3x3x!fir.char<1,4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.char<1,4>>>

  res = eoshift(arr, SHIFT=shift, DIM=dim)

! CHECK: %[[boundAlloc:.*]] = fir.alloca !fir.char<1,4>
! CHECK: %[[zero:.*]] = constant 0 : index
! CHECK: %[[len:.*]] = constant 4 : index
! CHECK: %[[blankVal:.*]] = constant 32 : i8
! CHECK: %[[single:.*]] = fir.undefined !fir.char<1>
! CHECK: %[[blank:.*]] = fir.insert_value %[[single]], %[[blankVal]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK: %[[one:.*]] = constant 1 : index
! CHECK: fir.do_loop %arg3 = %[[zero]] to %[[len]] step %[[one]] {
! CHECK:    %[[bound:.*]] = fir.convert %[[boundAlloc]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<!fir.array<4x!fir.char<1>>>
! CHECK:    %[[index:.*]] = fir.coordinate_of %[[bound]], %arg3 : (!fir.ref<!fir.array<4x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
!CHECK:    fir.store %[[blank]] to %[[index]] : !fir.ref<!fir.char<1>>
! CHECK: %[[boundBox:.*]] = fir.embox %[[boundAlloc]] : (!fir.ref<!fir.char<1,4>>) -> !fir.box<!fir.char<1,4>>
! CHECK: %[[shiftBox:.*]] = fir.embox %arg1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK: %[[resIRBox:.*]] = fir.convert %[[resBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[arrayBoxNone:.*]] = fir.convert %[[arrayBox]] : (!fir.box<!fir.array<3x3x!fir.char<1,4>>>) -> !fir.box<none>
! CHECK: %[[shiftBoxNone:.*]] = fir.convert %[[shiftBox]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK: %[[boundBoxNone:.*]] = fir.convert %[[boundBox]] : (!fir.box<!fir.char<1,4>>) -> !fir.box<none>
! CHECK: %[[tmp:.*]] = fir.call @_FortranAEoshift(%[[resIRBox]], %[[arrayBoxNone]], %[[shiftBoxNone]], %[[boundBoxNone]], %[[dim]], {{.*}}, {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
! CHECK: fir.array_merge_store %[[resLoad]], {{.*}} to %[[res]] : !fir.array<3x3x!fir.char<1,4>>, !fir.array<3x3x!fir.char<1,4>>, !fir.ref<!fir.array<3x3x!fir.char<1,4>>>
end subroutine eoshift_test3
