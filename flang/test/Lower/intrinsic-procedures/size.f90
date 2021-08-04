! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPsize_test() {
subroutine size_test()
  real, dimension(1:10, -10:10) :: a
  !CHECK-DAG: %[[sz0:.*]] = fir.alloca !fir.array<10x21xf32> {bindc_name = "a", uniq_name = "_QFsize_testEa"}
  integer :: dim = 1
  !CHECK-DAG: %[[sz1:.*]] = fir.address_of({{.*}}) : !fir.ref<i32>
  integer :: iSize
  !CHECK-DAG: %[[sz2:.*]] = fir.alloca i32 {bindc_name = "isize", uniq_name = "{{.*}}"}
  iSize = size(a(2:5, -1:1), dim, 8)
  !CHECK: %[[sz3:.*]] = fir.convert %{{.*}} : (i64) -> index
  !CHECK: %[[sz4:.*]] = fir.convert %{{.*}} : (i64) -> index
  !CHECK: %[[sz5:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index) -> !fir.shapeshift<2>
  !CHECK: %[[sz6:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}-1_i64, %{{.*}}, %{{.*}} : (i64, i64, i64, i64, i64, i64) -> !fir.slice<2>
  !CHECK: %[[sz7:.*]] = fir.array_load %[[sz0]](%[[sz5]]) [%[[sz6]]] : (!fir.ref<!fir.array<10x21xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.array<10x21xf32>
  !CHECK: %[[sz8:.*]] = fir.allocmem !fir.array<4x3xf32>
  !CHECK: %[[sz9:.*]] = fir.shape %[[sz3]], %[[sz4]] : (index, index) -> !fir.shape<2>
  !CHECK: %[[sz10:.*]] = fir.array_load %[[sz8]](%[[sz9]]) : (!fir.heap<!fir.array<4x3xf32>>, !fir.shape<2>) -> !fir.array<4x3xf32>
  !CHECK: %[[sz11:.*]] = subi %[[sz3]], %{{.*}} : index
  !CHECK: %[[sz12:.*]] = subi %[[sz4]], %{{.*}} : index
  !CHECK: %[[sz13:.*]] = fir.do_loop %arg0 = %{{.*}} to %[[sz12]] step %{{.*}} iter_args(%arg1 = %[[sz10]]) -> (!fir.array<4x3xf32>) {
  !CHECK %[[sz22:.*]] = fir.do_loop %arg2 = %{{.*}} to %[[sz11]] step %{{.*}} iter_args(%arg3 = %arg1) -> (!fir.array<4x3xf32>) {
  !CHECK: %[[sz23:.*]] = fir.array_fetch %[[sz7]], %arg2, %arg0 : (!fir.array<10x21xf32>, index, index) -> f32
  !CHECK: %[[sz24:.*]] = fir.array_update %arg3, %[[sz23]], %arg2, %arg0 : (!fir.array<4x3xf32>, f32, index, index) -> !fir.array<4x3xf32>
  !CHECK: fir.result %[[sz24]] : !fir.array<4x3xf32>
  !CHECK: fir.result %{{.*}} : !fir.array<4x3xf32>
  !CHECK: fir.array_merge_store %[[sz10]], %[[sz13]] to %[[sz8]] : !fir.array<4x3xf32>, !fir.array<4x3xf32>, !fir.heap<!fir.array<4x3xf32>>
  !CHECK: %[[sz14:.*]] = fir.load %[[sz1]] : !fir.ref<i32>
  !CHECK: %[[sz15:.*]] = fir.shape %[[sz3]], %[[sz4]] : (index, index) -> !fir.shape<2>
  !CHECK: %[[sz16:.*]] = fir.embox %[[sz8]](%[[sz15]]) : (!fir.heap<!fir.array<4x3xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<4x3xf32>>
  !CHECK: %[[sz17:.*]] = fir.convert %[[sz14]] : (i32) -> index
  !CHECK: %[[sz18:.*]] = subi %[[sz17]], %{{.*}} : index
  !CHECK: %[[sz19:.*]]:3 = fir.box_dims %[[sz16]], %[[sz18]] : (!fir.box<!fir.array<4x3xf32>>, index) -> (index, index, index)
  !CHECK: %[[sz20:.*]] = fir.convert %[[sz19]]#1 : (index) -> i64
  !CHECK: %[[sz21:.*]] = fir.convert %[[sz20]] : (i64) -> i32
  !CHECK: fir.store %[[sz21]] to %[[sz2]] : !fir.ref<i32>
  !CHECK: fir.freemem %[[sz8]] : !fir.heap<!fir.array<4x3xf32>>
end subroutine size_test
