! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine trans_test(store, word)
  ! CHECK-LABEL: func @_QPtrans_test(
  ! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i32>,
  ! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<f32>) {
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
  ! CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK:         %[[VAL_5:.*]] = fir.zero_bits !fir.heap<i32>
  ! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK:         fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_8:.*]] = arith.constant {{.*}} : i32
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_3]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_4]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranATransfer(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK:         %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_15]] : !fir.heap<i32>
  ! CHECK:         fir.store %[[VAL_16]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:         fir.freemem %[[VAL_15]] : !fir.heap<i32>
  ! CHECK:         return
  ! CHECK:       }
  integer :: store
  real :: word
  store = transfer(word, store)
end subroutine

subroutine trans_test2(store, word)
  ! CHECK-LABEL: func @_QPtrans_test2(
  ! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.array<3xi32>>,
  ! CHECK-SAME:                       %[[VAL_1:.*]]: !fir.ref<f32>) {
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK:         %[[VAL_3:.*]] = arith.constant 3 : index
  ! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_5:.*]] = fir.array_load %[[VAL_0]](%[[VAL_4]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
  ! CHECK:         %[[VAL_6:.*]] = arith.constant 3 : i64
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 3 : i32
  ! CHECK:         %[[VAL_9:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_0]](%[[VAL_10]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK:         %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_15:.*]] = fir.embox %[[VAL_12]](%[[VAL_14]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK:         fir.store %[[VAL_15]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:         %[[VAL_16:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_17:.*]] = arith.constant {{.*}} : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_9]] : (!fir.box<f32>) -> !fir.box<none>
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_8]] : (i32) -> i64
  ! CHECK:         %[[VAL_23:.*]] = fir.call @_FortranATransferSize(%[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_17]], %[[VAL_22]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32, i64) -> none
  ! CHECK:         %[[VAL_24:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK:         %[[VAL_25:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_24]], %[[VAL_25]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  ! CHECK:         %[[VAL_27:.*]] = fir.box_addr %[[VAL_24]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  ! CHECK:         %[[VAL_28:.*]] = fir.shape_shift %[[VAL_26]]#0, %[[VAL_26]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK:         %[[VAL_29:.*]] = fir.array_load %[[VAL_27]](%[[VAL_28]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.array<?xi32>
  ! CHECK:         %[[VAL_30:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_32:.*]] = arith.subi %[[VAL_7]], %[[VAL_30]] : index
  ! CHECK:         %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_5]]) -> (!fir.array<3xi32>) {
  ! CHECK:           %[[VAL_36:.*]] = fir.array_fetch %[[VAL_29]], %[[VAL_34]] : (!fir.array<?xi32>, index) -> i32
  ! CHECK:           %[[VAL_37:.*]] = fir.array_update %[[VAL_35]], %[[VAL_36]], %[[VAL_34]] : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
  ! CHECK:           fir.result %[[VAL_37]] : !fir.array<3xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_5]], %[[VAL_38:.*]] to %[[VAL_0]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
  ! CHECK:         fir.freemem %[[VAL_27]] : !fir.heap<!fir.array<?xi32>>
  ! CHECK:         return
  ! CHECK:       }
  integer :: store(3)
  real :: word
  store = transfer(word, store, 3)
end subroutine

integer function trans_test3(p)
  ! CHECK-LABEL: func @_QPtrans_test3(
  ! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<i32>) -> i32 {
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QFtrans_test3Tobj{x:i32}> {bindc_name = "t", uniq_name = "_QFtrans_test3Et"}
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "trans_test3", uniq_name = "_QFtrans_test3Etrans_test3"}
  ! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.type<_QFtrans_test3Tobj{x:i32}>>) -> !fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  ! CHECK:         %[[VAL_7:.*]] = fir.zero_bits !fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  ! CHECK:         %[[VAL_8:.*]] = fir.embox %[[VAL_7]] : (!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>) -> !fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>
  ! CHECK:         fir.store %[[VAL_8]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>>
  ! CHECK:         %[[VAL_9:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_10:.*]] = arith.constant {{.*}} : i32
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_5]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_15:.*]] = fir.call @_FortranATransfer(%[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_10]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>>
  ! CHECK:         %[[VAL_17:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>) -> !fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  ! CHECK:         %[[VAL_18:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.type<_QFtrans_test3Tobj{x:i32}>>) -> !fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  ! CHECK:         fir.store %[[VAL_18]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>>
  ! CHECK:         %[[VAL_19:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
  ! CHECK:         %[[VAL_20:.*]] = arith.constant {{.*}} : i32
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.type<_QFtrans_test3Tobj{x:i32}>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_24:.*]] = fir.call @_FortranAAssign(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_20]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         fir.freemem %[[VAL_17]] : !fir.heap<!fir.type<_QFtrans_test3Tobj{x:i32}>>
  ! CHECK:         %[[VAL_25:.*]] = fir.field_index x, !fir.type<_QFtrans_test3Tobj{x:i32}>
  ! CHECK:         %[[VAL_26:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_25]] : (!fir.ref<!fir.type<_QFtrans_test3Tobj{x:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:         %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<i32>
  ! CHECK:         fir.store %[[VAL_27]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_28:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:         return %[[VAL_28]] : i32
  ! CHECK:       }
  type obj
    integer :: x
  end type
  type (obj) :: t
  integer :: p
  t = transfer(p, t)
  trans_test3 = t%x
end function
