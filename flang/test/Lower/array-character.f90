! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPissue(
! CHECK-SAME:                 %[[VAL_0:.*]]: !fir.boxchar<1>,
! CHECK-SAME:                 %[[VAL_1:.*]]: !fir.boxchar<1>) {
subroutine issue(c1, c2)
  ! CHECK:         %[[VAL_2:.*]] = constant false
  ! CHECK:         %[[VAL_3:.*]] = constant 32 : i8
  ! CHECK:         %[[VAL_4:.*]] = constant 3 : index
  ! CHECK:         %[[VAL_5:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_6:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK:         %[[VAL_9:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
  ! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK:         br ^bb1(%[[VAL_5]], %[[VAL_4]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
  ! CHECK:         %[[VAL_14:.*]] = cmpi sgt, %[[VAL_13]], %[[VAL_5]] : index
  ! CHECK:         cond_br %[[VAL_14]], ^bb2, ^bb6
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_15:.*]] = addi %[[VAL_12]], %[[VAL_6]] : index
  ! CHECK:         %[[VAL_16:.*]] = fir.array_coor %[[VAL_10]](%[[VAL_11]]) %[[VAL_15]] typeparams %[[VAL_9]]#1 : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_17:.*]] = fir.array_coor %[[VAL_8]](%[[VAL_11]]) %[[VAL_15]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,4>>
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_9]]#1 : (index) -> i64
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_19]], %[[VAL_20]], %[[VAL_18]], %[[VAL_2]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_21:.*]] = subi %[[VAL_9]]#1, %[[VAL_6]] : index
  ! CHECK:         %[[VAL_22:.*]] = fir.undefined !fir.char<1>
  ! CHECK:         %[[VAL_23:.*]] = fir.insert_value %[[VAL_22]], %[[VAL_3]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:         %[[VAL_24:.*]] = subi %[[VAL_21]], %[[VAL_9]]#1 : index
  ! CHECK:         %[[VAL_25:.*]] = addi %[[VAL_24]], %[[VAL_6]] : index
  ! CHECK:         br ^bb3(%[[VAL_9]]#1, %[[VAL_25]] : index, index)
  ! CHECK:       ^bb3(%[[VAL_26:.*]]: index, %[[VAL_27:.*]]: index):
  ! CHECK:         %[[VAL_28:.*]] = cmpi sgt, %[[VAL_27]], %[[VAL_5]] : index
  ! CHECK:         cond_br %[[VAL_28]], ^bb4, ^bb5
  ! CHECK:       ^bb4:
  ! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<!fir.array<4x!fir.char<1>>>
  ! CHECK:         %[[VAL_30:.*]] = fir.coordinate_of %[[VAL_29]], %[[VAL_26]] : (!fir.ref<!fir.array<4x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         fir.store %[[VAL_23]] to %[[VAL_30]] : !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_31:.*]] = addi %[[VAL_26]], %[[VAL_6]] : index
  ! CHECK:         %[[VAL_32:.*]] = subi %[[VAL_27]], %[[VAL_6]] : index
  ! CHECK:         br ^bb3(%[[VAL_31]], %[[VAL_32]] : index, index)
  ! CHECK:       ^bb5:
  ! CHECK:         %[[VAL_33:.*]] = subi %[[VAL_13]], %[[VAL_6]] : index
  ! CHECK:         br ^bb1(%[[VAL_15]], %[[VAL_33]] : index, index)
  character(4) :: c1(3)
  character(*) :: c2(3)
  c1 = c2
  ! CHECK:         return
  ! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QQmain() {
program p
  ! CHECK:         %[[VAL_0:.*]] = constant 4 : index
  ! CHECK:         %[[VAL_1:.*]] = constant 3 : index
  ! CHECK:         %[[VAL_3:.*]] = constant -1 : i32
  ! CHECK:         %[[VAL_5:.*]] = fir.address_of(@_QEc1) : !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK:         %[[VAL_6:.*]] = fir.address_of(@_QEc2) : !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_9:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_3]], %[[VAL_8]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_6]](%[[VAL_10]]) : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1,4>>>
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<3x!fir.char<1,4>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_9]], %[[VAL_12]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK:         %[[VAL_14:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_9]]) : (!fir.ref<i8>) -> i32
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_16:.*]] = fir.emboxchar %[[VAL_15]], %[[VAL_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_18:.*]] = fir.emboxchar %[[VAL_17]], %[[VAL_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:         fir.call @_QPissue(%[[VAL_16]], %[[VAL_18]]) : (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
  ! CHECK:         %[[VAL_19:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_3]], %[[VAL_8]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_20:.*]] = fir.embox %[[VAL_5]](%[[VAL_10]]) : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1,4>>>
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<3x!fir.char<1,4>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_22:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_19]], %[[VAL_21]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK:         %[[VAL_23:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_19]]) : (!fir.ref<i8>) -> i32
  ! CHECK:         fir.call @_QPcharlit() : () -> ()
  character(4) :: c1(3)
  character(4) :: c2(3) = ["abcd", "    ", "    "]
  print *, c2
  call issue(c1, c2)
  print *, c1
  call charlit
  ! CHECK:         return
  ! CHECK:       }
end program p

! CHECK-LABEL: func @_QPcharlit() {
subroutine charlit
  ! CHECK:         %[[VAL_2:.*]] = constant -1 : i32
  ! CHECK:         %[[VAL_4:.*]] = constant 3 : i64
  ! CHECK:         %[[VAL_5:.*]] = constant false
  ! CHECK:         %[[VAL_6:.*]] = constant 4 : index
  ! CHECK:         %[[VAL_7:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_8:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_11:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_10]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_12:.*]] = fir.address_of(@_QQro.4x3xc1.1636b396a657de68ffb870a885ac44b4) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_14:.*]] = fir.allocmem !fir.array<4x!fir.char<1,3>>
  ! CHECK:         br ^bb1(%[[VAL_7]], %[[VAL_6]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
  ! CHECK:         %[[VAL_17:.*]] = cmpi sgt, %[[VAL_16]], %[[VAL_7]] : index
  ! CHECK:         cond_br %[[VAL_17]], ^bb2, ^bb3
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_18:.*]] = addi %[[VAL_15]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_19:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_18]] : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:         %[[VAL_20:.*]] = fir.array_coor %[[VAL_14]](%[[VAL_13]]) %[[VAL_18]] : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_21]], %[[VAL_22]], %[[VAL_4]], %[[VAL_5]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_23:.*]] = subi %[[VAL_16]], %[[VAL_8]] : index
  ! CHECK:         br ^bb1(%[[VAL_18]], %[[VAL_23]] : index, index)
  ! CHECK:       ^bb3:
  ! CHECK:         %[[VAL_24:.*]] = fir.embox %[[VAL_14]](%[[VAL_13]]) : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_26:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_11]], %[[VAL_25]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK:         fir.freemem %[[VAL_14]] : !fir.heap<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_27:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_11]]) : (!fir.ref<i8>) -> i32
  ! CHECK:         %[[VAL_28:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_10]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_29:.*]] = fir.allocmem !fir.array<4x!fir.char<1,3>>
  ! CHECK:         br ^bb4(%[[VAL_7]], %[[VAL_6]] : index, index)
  ! CHECK:       ^bb4(%[[VAL_30:.*]]: index, %[[VAL_31:.*]]: index):
  ! CHECK:         %[[VAL_32:.*]] = cmpi sgt, %[[VAL_31]], %[[VAL_7]] : index
  ! CHECK:         cond_br %[[VAL_32]], ^bb5, ^bb6
  ! CHECK:       ^bb5:
  ! CHECK:         %[[VAL_33:.*]] = addi %[[VAL_30]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_34:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_33]] : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:         %[[VAL_35:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_13]]) %[[VAL_33]] : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:         %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_36]], %[[VAL_37]], %[[VAL_4]], %[[VAL_5]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_38:.*]] = subi %[[VAL_31]], %[[VAL_8]] : index
  ! CHECK:         br ^bb4(%[[VAL_33]], %[[VAL_38]] : index, index)
  ! CHECK:       ^bb6:
  ! CHECK:         %[[VAL_39:.*]] = fir.embox %[[VAL_29]](%[[VAL_13]]) : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_41:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_28]], %[[VAL_40]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK:         fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_42:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_28]]) : (!fir.ref<i8>) -> i32
  ! CHECK:         %[[VAL_43:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_10]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_44:.*]] = fir.allocmem !fir.array<4x!fir.char<1,3>>
  ! CHECK:         br ^bb7(%[[VAL_7]], %[[VAL_6]] : index, index)
  ! CHECK:       ^bb7(%[[VAL_45:.*]]: index, %[[VAL_46:.*]]: index):
  ! CHECK:         %[[VAL_47:.*]] = cmpi sgt, %[[VAL_46]], %[[VAL_7]] : index
  ! CHECK:         cond_br %[[VAL_47]], ^bb8, ^bb9
  ! CHECK:       ^bb8:
  ! CHECK:         %[[VAL_48:.*]] = addi %[[VAL_45]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_49:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_48]] : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:         %[[VAL_50:.*]] = fir.array_coor %[[VAL_44]](%[[VAL_13]]) %[[VAL_48]] : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK:         %[[VAL_51:.*]] = fir.convert %[[VAL_50]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_52:.*]] = fir.convert %[[VAL_49]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_51]], %[[VAL_52]], %[[VAL_4]], %[[VAL_5]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_53:.*]] = subi %[[VAL_46]], %[[VAL_8]] : index
  ! CHECK:         br ^bb7(%[[VAL_48]], %[[VAL_53]] : index, index)
  ! CHECK:       ^bb9:
  ! CHECK:         %[[VAL_54:.*]] = fir.embox %[[VAL_44]](%[[VAL_13]]) : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_56:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_43]], %[[VAL_55]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK:         fir.freemem %[[VAL_44]] : !fir.heap<!fir.array<4x!fir.char<1,3>>>
  ! CHECK:         %[[VAL_57:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_43]]) : (!fir.ref<i8>) -> i32
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  ! CHECK:         return
  ! CHECK:       }
end

! CHECK: fir.global internal @_QQro.4x3xc1.1636b396a657de68ffb870a885ac44b4 constant : !fir.array<4x!fir.char<1,3>>
! CHECK: AA
! CHECK: MM
! CHECK: ZZ
! CHECK-NOT: fir.global internal @_QQro.4x3xc1
! CHECK-NOT: AA
! CHECK-NOT: MM
! CHECK-NOT: ZZ
