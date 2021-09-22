! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QQmain() {
program nested_where
  integer :: a(3) = 0
  logical :: mask1(3) = (/ .true.,.false.,.true. /)
  logical :: mask2(3) = (/ .true.,.true.,.false. /)
  forall (i=1:3)
    where (mask1)
      where (mask2)
        a = 1
      end where
    endwhere
  end forall
end program nested_where

! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.heap<index>
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.heap<i8>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<index>
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.heap<i8>
! CHECK:         %[[VAL_5:.*]] = fir.address_of(@_QEa) : !fir.ref<!fir.array<3xi32>>
! CHECK:         %[[VAL_6:.*]] = constant 3 : index
! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QEmask1) : !fir.ref<!fir.array<3x!fir.logical<4>>>
! CHECK:         %[[VAL_8:.*]] = constant 3 : index
! CHECK:         %[[VAL_9:.*]] = fir.address_of(@_QEmask2) : !fir.ref<!fir.array<3x!fir.logical<4>>>
! CHECK:         %[[VAL_10:.*]] = constant 3 : index
! CHECK:         %[[VAL_11:.*]] = fir.zero_bits !fir.heap<i8>
! CHECK:         fir.store %[[VAL_11]] to %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_12:.*]] = fir.zero_bits !fir.heap<index>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_3]] : !fir.ref<!fir.heap<index>>
! CHECK:         %[[VAL_13:.*]] = fir.zero_bits !fir.heap<i8>
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_14:.*]] = fir.zero_bits !fir.heap<index>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_1]] : !fir.ref<!fir.heap<index>>
! CHECK:         %[[VAL_15:.*]] = constant 1 : i32
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
! CHECK:         %[[VAL_17:.*]] = constant 3 : i32
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
! CHECK:         %[[VAL_19:.*]] = constant 1 : index
! CHECK:         %[[VAL_20:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_5]](%[[VAL_20]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK:         %[[VAL_22:.*]] = fir.do_loop %[[VAL_23:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_19]] unordered iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (!fir.array<3xi32>) {
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (index) -> i32
! CHECK:           fir.store %[[VAL_25]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_26:.*]] = constant 3 : i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = constant 1 : i32
! CHECK:           %[[VAL_29:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_30:.*]] = fir.array_load %[[VAL_7]](%[[VAL_29]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
! CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_33:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_34:.*]] = fir.array_load %[[VAL_32]](%[[VAL_33]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (!fir.heap<!fir.array<?xi8>>) -> i64
! CHECK:           %[[VAL_38:.*]] = constant 0 : i64
! CHECK:           %[[VAL_39:.*]] = cmpi eq, %[[VAL_37]], %[[VAL_38]] : i64
! CHECK:           fir.if %[[VAL_39]] {
! CHECK:             %[[VAL_40:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_8]] {uniq_name = ".lazy.mask"}
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             fir.store %[[VAL_40]] to %[[VAL_41]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_42:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
! CHECK:             %[[VAL_43:.*]] = constant 0 : index
! CHECK:             %[[VAL_44:.*]] = fir.coordinate_of %[[VAL_42]], %[[VAL_43]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
! CHECK:             fir.store %[[VAL_8]] to %[[VAL_44]] : !fir.ref<index>
! CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:             fir.store %[[VAL_42]] to %[[VAL_45]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:           }
! CHECK:           %[[VAL_46:.*]] = constant 1 : index
! CHECK:           %[[VAL_47:.*]] = constant 0 : index
! CHECK:           %[[VAL_48:.*]] = subi %[[VAL_8]], %[[VAL_46]] : index
! CHECK:           %[[VAL_49:.*]] = fir.do_loop %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_46]] unordered iter_args(%[[VAL_51:.*]] = %[[VAL_34]]) -> (!fir.array<?xi8>) {
! CHECK:             %[[VAL_52:.*]] = fir.array_fetch %[[VAL_30]], %[[VAL_50]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:             %[[VAL_53:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:             %[[VAL_55:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_56:.*]] = constant 1 : index
! CHECK:             %[[VAL_57:.*]] = addi %[[VAL_50]], %[[VAL_56]] : index
! CHECK:             %[[VAL_58:.*]] = fir.array_coor %[[VAL_54]](%[[VAL_55]]) %[[VAL_57]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_59:.*]] = fir.convert %[[VAL_52]] : (!fir.logical<4>) -> i8
! CHECK:             fir.store %[[VAL_59]] to %[[VAL_58]] : !fir.ref<i8>
! CHECK:             fir.result %[[VAL_51]] : !fir.array<?xi8>
! CHECK:           }
! CHECK:           %[[VAL_60:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           fir.array_merge_store %[[VAL_34]], %[[VAL_62:.*]] to %[[VAL_61]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_63:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_64:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_65:.*]] = fir.array_load %[[VAL_9]](%[[VAL_64]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
! CHECK:           %[[VAL_66:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_67:.*]] = fir.convert %[[VAL_66]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_68:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_69:.*]] = fir.array_load %[[VAL_67]](%[[VAL_68]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
! CHECK:           %[[VAL_70:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_71:.*]] = fir.convert %[[VAL_70]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (!fir.heap<!fir.array<?xi8>>) -> i64
! CHECK:           %[[VAL_73:.*]] = constant 0 : i64
! CHECK:           %[[VAL_74:.*]] = cmpi eq, %[[VAL_72]], %[[VAL_73]] : i64
! CHECK:           fir.if %[[VAL_74]] {
! CHECK:             %[[VAL_75:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_10]] {uniq_name = ".lazy.mask"}
! CHECK:             %[[VAL_76:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             fir.store %[[VAL_75]] to %[[VAL_76]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
! CHECK:             %[[VAL_77:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
! CHECK:             %[[VAL_78:.*]] = constant 0 : index
! CHECK:             %[[VAL_79:.*]] = fir.coordinate_of %[[VAL_77]], %[[VAL_78]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_79]] : !fir.ref<index>
! CHECK:             %[[VAL_80:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:             fir.store %[[VAL_77]] to %[[VAL_80]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
! CHECK:           }
! CHECK:           %[[VAL_81:.*]] = constant 1 : index
! CHECK:           %[[VAL_82:.*]] = constant 0 : index
! CHECK:           %[[VAL_83:.*]] = subi %[[VAL_10]], %[[VAL_81]] : index
! CHECK:           %[[VAL_84:.*]] = fir.do_loop %[[VAL_85:.*]] = %[[VAL_82]] to %[[VAL_83]] step %[[VAL_81]] unordered iter_args(%[[VAL_86:.*]] = %[[VAL_69]]) -> (!fir.array<?xi8>) {
! CHECK:             %[[VAL_87:.*]] = fir.array_fetch %[[VAL_65]], %[[VAL_85]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
! CHECK:             %[[VAL_88:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:             %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:             %[[VAL_90:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_91:.*]] = constant 1 : index
! CHECK:             %[[VAL_92:.*]] = addi %[[VAL_85]], %[[VAL_91]] : index
! CHECK:             %[[VAL_93:.*]] = fir.array_coor %[[VAL_89]](%[[VAL_90]]) %[[VAL_92]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_94:.*]] = fir.convert %[[VAL_87]] : (!fir.logical<4>) -> i8
! CHECK:             fir.store %[[VAL_94]] to %[[VAL_93]] : !fir.ref<i8>
! CHECK:             fir.result %[[VAL_86]] : !fir.array<?xi8>
! CHECK:           }
! CHECK:           %[[VAL_95:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:           %[[VAL_96:.*]] = fir.convert %[[VAL_95]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
! CHECK:           fir.array_merge_store %[[VAL_69]], %[[VAL_97:.*]] to %[[VAL_96]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
! CHECK:           %[[VAL_98:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_99:.*]] = constant 1 : index
! CHECK:           %[[VAL_100:.*]] = constant 0 : index
! CHECK:           %[[VAL_101:.*]] = subi %[[VAL_27]], %[[VAL_99]] : index
! CHECK:           %[[VAL_102:.*]] = fir.do_loop %[[VAL_103:.*]] = %[[VAL_100]] to %[[VAL_101]] step %[[VAL_99]] unordered iter_args(%[[VAL_104:.*]] = %[[VAL_21]]) -> (!fir.array<3xi32>) {
! CHECK:             %[[VAL_105:.*]] = constant 1 : index
! CHECK:             %[[VAL_106:.*]] = addi %[[VAL_103]], %[[VAL_105]] : index
! CHECK:             %[[VAL_107:.*]] = fir.array_coor %[[VAL_61]](%[[VAL_63]]) %[[VAL_106]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_108:.*]] = fir.load %[[VAL_107]] : !fir.ref<i8>
! CHECK:             %[[VAL_109:.*]] = fir.convert %[[VAL_108]] : (i8) -> i1
! CHECK:             %[[VAL_110:.*]] = fir.if %[[VAL_109]] -> (!fir.array<3xi32>) {
! CHECK:               %[[VAL_111:.*]] = constant 1 : index
! CHECK:               %[[VAL_112:.*]] = addi %[[VAL_103]], %[[VAL_111]] : index
! CHECK:               %[[VAL_113:.*]] = fir.array_coor %[[VAL_96]](%[[VAL_98]]) %[[VAL_112]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
! CHECK:               %[[VAL_114:.*]] = fir.load %[[VAL_113]] : !fir.ref<i8>
! CHECK:               %[[VAL_115:.*]] = fir.convert %[[VAL_114]] : (i8) -> i1
! CHECK:               %[[VAL_116:.*]] = fir.if %[[VAL_115]] -> (!fir.array<3xi32>) {
! CHECK:                 %[[VAL_117:.*]] = constant 1 : index
! CHECK:                 %[[VAL_118:.*]] = addi %[[VAL_103]], %[[VAL_117]] : index
! CHECK:                 %[[VAL_119:.*]] = fir.array_update %[[VAL_24]], %[[VAL_28]], %[[VAL_118]] {Fortran.offsets} : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK:                 fir.result %[[VAL_119]] : !fir.array<3xi32>
! CHECK:               } else {
! CHECK:                 fir.result %[[VAL_104]] : !fir.array<3xi32>
! CHECK:               }
! CHECK:               fir.result %[[VAL_120:.*]] : !fir.array<3xi32>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_104]] : !fir.array<3xi32>
! CHECK:             }
! CHECK:             fir.result %[[VAL_121:.*]] : !fir.array<3xi32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_122:.*]] : !fir.array<3xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_123:.*]] to %[[VAL_5]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
! CHECK:         %[[VAL_124:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_125:.*]] = fir.convert %[[VAL_124]] : (!fir.heap<i8>) -> i64
! CHECK:         %[[VAL_126:.*]] = constant 0 : i64
! CHECK:         %[[VAL_127:.*]] = cmpi ne, %[[VAL_125]], %[[VAL_126]] : i64
! CHECK:         fir.if %[[VAL_127]] {
! CHECK:           fir.freemem %[[VAL_124]] : !fir.heap<i8>
! CHECK:         }
! CHECK:         %[[VAL_128:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
! CHECK:         %[[VAL_129:.*]] = fir.convert %[[VAL_128]] : (!fir.heap<i8>) -> i64
! CHECK:         %[[VAL_130:.*]] = constant 0 : i64
! CHECK:         %[[VAL_131:.*]] = cmpi ne, %[[VAL_129]], %[[VAL_130]] : i64
! CHECK:         fir.if %[[VAL_131]] {
! CHECK:           fir.freemem %[[VAL_128]] : !fir.heap<i8>
! CHECK:         }
! CHECK:         return
! CHECK:       }
