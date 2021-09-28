! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QQmain() {
program nested_where
  ! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<index>
  ! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.heap<i8>
  ! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.heap<index>
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.heap<i8>
  ! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QEa) : !fir.ref<!fir.array<3xi32>>
  ! CHECK:         %[[VAL_8:.*]] = constant 3 : index
  ! CHECK:         %[[VAL_9:.*]] = fir.address_of(@_QEmask1) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:         %[[VAL_10:.*]] = constant 3 : index
  ! CHECK:         %[[VAL_11:.*]] = fir.address_of(@_QEmask2) : !fir.ref<!fir.array<3x!fir.logical<4>>>
  ! CHECK:         %[[VAL_12:.*]] = constant 3 : index
  ! CHECK:         %[[VAL_13:.*]] = fir.zero_bits !fir.heap<i8>
  ! CHECK:         fir.store %[[VAL_13]] to %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_14:.*]] = fir.zero_bits !fir.heap<index>
  ! CHECK:         fir.store %[[VAL_14]] to %[[VAL_5]] : !fir.ref<!fir.heap<index>>
  ! CHECK:         %[[VAL_15:.*]] = fir.zero_bits !fir.heap<i8>
  ! CHECK:         fir.store %[[VAL_15]] to %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_16:.*]] = fir.zero_bits !fir.heap<index>
  ! CHECK:         fir.store %[[VAL_16]] to %[[VAL_3]] : !fir.ref<!fir.heap<index>>
  ! CHECK:         %[[VAL_17:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
  ! CHECK:         %[[VAL_19:.*]] = constant 3 : i32
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> index
  ! CHECK:         %[[VAL_21:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_23:.*]] = fir.array_load %[[VAL_7]](%[[VAL_22]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
  ! CHECK:         %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_23]]) -> (!fir.array<3xi32>) {
  ! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_27]] to %[[VAL_2]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_28:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_29:.*]] = fir.array_load %[[VAL_9]](%[[VAL_28]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
  ! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:           %[[VAL_32:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_33:.*]] = fir.array_load %[[VAL_31]](%[[VAL_32]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
  ! CHECK:           %[[VAL_34:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:           %[[VAL_37:.*]] = constant 0 : i64
  ! CHECK:           %[[VAL_38:.*]] = cmpi eq, %[[VAL_36]], %[[VAL_37]] : i64
  ! CHECK:           fir.if %[[VAL_38]] {
  ! CHECK:             %[[VAL_39:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_10]] {uniq_name = ".lazy.mask"}
  ! CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:             fir.store %[[VAL_39]] to %[[VAL_40]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:             %[[VAL_41:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
  ! CHECK:             %[[VAL_42:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_41]], %[[VAL_42]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
  ! CHECK:             fir.store %[[VAL_10]] to %[[VAL_43]] : !fir.ref<index>
  ! CHECK:             %[[VAL_44:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
  ! CHECK:             fir.store %[[VAL_41]] to %[[VAL_44]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
  ! CHECK:           }
  ! CHECK:           %[[VAL_45:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_46:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_47:.*]] = subi %[[VAL_10]], %[[VAL_45]] : index
  ! CHECK:           %[[VAL_48:.*]] = fir.do_loop %[[VAL_49:.*]] = %[[VAL_46]] to %[[VAL_47]] step %[[VAL_45]] unordered iter_args(%[[VAL_50:.*]] = %[[VAL_33]]) -> (!fir.array<?xi8>) {
  ! CHECK:             %[[VAL_51:.*]] = fir.array_fetch %[[VAL_29]], %[[VAL_49]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
  ! CHECK:             %[[VAL_52:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:             %[[VAL_54:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_55:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_56:.*]] = addi %[[VAL_49]], %[[VAL_55]] : index
  ! CHECK:             %[[VAL_57:.*]] = fir.array_coor %[[VAL_53]](%[[VAL_54]]) %[[VAL_56]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:             %[[VAL_58:.*]] = fir.convert %[[VAL_51]] : (!fir.logical<4>) -> i8
  ! CHECK:             fir.store %[[VAL_58]] to %[[VAL_57]] : !fir.ref<i8>
  ! CHECK:             fir.result %[[VAL_50]] : !fir.array<?xi8>
  ! CHECK:           }
  ! CHECK:           %[[VAL_59:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:           %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:           fir.array_merge_store %[[VAL_33]], %[[VAL_61:.*]] to %[[VAL_60]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
  ! CHECK:           fir.result %[[VAL_26]] : !fir.array<3xi32>
  ! CHECK:         }
  ! CHECK:         %[[VAL_62:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:         %[[VAL_64:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.heap<index>>
  ! CHECK:         %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (!fir.heap<index>) -> !fir.heap<!fir.array<?xindex>>
  ! CHECK:         %[[VAL_66:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_67:.*]] = fir.coordinate_of %[[VAL_65]], %[[VAL_66]] : (!fir.heap<!fir.array<?xindex>>, index) -> !fir.ref<index>
  ! CHECK:         %[[VAL_68:.*]] = fir.load %[[VAL_67]] : !fir.ref<index>
  ! CHECK:         %[[VAL_69:.*]] = fir.shape %[[VAL_68]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_70:.*]] = fir.do_loop %[[VAL_71:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] unordered iter_args(%[[VAL_72:.*]] = %[[VAL_23]]) -> (!fir.array<3xi32>) {
  ! CHECK:           %[[VAL_73:.*]] = fir.convert %[[VAL_71]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_73]] to %[[VAL_1]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_74:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_75:.*]] = fir.array_load %[[VAL_11]](%[[VAL_74]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
  ! CHECK:           %[[VAL_76:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:           %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:           %[[VAL_78:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK:           %[[VAL_79:.*]] = fir.array_load %[[VAL_77]](%[[VAL_78]]) : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>) -> !fir.array<?xi8>
  ! CHECK:           %[[VAL_80:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:           %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:           %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (!fir.heap<!fir.array<?xi8>>) -> i64
  ! CHECK:           %[[VAL_83:.*]] = constant 0 : i64
  ! CHECK:           %[[VAL_84:.*]] = cmpi eq, %[[VAL_82]], %[[VAL_83]] : i64
  ! CHECK:           fir.if %[[VAL_84]] {
  ! CHECK:             %[[VAL_85:.*]] = fir.allocmem !fir.array<?xi8>, %[[VAL_12]] {uniq_name = ".lazy.mask"}
  ! CHECK:             %[[VAL_86:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.heap<i8>>) -> !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:             fir.store %[[VAL_85]] to %[[VAL_86]] : !fir.ref<!fir.heap<!fir.array<?xi8>>>
  ! CHECK:             %[[VAL_87:.*]] = fir.allocmem !fir.array<1xindex> {uniq_name = ".lazy.mask.shape"}
  ! CHECK:             %[[VAL_88:.*]] = constant 0 : index
  ! CHECK:             %[[VAL_89:.*]] = fir.coordinate_of %[[VAL_87]], %[[VAL_88]] : (!fir.heap<!fir.array<1xindex>>, index) -> !fir.ref<index>
  ! CHECK:             fir.store %[[VAL_12]] to %[[VAL_89]] : !fir.ref<index>
  ! CHECK:             %[[VAL_90:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.heap<index>>) -> !fir.ref<!fir.heap<!fir.array<1xindex>>>
  ! CHECK:             fir.store %[[VAL_87]] to %[[VAL_90]] : !fir.ref<!fir.heap<!fir.array<1xindex>>>
  ! CHECK:           }
  ! CHECK:           %[[VAL_91:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_92:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_93:.*]] = subi %[[VAL_12]], %[[VAL_91]] : index
  ! CHECK:           %[[VAL_94:.*]] = fir.do_loop %[[VAL_95:.*]] = %[[VAL_92]] to %[[VAL_93]] step %[[VAL_91]] unordered iter_args(%[[VAL_96:.*]] = %[[VAL_79]]) -> (!fir.array<?xi8>) {
  ! CHECK:             %[[VAL_97:.*]] = fir.array_fetch %[[VAL_75]], %[[VAL_95]] : (!fir.array<3x!fir.logical<4>>, index) -> !fir.logical<4>
  ! CHECK:             %[[VAL_98:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:             %[[VAL_99:.*]] = fir.convert %[[VAL_98]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:             %[[VAL_100:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK:             %[[VAL_101:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_102:.*]] = addi %[[VAL_95]], %[[VAL_101]] : index
  ! CHECK:             %[[VAL_103:.*]] = fir.array_coor %[[VAL_99]](%[[VAL_100]]) %[[VAL_102]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:             %[[VAL_104:.*]] = fir.convert %[[VAL_97]] : (!fir.logical<4>) -> i8
  ! CHECK:             fir.store %[[VAL_104]] to %[[VAL_103]] : !fir.ref<i8>
  ! CHECK:             fir.result %[[VAL_96]] : !fir.array<?xi8>
  ! CHECK:           }
  ! CHECK:           %[[VAL_105:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:           %[[VAL_106:.*]] = fir.convert %[[VAL_105]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:           fir.array_merge_store %[[VAL_79]], %[[VAL_107:.*]] to %[[VAL_106]] : !fir.array<?xi8>, !fir.array<?xi8>, !fir.heap<!fir.array<?xi8>>
  ! CHECK:           fir.result %[[VAL_72]] : !fir.array<3xi32>
  ! CHECK:         }
  ! CHECK:         %[[VAL_108:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_109:.*]] = fir.convert %[[VAL_108]] : (!fir.heap<i8>) -> !fir.heap<!fir.array<?xi8>>
  ! CHECK:         %[[VAL_110:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<index>>
  ! CHECK:         %[[VAL_111:.*]] = fir.convert %[[VAL_110]] : (!fir.heap<index>) -> !fir.heap<!fir.array<?xindex>>
  ! CHECK:         %[[VAL_112:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_113:.*]] = fir.coordinate_of %[[VAL_111]], %[[VAL_112]] : (!fir.heap<!fir.array<?xindex>>, index) -> !fir.ref<index>
  ! CHECK:         %[[VAL_114:.*]] = fir.load %[[VAL_113]] : !fir.ref<index>
  ! CHECK:         %[[VAL_115:.*]] = fir.shape %[[VAL_114]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_116:.*]] = constant 1 : i32
  ! CHECK:         %[[VAL_117:.*]] = fir.do_loop %[[VAL_118:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_21]] unordered iter_args(%[[VAL_119:.*]] = %[[VAL_23]]) -> (!fir.array<3xi32>) {
  ! CHECK:           %[[VAL_120:.*]] = fir.convert %[[VAL_118]] : (index) -> i32
  ! CHECK:           fir.store %[[VAL_120]] to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:           %[[VAL_121:.*]] = constant 1 : index
  ! CHECK:           %[[VAL_122:.*]] = constant 0 : index
  ! CHECK:           %[[VAL_123:.*]] = subi %[[VAL_114]], %[[VAL_121]] : index
  ! CHECK:           %[[VAL_124:.*]] = fir.do_loop %[[VAL_125:.*]] = %[[VAL_122]] to %[[VAL_123]] step %[[VAL_121]] unordered iter_args(%[[VAL_126:.*]] = %[[VAL_119]]) -> (!fir.array<3xi32>) {
  ! CHECK:             %[[VAL_127:.*]] = constant 1 : index
  ! CHECK:             %[[VAL_128:.*]] = addi %[[VAL_125]], %[[VAL_127]] : index
  ! CHECK:             %[[VAL_129:.*]] = fir.array_coor %[[VAL_63]](%[[VAL_69]]) %[[VAL_128]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:             %[[VAL_130:.*]] = fir.load %[[VAL_129]] : !fir.ref<i8>
  ! CHECK:             %[[VAL_131:.*]] = fir.convert %[[VAL_130]] : (i8) -> i1
  ! CHECK:             %[[VAL_132:.*]] = fir.if %[[VAL_131]] -> (!fir.array<3xi32>) {
  ! CHECK:               %[[VAL_133:.*]] = constant 1 : index
  ! CHECK:               %[[VAL_134:.*]] = addi %[[VAL_125]], %[[VAL_133]] : index
  ! CHECK:               %[[VAL_135:.*]] = fir.array_coor %[[VAL_109]](%[[VAL_115]]) %[[VAL_134]] : (!fir.heap<!fir.array<?xi8>>, !fir.shape<1>, index) -> !fir.ref<i8>
  ! CHECK:               %[[VAL_136:.*]] = fir.load %[[VAL_135]] : !fir.ref<i8>
  ! CHECK:               %[[VAL_137:.*]] = fir.convert %[[VAL_136]] : (i8) -> i1
  ! CHECK:               %[[VAL_138:.*]] = fir.if %[[VAL_137]] -> (!fir.array<3xi32>) {
  ! CHECK:                 %[[VAL_139:.*]] = constant 1 : index
  ! CHECK:                 %[[VAL_140:.*]] = addi %[[VAL_125]], %[[VAL_139]] : index
  ! CHECK:                 %[[VAL_141:.*]] = fir.array_update %[[VAL_126]], %[[VAL_116]], %[[VAL_140]] {Fortran.offsets} : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
  ! CHECK:                 fir.result %[[VAL_141]] : !fir.array<3xi32>
  ! CHECK:               } else {
  ! CHECK:                 fir.result %[[VAL_126]] : !fir.array<3xi32>
  ! CHECK:               }
  ! CHECK:               fir.result %[[VAL_142:.*]] : !fir.array<3xi32>
  ! CHECK:             } else {
  ! CHECK:               fir.result %[[VAL_126]] : !fir.array<3xi32>
  ! CHECK:             }
  ! CHECK:             fir.result %[[VAL_143:.*]] : !fir.array<3xi32>
  ! CHECK:           }
  ! CHECK:           fir.result %[[VAL_144:.*]] : !fir.array<3xi32>
  ! CHECK:         }
  ! CHECK:         fir.array_merge_store %[[VAL_23]], %[[VAL_145:.*]] to %[[VAL_7]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
  ! CHECK:         %[[VAL_146:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_147:.*]] = fir.convert %[[VAL_146]] : (!fir.heap<i8>) -> i64
  ! CHECK:         %[[VAL_148:.*]] = constant 0 : i64
  ! CHECK:         %[[VAL_149:.*]] = cmpi ne, %[[VAL_147]], %[[VAL_148]] : i64
  ! CHECK:         fir.if %[[VAL_149]] {
  ! CHECK:           fir.freemem %[[VAL_146]] : !fir.heap<i8>
  ! CHECK:         }
  ! CHECK:         %[[VAL_150:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.heap<i8>>
  ! CHECK:         %[[VAL_151:.*]] = fir.convert %[[VAL_150]] : (!fir.heap<i8>) -> i64
  ! CHECK:         %[[VAL_152:.*]] = constant 0 : i64
  ! CHECK:         %[[VAL_153:.*]] = cmpi ne, %[[VAL_151]], %[[VAL_152]] : i64
  ! CHECK:         fir.if %[[VAL_153]] {
  ! CHECK:           fir.freemem %[[VAL_150]] : !fir.heap<i8>
  ! CHECK:         }

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
  ! CHECK:         return
  ! CHECK:       }
end program nested_where

