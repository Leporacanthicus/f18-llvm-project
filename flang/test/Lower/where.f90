  ! RUN: bbc -emit-fir %s -o - | FileCheck %s
  
  ! CHECK-LABEL: func @_QQmain() {
  ! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QEa) : !fir.ref<!fir.array<10xf32>>
  ! CHECK: %[[VAL_1:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_2:.*]] = fir.address_of(@_QEb) : !fir.ref<!fir.array<10xf32>>
  ! CHECK: %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_4:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
  ! CHECK: %[[VAL_6:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_7:.*]] = fir.array_load %[[VAL_0]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_8:.*]] = arith.constant 4.000000e+00 : f32
  ! CHECK: %[[VAL_9:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
  ! CHECK: %[[VAL_10:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_11:.*]] = fir.array_load %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  ! CHECK: %[[VAL_12:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_13:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_14:.*]] = arith.subi %[[VAL_5]], %[[VAL_12]] : index
  ! CHECK: %[[VAL_15:.*]] = fir.do_loop %[[VAL_16:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_12]] unordered iter_args(%[[VAL_17:.*]] = %[[VAL_11]]) -> (!fir.array<10x!fir.logical<4>>) {
  ! CHECK: %[[VAL_18:.*]] = fir.array_fetch %[[VAL_7]], %[[VAL_16]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_19:.*]] = cmpf ogt, %[[VAL_18]], %[[VAL_8]] : f32
  ! CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i1) -> !fir.logical<4>
  ! CHECK: %[[VAL_21:.*]] = fir.array_update %[[VAL_17]], %[[VAL_20]], %[[VAL_16]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
  ! CHECK: fir.result %[[VAL_21]] : !fir.array<10x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_11]], %[[VAL_22:.*]] to %[[VAL_9]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK: %[[VAL_23:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_24:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_25:.*]] = fir.array_load %[[VAL_2]](%[[VAL_24]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_26:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
  ! CHECK: %[[VAL_28:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_29:.*]] = fir.array_load %[[VAL_0]](%[[VAL_28]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_30:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_31:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_32:.*]] = arith.subi %[[VAL_27]], %[[VAL_30]] : index
  ! CHECK: %[[VAL_33:.*]] = fir.do_loop %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_32]] step %[[VAL_30]] unordered iter_args(%[[VAL_35:.*]] = %[[VAL_25]]) -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_36:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_37:.*]] = arith.addi %[[VAL_34]], %[[VAL_36]] : index
  ! CHECK: %[[VAL_38:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_23]]) %[[VAL_37]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_41:.*]] = fir.if %[[VAL_40]] -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_42:.*]] = fir.array_fetch %[[VAL_29]], %[[VAL_34]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_43:.*]] = negf %[[VAL_42]] : f32
  ! CHECK: %[[VAL_44:.*]] = fir.array_update %[[VAL_35]], %[[VAL_43]], %[[VAL_34]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %[[VAL_44]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[VAL_35]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_45:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_25]], %[[VAL_46:.*]] to %[[VAL_2]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK: fir.freemem %[[VAL_9]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK: %[[VAL_47:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (i64) -> index
  ! CHECK: %[[VAL_49:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_50:.*]] = fir.array_load %[[VAL_0]](%[[VAL_49]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_51:.*]] = arith.constant 1.000000e+02 : f32
  ! CHECK: %[[VAL_52:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
  ! CHECK: %[[VAL_53:.*]] = fir.shape %[[VAL_48]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_54:.*]] = fir.array_load %[[VAL_52]](%[[VAL_53]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  ! CHECK: %[[VAL_55:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_56:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_57:.*]] = arith.subi %[[VAL_48]], %[[VAL_55]] : index
  ! CHECK: %[[VAL_58:.*]] = fir.do_loop %[[VAL_59:.*]] = %[[VAL_56]] to %[[VAL_57]] step %[[VAL_55]] unordered iter_args(%[[VAL_60:.*]] = %[[VAL_54]]) -> (!fir.array<10x!fir.logical<4>>) {
  ! CHECK: %[[VAL_61:.*]] = fir.array_fetch %[[VAL_50]], %[[VAL_59]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_62:.*]] = cmpf ogt, %[[VAL_61]], %[[VAL_51]] : f32
  ! CHECK: %[[VAL_63:.*]] = fir.convert %[[VAL_62]] : (i1) -> !fir.logical<4>
  ! CHECK: %[[VAL_64:.*]] = fir.array_update %[[VAL_60]], %[[VAL_63]], %[[VAL_59]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
  ! CHECK: fir.result %[[VAL_64]] : !fir.array<10x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_54]], %[[VAL_65:.*]] to %[[VAL_52]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK: %[[VAL_66:.*]] = fir.shape %[[VAL_48]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_67:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_68:.*]] = fir.array_load %[[VAL_2]](%[[VAL_67]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_69:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_70:.*]] = fir.convert %[[VAL_69]] : (i64) -> index
  ! CHECK: %[[VAL_71:.*]] = arith.constant 2.000000e+00 : f32
  ! CHECK: %[[VAL_72:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_73:.*]] = fir.array_load %[[VAL_0]](%[[VAL_72]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_74:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_75:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_76:.*]] = arith.subi %[[VAL_70]], %[[VAL_74]] : index
  ! CHECK: %[[VAL_77:.*]] = fir.do_loop %[[VAL_78:.*]] = %[[VAL_75]] to %[[VAL_76]] step %[[VAL_74]] unordered iter_args(%[[VAL_79:.*]] = %[[VAL_68]]) -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_80:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_81:.*]] = arith.addi %[[VAL_78]], %[[VAL_80]] : index
  ! CHECK: %[[VAL_82:.*]] = fir.array_coor %[[VAL_52]](%[[VAL_66]]) %[[VAL_81]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_83:.*]] = fir.load %[[VAL_82]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_84:.*]] = fir.convert %[[VAL_83]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_85:.*]] = fir.if %[[VAL_84]] -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_86:.*]] = fir.array_fetch %[[VAL_73]], %[[VAL_78]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_87:.*]] = arith.mulf %[[VAL_71]], %[[VAL_86]] : f32
  ! CHECK: %[[VAL_88:.*]] = fir.array_update %[[VAL_79]], %[[VAL_87]], %[[VAL_78]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %[[VAL_88]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[VAL_79]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_89:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_68]], %[[VAL_90:.*]] to %[[VAL_2]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK: %[[VAL_91:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_92:.*]] = fir.convert %[[VAL_91]] : (i64) -> index
  ! CHECK: %[[VAL_93:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_94:.*]] = fir.array_load %[[VAL_0]](%[[VAL_93]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_95:.*]] = arith.constant 5.000000e+01 : f32
  ! CHECK: %[[VAL_96:.*]] = fir.allocmem !fir.array<10x!fir.logical<4>>
  ! CHECK: %[[VAL_97:.*]] = fir.shape %[[VAL_92]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_98:.*]] = fir.array_load %[[VAL_96]](%[[VAL_97]]) : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<10x!fir.logical<4>>
  ! CHECK: %[[VAL_99:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_100:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_101:.*]] = arith.subi %[[VAL_92]], %[[VAL_99]] : index
  ! CHECK: %[[VAL_102:.*]] = fir.do_loop %[[VAL_103:.*]] = %[[VAL_100]] to %[[VAL_101]] step %[[VAL_99]] unordered iter_args(%[[VAL_104:.*]] = %[[VAL_98]]) -> (!fir.array<10x!fir.logical<4>>) {
  ! CHECK: %[[VAL_105:.*]] = fir.array_fetch %[[VAL_94]], %[[VAL_103]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_106:.*]] = cmpf ogt, %[[VAL_105]], %[[VAL_95]] : f32
  ! CHECK: %[[VAL_107:.*]] = fir.convert %[[VAL_106]] : (i1) -> !fir.logical<4>
  ! CHECK: %[[VAL_108:.*]] = fir.array_update %[[VAL_104]], %[[VAL_107]], %[[VAL_103]] : (!fir.array<10x!fir.logical<4>>, !fir.logical<4>, index) -> !fir.array<10x!fir.logical<4>>
  ! CHECK: fir.result %[[VAL_108]] : !fir.array<10x!fir.logical<4>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_98]], %[[VAL_109:.*]] to %[[VAL_96]] : !fir.array<10x!fir.logical<4>>, !fir.array<10x!fir.logical<4>>, !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK: %[[VAL_110:.*]] = fir.shape %[[VAL_92]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_111:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_112:.*]] = fir.array_load %[[VAL_2]](%[[VAL_111]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_113:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_114:.*]] = fir.convert %[[VAL_113]] : (i64) -> index
  ! CHECK: %[[VAL_115:.*]] = arith.constant 3.000000e+00 : f32
  ! CHECK: %[[VAL_116:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_117:.*]] = fir.array_load %[[VAL_0]](%[[VAL_116]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_118:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_119:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_120:.*]] = arith.subi %[[VAL_114]], %[[VAL_118]] : index
  ! CHECK: %[[VAL_121:.*]] = fir.do_loop %[[VAL_122:.*]] = %[[VAL_119]] to %[[VAL_120]] step %[[VAL_118]] unordered iter_args(%[[VAL_123:.*]] = %[[VAL_112]]) -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_124:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_125:.*]] = arith.addi %[[VAL_122]], %[[VAL_124]] : index
  ! CHECK: %[[VAL_126:.*]] = fir.array_coor %[[VAL_52]](%[[VAL_66]]) %[[VAL_125]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_127:.*]] = fir.load %[[VAL_126]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_128:.*]] = fir.convert %[[VAL_127]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_129:.*]] = fir.if %[[VAL_128]] -> (!fir.array<10xf32>) {
  ! CHECK: fir.result %[[VAL_123]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: %[[VAL_130:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_131:.*]] = arith.addi %[[VAL_122]], %[[VAL_130]] : index
  ! CHECK: %[[VAL_132:.*]] = fir.array_coor %[[VAL_96]](%[[VAL_110]]) %[[VAL_131]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_133:.*]] = fir.load %[[VAL_132]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_134:.*]] = fir.convert %[[VAL_133]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_135:.*]] = fir.if %[[VAL_134]] -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_136:.*]] = fir.array_fetch %[[VAL_117]], %[[VAL_122]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_137:.*]] = arith.addf %[[VAL_115]], %[[VAL_136]] : f32
  ! CHECK: %[[VAL_138:.*]] = fir.array_update %[[VAL_123]], %[[VAL_137]], %[[VAL_122]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %[[VAL_138]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[VAL_123]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_139:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_140:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_112]], %[[VAL_141:.*]] to %[[VAL_2]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK: %[[VAL_142:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_143:.*]] = fir.array_load %[[VAL_0]](%[[VAL_142]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_144:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_145:.*]] = fir.convert %[[VAL_144]] : (i64) -> index
  ! CHECK: %[[VAL_146:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_147:.*]] = fir.array_load %[[VAL_0]](%[[VAL_146]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_148:.*]] = arith.constant 1.000000e+00 : f32
  ! CHECK: %[[VAL_149:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_150:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_151:.*]] = arith.subi %[[VAL_145]], %[[VAL_149]] : index
  ! CHECK: %[[VAL_152:.*]] = fir.do_loop %[[VAL_153:.*]] = %[[VAL_150]] to %[[VAL_151]] step %[[VAL_149]] unordered iter_args(%[[VAL_154:.*]] = %[[VAL_143]]) -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_155:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_156:.*]] = arith.addi %[[VAL_153]], %[[VAL_155]] : index
  ! CHECK: %[[VAL_157:.*]] = fir.array_coor %[[VAL_52]](%[[VAL_66]]) %[[VAL_156]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_158:.*]] = fir.load %[[VAL_157]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_159:.*]] = fir.convert %[[VAL_158]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_160:.*]] = fir.if %[[VAL_159]] -> (!fir.array<10xf32>) {
  ! CHECK: fir.result %[[VAL_154]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: %[[VAL_161:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_162:.*]] = arith.addi %[[VAL_153]], %[[VAL_161]] : index
  ! CHECK: %[[VAL_163:.*]] = fir.array_coor %[[VAL_96]](%[[VAL_110]]) %[[VAL_162]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_164:.*]] = fir.load %[[VAL_163]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_165:.*]] = fir.convert %[[VAL_164]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_166:.*]] = fir.if %[[VAL_165]] -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_167:.*]] = fir.array_fetch %[[VAL_147]], %[[VAL_153]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_168:.*]] = subf %[[VAL_167]], %[[VAL_148]] : f32
  ! CHECK: %[[VAL_169:.*]] = fir.array_update %[[VAL_154]], %[[VAL_168]], %[[VAL_153]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %[[VAL_169]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: fir.result %[[VAL_154]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_170:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_171:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_143]], %[[VAL_172:.*]] to %[[VAL_0]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK: %[[VAL_173:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_174:.*]] = fir.array_load %[[VAL_0]](%[[VAL_173]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_175:.*]] = arith.constant 10 : i64
  ! CHECK: %[[VAL_176:.*]] = fir.convert %[[VAL_175]] : (i64) -> index
  ! CHECK: %[[VAL_177:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_178:.*]] = fir.array_load %[[VAL_0]](%[[VAL_177]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[VAL_179:.*]] = arith.constant 2.000000e+00 : f32
  ! CHECK: %[[VAL_180:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_181:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_182:.*]] = arith.subi %[[VAL_176]], %[[VAL_180]] : index
  ! CHECK: %[[VAL_183:.*]] = fir.do_loop %[[VAL_184:.*]] = %[[VAL_181]] to %[[VAL_182]] step %[[VAL_180]] unordered iter_args(%[[VAL_185:.*]] = %[[VAL_174]]) -> (!fir.array<10xf32>) {
  ! CHECK: %[[VAL_186:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_187:.*]] = arith.addi %[[VAL_184]], %[[VAL_186]] : index
  ! CHECK: %[[VAL_188:.*]] = fir.array_coor %[[VAL_52]](%[[VAL_66]]) %[[VAL_187]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_189:.*]] = fir.load %[[VAL_188]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_190:.*]] = fir.convert %[[VAL_189]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_191:.*]] = fir.if %[[VAL_190]] -> (!fir.array<10xf32>) {
  ! CHECK: fir.result %[[VAL_185]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: %[[VAL_192:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_193:.*]] = arith.addi %[[VAL_184]], %[[VAL_192]] : index
  ! CHECK: %[[VAL_194:.*]] = fir.array_coor %[[VAL_96]](%[[VAL_110]]) %[[VAL_193]] : (!fir.heap<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_195:.*]] = fir.load %[[VAL_194]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[VAL_196:.*]] = fir.convert %[[VAL_195]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[VAL_197:.*]] = fir.if %[[VAL_196]] -> (!fir.array<10xf32>) {
  ! CHECK: fir.result %[[VAL_185]] : !fir.array<10xf32>
  ! CHECK: } else {
  ! CHECK: %[[VAL_198:.*]] = fir.array_fetch %[[VAL_178]], %[[VAL_184]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[VAL_199:.*]] = divf %[[VAL_198]], %[[VAL_179]] : f32
  ! CHECK: %[[VAL_200:.*]] = fir.array_update %[[VAL_185]], %[[VAL_199]], %[[VAL_184]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %[[VAL_200]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_201:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.result %[[VAL_202:.*]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_174]], %[[VAL_203:.*]] to %[[VAL_0]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  ! CHECK: fir.freemem %[[VAL_96]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK: fir.freemem %[[VAL_52]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
  ! CHECK: return
  ! CHECK: }

  real :: a(10), b(10)

  ! Statement
  where (a > 4.0) b = -a

  ! Construct
  where (a > 100.0)
     b = 2.0 * a
  elsewhere (a > 50.0)
     b = 3.0 + a
     a = a - 1.0
  elsewhere
     a = a / 2.0
  end where
end
