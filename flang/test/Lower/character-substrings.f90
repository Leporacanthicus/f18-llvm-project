! Test character substring lowering
! RUN: bbc %s -o - -emit-fir | FileCheck %s

! Test substring lower where the parent is a scalar-char-literal-constant
! CHECK-LABEL: func @_QPscalar_substring_embox(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>,
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i64>) {
subroutine scalar_substring_embox(i, j)
  ! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,18>>
  ! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
  ! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
  ! CHECK:         %[[VAL_7:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_8:.*]] = subi %[[VAL_5]], %[[VAL_7]] : index
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,18>>) -> !fir.ref<!fir.array<18x!fir.char<1>>>
  ! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_8]] : (!fir.ref<!fir.array<18x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_12:.*]] = subi %[[VAL_6]], %[[VAL_5]] : index
  ! CHECK:         %[[VAL_13:.*]] = addi %[[VAL_12]], %[[VAL_7]] : index
  ! CHECK:         %[[VAL_14:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_15:.*]] = cmpi slt, %[[VAL_13]], %[[VAL_14]] : index
  ! CHECK:         %[[VAL_16:.*]] = select %[[VAL_15]], %[[VAL_14]], %[[VAL_13]] : index
  ! CHECK:         %[[VAL_17:.*]] = fir.emboxchar %[[VAL_11]], %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:         fir.call @_QPbar(%[[VAL_17]]) : (!fir.boxchar<1>) -> ()
  integer(8) :: i, j
  call bar("abcHello World!dfg"(i:j))
  ! CHECK:         return
  ! CHECK:       }
end subroutine scalar_substring_embox

! CHECK-LABEL: func @_QParray_substring_embox(
! CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.boxchar<1>) {
subroutine array_substring_embox(arr)
  ! CHECK:         %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<4x!fir.char<1,7>>>
  ! CHECK:         %[[VAL_3:.*]] = constant 4 : index
  ! CHECK:         %[[VAL_4:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_5:.*]] = addi %[[VAL_4]], %[[VAL_3]] : index
  ! CHECK:         %[[VAL_6:.*]] = subi %[[VAL_5]], %[[VAL_4]] : index
  ! CHECK:         %[[VAL_7:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         %[[VAL_9:.*]] = constant 5 : i64
  ! CHECK:         %[[VAL_10:.*]] = constant 7 : i64
  ! CHECK:         %[[VAL_11:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_12:.*]] = subi %[[VAL_9]], %[[VAL_11]] : i64
  ! CHECK:         %[[VAL_13:.*]] = subi %[[VAL_10]], %[[VAL_12]] : i64
  ! CHECK:         %[[VAL_14:.*]] = fir.slice %[[VAL_4]], %[[VAL_6]], %[[VAL_7]] substr %[[VAL_12]], %[[VAL_13]] : (index, index, i64, i64, i64) -> !fir.slice<1>
  ! CHECK:         %[[VAL_15:.*]] = fir.embox %[[VAL_2]](%[[VAL_8]]) {{\[}}%[[VAL_14]]] : (!fir.ref<!fir.array<4x!fir.char<1,7>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1>>>
  ! CHECK:         fir.call @_QPs(%[[VAL_16]]) : (!fir.box<!fir.array<?x!fir.char<1>>>) -> ()
  interface
    subroutine s(a)
     character(1) :: a(:)
    end subroutine s
  end interface

  character(7) arr(4)

  call s(arr(:)(5:7))
  ! CHECK:         return
  ! CHECK:       }
end subroutine array_substring_embox

! CHECK-LABEL: func @_QPsubstring_assignment(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.boxchar<1>,
! CHECK-SAME:                                %[[VAL_1:.*]]: !fir.boxchar<1>) {
subroutine substring_assignment(a,b)
  ! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_4:.*]] = constant 3 : i64
  ! CHECK:         %[[VAL_5:.*]] = constant 4 : i64
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
  ! CHECK:         %[[VAL_8:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = subi %[[VAL_6]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_9]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_13:.*]] = subi %[[VAL_7]], %[[VAL_6]] : index
  ! CHECK:         %[[VAL_14:.*]] = addi %[[VAL_13]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_15:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_16:.*]] = cmpi slt, %[[VAL_14]], %[[VAL_15]] : index
  ! CHECK:         %[[VAL_17:.*]] = select %[[VAL_16]], %[[VAL_15]], %[[VAL_14]] : index
  ! CHECK:         %[[VAL_18:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_19:.*]] = constant 2 : i64
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
  ! CHECK:         %[[VAL_22:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = subi %[[VAL_20]], %[[VAL_22]] : index
  ! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:         %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_27:.*]] = subi %[[VAL_21]], %[[VAL_20]] : index
  ! CHECK:         %[[VAL_28:.*]] = addi %[[VAL_27]], %[[VAL_22]] : index
  ! CHECK:         %[[VAL_29:.*]] = constant 0 : index
  ! CHECK:         %[[VAL_30:.*]] = cmpi slt, %[[VAL_28]], %[[VAL_29]] : index
  ! CHECK:         %[[VAL_31:.*]] = select %[[VAL_30]], %[[VAL_29]], %[[VAL_28]] : index
  ! CHECK:         %[[VAL_32:.*]] = cmpi slt, %[[VAL_31]], %[[VAL_17]] : index
  ! CHECK:         %[[VAL_33:.*]] = select %[[VAL_32]], %[[VAL_31]], %[[VAL_17]] : index
  ! CHECK:         %[[VAL_34:.*]] = constant 1 : i64
  ! CHECK:         %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK:         %[[VAL_36:.*]] = muli %[[VAL_34]], %[[VAL_35]] : i64
  ! CHECK:         %[[VAL_37:.*]] = constant false
  ! CHECK:         %[[VAL_38:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_39:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_38]], %[[VAL_39]], %[[VAL_36]], %[[VAL_37]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_40:.*]] = constant 1 : index
  ! CHECK:         %[[VAL_41:.*]] = subi %[[VAL_31]], %[[VAL_40]] : index
  ! CHECK:         %[[VAL_42:.*]] = constant 32 : i8
  ! CHECK:         %[[VAL_43:.*]] = fir.undefined !fir.char<1>
  ! CHECK:         %[[VAL_44:.*]] = fir.insert_value %[[VAL_43]], %[[VAL_42]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:         %[[VAL_45:.*]] = constant 1 : index
  ! CHECK:         fir.do_loop %[[VAL_46:.*]] = %[[VAL_33]] to %[[VAL_41]] step %[[VAL_45]] {
  ! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:           %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_47]], %[[VAL_46]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:           fir.store %[[VAL_44]] to %[[VAL_48]] : !fir.ref<!fir.char<1>>
  ! CHECK:         }
  
  character(4) :: a, b
  a(1:2) = b(3:4)
  ! CHECK:         return
  ! CHECK:       }
end subroutine substring_assignment

subroutine array_substring_assignment(a)
  character(5) :: a(6)
  a(:)(3:5) = "BAD"
end subroutine array_substring_assignment

subroutine array_substring_assignment2(a)
  type t
     character(7) :: ch
  end type t
  type(t) :: a(8)
  !a%ch(4:7) = "nice"
end subroutine array_substring_assignment2

subroutine array_substring_assignment3(a,b)
  type t
     character(7) :: ch
  end type t
  type(t) :: a(8), b(8)
  !a%ch(4:7) = b%ch(2:5)
end subroutine array_substring_assignment3
