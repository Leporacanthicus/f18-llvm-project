! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test that IO item list are lowered and passed correctly

! CHECK-LABEL: func @_QPpass_assumed_len_char_unformatted_io
subroutine pass_assumed_len_char_unformatted_io(c)
  character(*) :: c
  ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  write(1, rec=1) c
  ! CHECK: %[[box:.*]] = fir.embox %[[unbox]]#0 typeparams %[[unbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[castedBox:.*]] = fir.convert %[[box]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[castedBox]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
end

! CHECK-LABEL: func @_QPpass_assumed_len_char_array
subroutine pass_assumed_len_char_array(carray)
  character(*) :: carray(2, 3)
  ! CHECK-DAG: %[[unboxed:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[buffer:.*]] = fir.convert %[[unboxed]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<2x3x!fir.char<1,?>>>
  ! CHECK-DAG: %[[c2:.*]] = constant 2 : index
  ! CHECK-DAG: %[[c3:.*]] = constant 3 : index
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %[[c2]], %[[c3]] : (index, index) -> !fir.shape<2>
  ! CHECK: %[[box:.*]] = fir.embox %[[buffer]](%[[shape]]) typeparams %[[unboxed]]#1 : (!fir.ref<!fir.array<2x3x!fir.char<1,?>>>, !fir.shape<2>, index) -> !fir.box<!fir.array<2x3x!fir.char<1,?>>>
  ! CHECK: %[[descriptor:.*]] = fir.convert %[[box]] : (!fir.box<!fir.array<2x3x!fir.char<1,?>>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[descriptor]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, carray
end

! CHECK-LABEL: func @_QPpass_array_slice_read
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>
subroutine pass_array_slice_read(x)
  real :: x(:)
  read(5, *) x(101:200:2)
  ! CHECK: %[[slice:.*]] = fir.slice %c101{{.*}}, %c200{{.*}}, %c2{{.*}} : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[box:.*]] = fir.rebox %[[x]] [%[[slice]]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[box_cast:.*]] = fir.convert %[[box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioInputDescriptor(%{{.*}}, %[[box_cast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
end

! CHECK-LABEL: func @_QPpass_array_slice_write
! CHECK-SAME: %[[x:.*]]: !fir.box<!fir.array<?xf32>>
subroutine pass_array_slice_write(x)
  real :: x(:)
  write(1, rec=1) x(101:200:2)
  ! CHECK: %[[slice:.*]] = fir.slice %c101{{.*}}, %c200{{.*}}, %c2{{.*}} : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[box:.*]] = fir.rebox %[[x]] [%[[slice]]] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  ! CHECK: %[[box_cast:.*]] = fir.convert %[[box]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[box_cast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
end


! CHECK-LABEL: func @_QPpass_vector_subscript_write(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.array<100xf32>>, %[[j:.*]]: !fir.ref<!fir.array<10xi32>>)
subroutine pass_vector_subscript_write(x, j)
  ! Check that a temp is made for array with vector subscript in output IO.
  integer :: j(10)
  real :: x(100)
  ! CHECK: %[[jload:.*]] = fir.array_load %[[j]](%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
  ! CHECK: %[[xload:.*]] = fir.array_load %[[x]](%{{.*}}) [%{{.*}}] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<100xf32>
  ! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<10xf32>
  ! CHECK: %[[tempload:.*]] = fir.array_load %[[temp]](%{{.*}}) : (!fir.heap<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[copy:.*]] = fir.do_loop
  ! CHECK:   %[[jfetch:.*]] = fir.array_fetch %[[jload]], %{{.*}} : (!fir.array<10xi32>, index) -> i32
  ! CHECK:   %[[jcast:.*]] = fir.convert %[[jfetch]] : (i32) -> index
  ! CHECK:   %[[jindex:.*]] = subi %[[jcast]], %c1{{.*}} : index
  ! CHECK:   %[[xfetch:.*]] = fir.array_fetch %[[xload]], %[[jindex]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK:   %[[update:.*]] = fir.array_update %{{.*}}, %[[xfetch]], %{{.*}} : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:   fir.result %[[update]] : !fir.array<10xf32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[tempload]], %[[copy]] to %[[temp]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.heap<!fir.array<10xf32>>
  ! CHECK: %[[embox:.*]] = fir.embox %[[temp]](%{{.*}}) : (!fir.heap<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  ! CHECK: %[[boxCast:.*]] = fir.convert %[[embox]] : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[boxCast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<10xf32>>
  write(1, rec=1) x(j)
end

! TODO: handle vector subscript in input IO
!subroutine pass_vector_subscript_read(x, j)
!  integer :: j(10)
!  real :: x(100)
!  read(5, *) x(j)
!end subroutine
