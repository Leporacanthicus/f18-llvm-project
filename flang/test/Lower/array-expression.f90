! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1
subroutine test1(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c(n)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[C]]
  ! CHECK: %[[rv:.*]] = addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1

! CHECK-LABEL: func @_QPtest1b
subroutine test1b(a,b,c,d,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c(n), d(n)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK-DAG: %[[D:.*]] = fir.array_load %arg3(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[C]]
  ! CHECK: %[[rv1:.*]] = addf %[[Bi]], %[[Ci]]
  ! CHECK: %[[Di:.*]] = fir.array_fetch %[[D]]
  ! CHECK: %[[rv:.*]] = addf %[[rv1]], %[[Di]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = b + c + d
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test1b

! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME: %[[aarg:[^:]*]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME: %[[barg:[^:]+]]: !fir.box<!fir.array<?xf32>>,
! CHECK-SAME: %[[carg:[^:]+]]: !fir.box<!fir.array<?xf32>>)
subroutine test2(a,b,c)
  real, intent(out) :: a(:)
  real, intent(in) :: b(:), c(:)
  ! CHECK: %[[a:.*]] = fir.array_load %[[aarg]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[b:.*]] = fir.array_load %[[barg]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %[[c:.*]] = fir.array_load %[[carg]] : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
  ! CHECK: %{{[^:]+}}:3 = fir.box_dims %[[aarg]], %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  ! CHECK: fir.do_loop {{.*}} iter_args(%{{.*}} = %[[a]]) -> (!fir.array<?xf32>
  ! CHECK: fir.array_fetch %[[b]], %{{.*}} : (!fir.array<?xf32>, index) -> f32
  ! CHECK: fir.array_fetch %[[c]], %{{.*}} : (!fir.array<?xf32>, index) -> f32
  ! CHECK: fir.array_update %{{.*}} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK: fir.array_merge_store %[[a]], %{{.*}} to %[[aarg]] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
 a = b + c
end subroutine test2

! CHECK-LABEL: func @_QPtest3
subroutine test3(a,b,c,n)
  integer :: n
  real, intent(out) :: a(n)
  real, intent(in) :: b(n), c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK-DAG: %[[C:.*]] = fir.load %arg2
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK: %[[Bi:.*]] = fir.array_fetch %[[B]]
  ! CHECK: %[[rv:.*]] = addf %[[Bi]], %[[C]]
  ! CHECK: %[[Ti:.*]] = fir.array_update %{{.*}}, %[[rv]], %
  ! CHECK: fir.result %[[Ti]]
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %[[T]] to %arg0
end subroutine test3

! CHECK-LABEL: func @_QPtest4
subroutine test4(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
  real :: a(100) ! FIXME: fake it for now
  real, intent(in) :: b(:), c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test4

! CHECK-LABEL: func @_QPtest5
subroutine test5(a,b,c)
! TODO: this declaration fails in CallInterface lowering
!  real, allocatable, intent(out) :: a(:)
!  real, pointer, intent(in) :: b(:)
  real :: a(100), b(100) ! FIXME: fake it for now
  real, intent(in) :: c
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: fir.do_loop
  ! CHECK: fir.array_fetch %[[B]], %
  ! CHECK: fir.array_update
  a = b + c
  ! CHECK: fir.array_merge_store %[[A]], %{{.*}} to %arg0
end subroutine test5

! CHECK-LABEL: func @_QPtest6(
! CHECK-SAME: %[[aarg:[^:]+]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[barg:[^:]+]]: !fir.ref<!fir.array<?xf32>>,
! CHECK-SAME: %[[carg:[^:]+]]: !fir.ref<f32>,
! CHECK-SAME: %[[narg:[^:]+]]: !fir.ref<i32>,
! CHECK-SAME: %[[marg:[^:]+]]: !fir.ref<i32>)
subroutine test6(a,b,c,n,m)
  integer :: n, m
  real, intent(out) :: a(n)
  real, intent(in) :: b(m), c
  ! CHECK: %[[ashape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[aslice:.*]] = fir.slice %c3{{.*}}, %{{.*}}, %c4{{.*}} : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[a:.*]] = fir.array_load %[[aarg]](%[[ashape]]) [%[[aslice]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<?xf32>
  ! CHECK: %[[bshape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[b:.*]] = fir.array_load %[[barg]](%[[bshape]]) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
  ! CHECK: %[[loop:.*]] = fir.do_loop {{.*}} iter_args(%{{.*}} = %[[a]]) ->
  ! CHECK: %[[bv:.*]] = fir.array_fetch %[[b]], %{{.*}} : (!fir.array<?xf32>, index) -> f32
  ! CHECK: %[[sum:.*]] = addf %[[bv]], %{{.*}} : f32
  ! CHECK: %[[res:.*]] = fir.array_update %{{.*}}, %[[sum]], %{{.*}} : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<?xf32>
  ! CHECK: fir.array_merge_store %[[a]], %[[loop]] to %[[aarg]][%{{.*}}] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.slice<1>
  a(3:n:4) = b + c
end subroutine test6

! CHECK-LABEL: func @_QPtest6a
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<10x50xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<10xf32>>)
subroutine test6a(a,b)
  ! copy part of 1 row to b. a's projection has rank 1.
  real :: a(10,50)
  real :: b(10)
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %{{.*}} = fir.array_load %[[b]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i64, index, index, i64, i64, i64) -> !fir.slice<2>
  ! CHECK: %{{.*}} = fir.array_load %[[a]](%[[shape]]) [%[[slice]]] : (!fir.ref<!fir.array<10x50xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x50xf32>
  ! CHECK: %{{.*}} = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10xf32>)
  ! CHECK: %[[fetch:.*]] = fir.array_fetch %{{.*}}, %{{.*}}, %[[i]] : (!fir.array<10x50xf32>, index, index) -> f32
  ! CHECK: %[[update:.*]] = fir.array_update %{{.*}}, %[[fetch]], %[[i]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK: fir.result %{{.*}} : !fir.array<10xf32>
  ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[b]] : !fir.array<10xf32>, !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>
  b = a(4,41:50)
end subroutine test6a

! CHECK-LABEL: func @_QPtest6b
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<10x50xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<10xf32>>)
subroutine test6b(a,b)
  ! copy b to columns 41 to 50 of row 4 of a
  real :: a(10,50)
  real :: b(10)
  ! CHECK-DAG: %[[shape:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK-DAG: %[[slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i64, index, index, i64, i64, i64) -> !fir.slice<2>
  ! CHECK: %{{.*}} = fir.array_load %[[a]](%[[shape]]) [%[[slice]]] : (!fir.ref<!fir.array<10x50xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.array<10x50xf32>
  ! CHECK: %[[shape:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %{{.*}} = fir.array_load %[[b]](%[[shape]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %{{.*}} = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<10x50xf32>) {
  ! CHECK: %[[fetch:.*]] = fir.array_fetch %{{.*}}, %[[i]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK: %[[update:.*]] = fir.array_update %{{.*}}, %[[fetch]], %{{.*}}, %[[i]] : (!fir.array<10x50xf32>, f32, index, index) -> !fir.array<10x50xf32>
  ! CHECK: fir.result %{{.*}} : !fir.array<10x50xf32>
  ! CHECK: fir.array_merge_store %{{.*}}, %{{.*}} to %[[a]][%{{.*}}] : !fir.array<10x50xf32>, !fir.array<10x50xf32>, !fir.ref<!fir.array<10x50xf32>>, !fir.slice<2>
  a(4,41:50) = b
end subroutine test6b

! This is NOT a conflict. `a` appears on both the lhs and rhs here, but there
! are no loop-carried dependences and no copy is needed.
! CHECK-LABEL: func @_QPtest7
subroutine test7(a,b,n)
  integer :: n
  real, intent(inout) :: a(n)
  real, intent(in) :: b(n)
  ! CHECK: %[[Aout:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[Ain:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1(%
  ! CHECK: %[[T:.*]] = fir.do_loop
  ! CHECK-DAG: %[[Bi:.*]] = fir.array_fetch %[[Ain]]
  ! CHECK-DAG: %[[Ci:.*]] = fir.array_fetch %[[B]]
  ! CHECK: %[[rv:.*]] = addf %[[Bi]], %[[Ci]]
  ! CHECK: fir.array_update %{{.*}}, %[[rv]], %
  a = a + b
  ! CHECK: fir.array_merge_store %[[Aout]], %[[T]] to %arg0
end subroutine test7

! CHECK-LABEL: func @_QPtest8
subroutine test8(a,b)
  integer :: a(100), b(100)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0(%
  ! CHECK-DAG: %[[B1_addr:.*]] = fir.coordinate_of %arg1, %
  ! CHECK: %[[B1:.*]] = fir.load %[[B1_addr]]
  ! CHECK: %[[LOOP:.*]] = fir.do_loop
  ! CHECK: fir.array_update %{{.*}}, %[[B1]], %
  a = b(1)
  ! CHECK: fir.array_merge_store %[[A]], %[[LOOP]] to %arg0
end subroutine test8

! CHECK-LABEL: func @_QPtest10
subroutine test10(a,b,c,d)
  interface
     ! Function takea an array and yields an array
     function foo(a) result(res)
       real :: a(:)  ! FIXME: must be before res or semantics fails
                     ! as `size(a,1)` fails to resolve to the argument
       real, dimension(size(a,1)) :: res
     end function foo
  end interface
  interface
     ! Function takes an array and yields a scalar
     real function bar(a)
       real :: a(:)
     end function bar
  end interface
  real :: a(:), b(:), c(:), d(:)
!  a = b + foo(c + foo(d + bar(a)))
end subroutine test10

! CHECK-LABEL: func @_QPtest11
subroutine test11(a,b,c,d)
  real, external :: bar
  real :: a(100), b(100), c(100), d(100)
  ! CHECK-DAG: %[[A:.*]] = fir.array_load %arg0
  ! CHECK-DAG: %[[B:.*]] = fir.array_load %arg1
  ! CHECK-DAG: %[[C:.*]] = fir.array_load %arg2
  ! CHECK-DAG: %[[D:.*]] = fir.array_load %arg3
  ! CHECK-DAG: %[[tmp:.*]] = fir.allocmem
  ! CHECK-DAG: %[[T:.*]] = fir.array_load %[[tmp]]
  
  !    temporary <- c + d
  ! CHECK: %[[bar_in:.*]] = fir.do_loop
  !  CHECK-DAG: %[[c_i:.*]] = fir.array_fetch %[[C]]
  !  CHECK-DAG: %[[d_i:.*]] = fir.array_fetch %[[D]]
  !  CHECK: %[[sum:.*]] = addf %[[c_i]], %[[d_i]]
  !  CHECK: fir.array_update %{{.*}}, %[[sum]], %
  ! CHECK: fir.array_merge_store %[[T]], %[[bar_in]] to %[[tmp]]
  ! CHECK: %[[cast:.*]] = fir.convert %[[tmp]]
  ! CHECK: %[[bar_out:.*]] = fir.call @_QPbar(%[[cast]]
  
  !    a <- b + bar(?)
  ! CHECK: %[[S:.*]] = fir.do_loop
  !  CHECK: %[[b_i:.*]] = fir.array_fetch %[[B]], %
  !  CHECK: %[[sum2:.*]] = addf %[[b_i]], %[[bar_out]]
  !  CHECK: fir.array_update %{{.*}}, %[[sum2]], %
  ! CHECK: fir.array_merge_store %[[A]], %[[S]] to %arg0
  a = b + bar(c + d)
end subroutine test11

! CHECK-LABEL: func @_QPtest12
subroutine test12(a,b,c,d,n,m)
  integer :: n, m
  ! CHECK: %[[n:.*]] = fir.load %arg4
  ! CHECK: %[[m:.*]] = fir.load %arg5
  ! CHECK: %[[sha:.*]] = fir.shape %
  ! CHECK: %[[A:.*]] = fir.array_load %arg0(%[[sha]])
  ! CHECK: %[[shb:.*]] = fir.shape %
  ! CHECK: %[[B:.*]] = fir.array_load %arg1(%[[shb]])
  ! CHECK: %[[C:.*]] = fir.array_load %arg2(%
  ! CHECK: %[[D:.*]] = fir.array_load %arg3(%
  ! CHECK: %[[tmp:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {{{.*}}uniq_name = ".array.expr"}
  ! CHECK: %[[T:.*]] = fir.array_load %[[tmp]](%
  real, external :: bar
  real :: a(n), b(n), c(m), d(m)
  ! CHECK: %[[LOOP:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[T]])
    ! CHECK-DAG: fir.array_fetch %[[C]]
    ! CHECK-DAG: fir.array_fetch %[[D]]
  ! CHECK: fir.array_merge_store %[[T]], %[[LOOP]]
  ! CHECK: %[[CALL:.*]] = fir.call @_QPbar
  ! CHECK: %[[LOOP2:.*]] = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[A]])
    ! CHECK: fir.array_fetch %[[B]]
  ! CHECK: fir.array_merge_store %[[A]], %[[LOOP2]] to %arg0
  a = b + bar(c + d)
  ! CHECK: fir.freemem %[[tmp]] : !fir.heap<!fir.array<?xf32>>
end subroutine test12

! CHECK-LABEL: func @_QPtest13
subroutine test13(a,b,c,d,n,m,i)
  real :: a(n), b(m)
  complex :: c(n), d(m)
  ! CHECK: %[[A_shape:.*]] = fir.shape %
  ! CHECK: %[[A:.*]] = fir.array_load %arg0(%[[A_shape]])
  ! CHECK: %[[B_shape:.*]] = fir.shape %
  ! CHECK: %[[B_slice:.*]] = fir.slice %
  ! CHECK: %[[B:.*]] = fir.array_load %arg1(%[[B_shape]]) [%[[B_slice]]]
  ! CHECK: %[[C_shape:.*]] = fir.shape %
  ! CHECK: %[[C_slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} path %
  ! CHECK: %[[C:.*]] = fir.array_load %arg2(%[[C_shape]]) [%[[C_slice]]]
  ! CHECK: %[[D_shape:.*]] = fir.shape %
  ! CHECK: %[[D_slice:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} path %
  ! CHECK: %[[D:.*]] = fir.array_load %arg3(%[[D_shape]]) [%[[D_slice]]]
  ! CHECK: = constant -6.2598534E+18 : f32
  ! CHECK: %[[A_result:.*]] = fir.do_loop %{{.*}} = %{{.*}} iter_args(%[[A_in:.*]] = %[[A]]) ->
  ! CHECK: fir.array_fetch %[[B]],
  ! CHECK: fir.array_fetch %[[C]],
  ! CHECK: fir.array_fetch %[[D]],
  ! CHECK: fir.array_update %[[A_in]],
  a = b(i:i+2*n-2:2) + c%im - d(i:i+2*n-2:2)%re + x'deadbeef'
  ! CHECK: fir.array_merge_store %[[A]], %[[A_result]] to %arg0
end subroutine test13

! Test elemental call to function f
! CHECK-LABEL: func @_QPtest14(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test14(a,b)
  ! CHECK: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  interface
     real elemental function f1(i)
       real, intent(in) :: i
     end function f1
  end interface
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[ishift:.*]] = addi %[[i]], %c1{{.*}} : index
  ! CHECK: %[[tmp:.*]] = fir.array_coor %[[a]](%{{.*}}) %[[ishift]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %[[fres:.*]] = fir.call @_QPf1(%[[tmp]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = f1(a)
end subroutine test14

! Test elemental intrinsic function (abs)
! CHECK-LABEL: func @_QPtest15(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test15(a,b)
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK: %[[fres:.*]] = fir.call @llvm.fabs.f32(%[[val]]) : (f32) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = abs(a)
end subroutine test15

! Test elemental call to function f2 with VALUE attribute
! CHECK-LABEL: func @_QPtest16(
! CHECK-SAME: %[[a:.*]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test16(a,b)
  ! CHECK: %[[tmp:.*]] = fir.alloca f32 {adapt.valuebyref
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  interface
     real elemental function f2(i)
       real, VALUE :: i
     end function f2
  end interface
  real :: a(100), b(100)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK: fir.store %[[val]] to %[[tmp]]
  ! CHECK: %[[fres:.*]] = fir.call @_QPf2(%[[tmp]]) : (!fir.ref<f32>) -> f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fres]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
  b = f2(a)
end subroutine test16

! Test elemental impure call to function f3.
!
! CHECK-LABEL: func @_QPtest17(
! CHECK-SAME: %[[a:[^:]+]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[b:[^:]+]]: !fir.ref<!fir.array<100xf32>>,
! CHECK-SAME: %[[c:.*]]: !fir.ref<!fir.array<100xf32>>)
subroutine test17(a,b,c)
  ! CHECK-DAG: %[[aarr:.*]] = fir.array_load %[[a]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  ! CHECK-DAG: %[[barr:.*]] = fir.array_load %[[b]](%{{.*}}) : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>) -> !fir.array<100xf32>
  interface
     real elemental impure function f3(i,j,k)
       real, intent(inout) :: i, j, k
     end function f3
  end interface
  real :: a(100), b(2:101), c(3:102)
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[bth:.*]] = %[[barr]]) -> (!fir.array<100xf32>) {
  ! CHECK-DAG: %[[val:.*]] = fir.array_fetch %[[aarr]], %[[i]] : (!fir.array<100xf32>, index) -> f32
  ! CHECK-DAG: %[[ic:.*]] = addi %[[i]], %c3{{.*}} : index
  ! CHECK-DAG: %[[ccoor:.*]] = fir.array_coor %[[c]](%{{.*}}) %[[ic]] : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  ! CHECK-DAG: %[[ib:.*]] = addi %[[i]], %c2{{.*}} : index
  ! CHECK-DAG: %[[bcoor:.*]] = fir.array_coor %[[b]](%{{.*}}) %[[ib]] : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  ! CHECK-DAG: %[[ia:.*]] = addi %[[i]], %c1{{.*}} : index
  ! CHECK-DAG: %[[acoor:.*]] = fir.array_coor %[[a]](%{{.*}}) %[[ia]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: %[[fres:.*]] = fir.call @_QPf3(%[[ccoor]], %[[bcoor]], %[[acoor]]) : (!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>) -> f32
  ! CHECK: %[[fadd:.*]] = addf %[[val]], %[[fres]] : f32
  ! CHECK: %[[res:.*]] = fir.array_update %[[bth]], %[[fadd]], %[[i]] : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>

  ! See 10.1.4.p2 note 1. The expression below is illegal if `f3` defines the
  ! argument `a` for this statement. Since, this cannot be proven statically by
  ! the compiler, the constraint is left to the user. The compiler may give a
  ! warning that `k` is neither VALUE nor INTENT(IN) and the actual argument,
  ! `a`, appears elsewhere in the same statement.
  b = a + f3(c, b, a)

  ! CHECK: fir.result %[[res]] : !fir.array<100xf32>
  ! CHECK: fir.array_merge_store %[[barr]], %[[loop]] to %[[b]]
end subroutine test17

! CHECK-LABEL: func @_QPtest18(
subroutine test18
  integer, target :: array(10,10)
  integer, pointer :: row_i(:)
  ! CHECK: %[[iaddr:.*]] = fir.alloca i32 {{{.*}}uniq_name = "_QFtest18Ei"}
  ! CHECK: %[[i:.*]] = fir.load %[[iaddr]] : !fir.ref<i32>
  ! CHECK: %[[icast:.*]] = fir.convert %[[i]] : (i32) -> i64
  ! CHECK: %[[exact:.*]] = fir.undefined index
  ! CHECK: %[[ubound:.*]] = subi %{{.*}}, %c1 : index
  ! CHECK: %[[slice:.*]] = fir.slice %[[icast]], %[[exact]], %[[exact]], %c1, %[[ubound]], %c1{{.*}} : (i64, index, index, index, index, i64) -> !fir.slice<2>
  ! CHECK: = fir.embox %{{.*}}(%{{.*}}) [%[[slice]]] : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xi32>>
  row_i => array(i, :)
end subroutine test18

! CHECK-LABEL: func @_QPtest_column_and_row_order(
subroutine test_column_and_row_order(x)
  real :: x(2,3)
  ! CHECK-DAG: %[[c2:.*]] = fir.convert %c2{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[number_of_rows:.*]] = subi %[[c2]], %c1{{.*}} : index
  ! CHECK-DAG: %[[c3:.*]] = fir.convert %c3{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[number_of_columns:.*]] = subi %[[c3]], %c1{{.*}} : index
  ! CHECK: fir.do_loop %[[column:.*]] = %c0{{.*}} to %[[number_of_columns]]
  ! CHECK: fir.do_loop %[[row:.*]] = %c0{{.*}} to %[[number_of_rows]]
  ! CHECK: = fir.array_update %{{.*}}, %{{.*}}, %[[row]], %[[column]] : (!fir.array<2x3xf32>, f32, index, index) -> !fir.array<2x3xf32>
  x = 42
end subroutine

! CHECK-LABEL: func @_QPtest_assigning_to_assumed_shape_slices(
! CHECK-SAME:  %[[x:.*]]: !fir.box<!fir.array<?xi32>>
subroutine test_assigning_to_assumed_shape_slices(x)
  integer :: x(:)
  ! CHECK: fir.box_dims %[[x]], %c0{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  ! CHECK: %[[slice:.*]] = fir.array_load %[[x]] [%{{.*}}] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>) -> !fir.array<?xi32>
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[idx:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[dest:.*]] = %[[slice]]) -> (!fir.array<?xi32>) {
    ! CHECK: %[[res:.*]] = fir.array_update %[[dest]], %c42{{.*}}, %[[idx]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
    ! CHECK: fir.result %[[res]] : !fir.array<?xi32>
  ! CHECK: fir.array_merge_store %[[slice]], %[[loop]] to %[[x]][%{{.*}}] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>, !fir.slice<1>
  x(::2) = 42
end subroutine

! CHECK-LABEL: func @_QPtest19a(
! CHECK-SAME:                   %[[VAL_0:.*]]: !fir.boxchar<1>,
! CHECK-SAME:                   %[[VAL_1:.*]]: !fir.boxchar<1>) {
subroutine test19a(a,b)
  character(LEN=10) a(10)
  character(LEN=10) b(10)
  ! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
  ! CHECK: %[[VAL_4:.*]] = constant 10 : index
  ! CHECK: %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
  ! CHECK: %[[VAL_7:.*]] = constant 10 : index
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<10x!fir.char<1,10>>
  ! CHECK: %[[VAL_10:.*]] = constant 10 : i64
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_13:.*]] = fir.array_load %[[VAL_6]](%[[VAL_12]]) : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<10x!fir.char<1,10>>
  ! CHECK: %[[VAL_14:.*]] = constant 1 : index
  ! CHECK: %[[VAL_15:.*]] = constant 0 : index
  ! CHECK: %[[VAL_16:.*]] = subi %[[VAL_11]], %[[VAL_14]] : index
  ! CHECK: %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_9]]) -> (!fir.array<10x!fir.char<1,10>>) {
  ! CHECK: %[[VAL_20:.*]] = fir.array_access %[[VAL_13]], %[[VAL_18]] : (!fir.array<10x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_21:.*]] = fir.array_access %[[VAL_19]], %[[VAL_18]] : (!fir.array<10x!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_23:.*]] = constant 10 : index
  ! CHECK: %[[VAL_22:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
  ! CHECK: %[[VAL_31:.*]] = muli %[[VAL_22]], %[[VAL_30]] : i64
  ! CHECK: %[[VAL_32:.*]] = constant false
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_34:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_33]], %[[VAL_34]], %[[VAL_31]], %[[VAL_32]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_35:.*]] = fir.array_amend %[[VAL_19]], %[[VAL_21]] : (!fir.array<10x!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>) -> !fir.array<10x!fir.char<1,10>>
  ! CHECK: fir.result %[[VAL_35]] : !fir.array<10x!fir.char<1,10>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_9]], %[[VAL_36:.*]] to %[[VAL_3]] : !fir.array<10x!fir.char<1,10>>, !fir.array<10x!fir.char<1,10>>, !fir.ref<!fir.array<10x!fir.char<1,10>>>

  a = b
  ! CHECK: return
  ! CHECK: }
end subroutine test19a

! CHECK-LABEL: func @_QPtest19b(
! CHECK-SAME:                   %[[VAL_0:.*]]: !fir.boxchar<2>,
! CHECK-SAME:                   %[[VAL_1:.*]]: !fir.boxchar<2>) {
subroutine test19b(a,b)
  character(KIND=2, LEN=8) a(20)
  character(KIND=2, LEN=10) b(20)
  ! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<20x!fir.char<2,8>>>
  ! CHECK: %[[VAL_4:.*]] = constant 20 : index
  ! CHECK: %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  ! CHECK: %[[VAL_6:.*]] = constant 10 : index
  ! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<20x!fir.char<2,10>>>
  ! CHECK: %[[VAL_8:.*]] = constant 20 : index
  ! CHECK: %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<20x!fir.char<2,8>>>, !fir.shape<1>) -> !fir.array<20x!fir.char<2,8>>
  ! CHECK: %[[VAL_11:.*]] = constant 20 : i64
  ! CHECK: %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
  ! CHECK: %[[VAL_13:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[VAL_7]](%[[VAL_13]]) : (!fir.ref<!fir.array<20x!fir.char<2,10>>>, !fir.shape<1>) -> !fir.array<20x!fir.char<2,10>>
  ! CHECK: %[[VAL_15:.*]] = constant 1 : index
  ! CHECK: %[[VAL_16:.*]] = constant 0 : index
  ! CHECK: %[[VAL_17:.*]] = subi %[[VAL_12]], %[[VAL_15]] : index
  ! CHECK: %[[VAL_18:.*]] = fir.do_loop %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_17]] step %[[VAL_15]] unordered iter_args(%[[VAL_20:.*]] = %[[VAL_10]]) -> (!fir.array<20x!fir.char<2,8>>) {
  ! CHECK: %[[VAL_21:.*]] = fir.array_access %[[VAL_14]], %[[VAL_19]] : (!fir.array<20x!fir.char<2,10>>, index) -> !fir.ref<!fir.char<2,10>>
  ! CHECK: %[[VAL_22:.*]] = fir.array_access %[[VAL_20]], %[[VAL_19]] : (!fir.array<20x!fir.char<2,8>>, index) -> !fir.ref<!fir.char<2,8>>
  ! CHECK: %[[VAL_23:.*]] = constant 8 : index
  ! CHECK: %[[VAL_24:.*]] = cmpi slt, %[[VAL_23]], %[[VAL_6]] : index
  ! CHECK: %[[VAL_25:.*]] = select %[[VAL_24]], %[[VAL_23]], %[[VAL_6]] : index
  ! CHECK: %[[VAL_26:.*]] = constant 2 : i64
  ! CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (index) -> i64
  ! CHECK: %[[VAL_28:.*]] = muli %[[VAL_26]], %[[VAL_27]] : i64
  ! CHECK: %[[VAL_29:.*]] = constant false
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<2,8>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<2,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_30]], %[[VAL_31]], %[[VAL_28]], %[[VAL_29]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_32:.*]] = constant 1 : index
  ! CHECK: %[[VAL_33:.*]] = subi %[[VAL_23]], %[[VAL_32]] : index
  ! CHECK: %[[VAL_34:.*]] = constant 32 : i16
  ! CHECK: %[[VAL_35:.*]] = fir.undefined !fir.char<2>
  ! CHECK: %[[VAL_36:.*]] = fir.insert_value %[[VAL_35]], %[[VAL_34]], [0 : index] : (!fir.char<2>, i16) -> !fir.char<2>
  ! CHECK: %[[VAL_37:.*]] = constant 1 : index
  ! CHECK: fir.do_loop %[[VAL_38:.*]] = %[[VAL_25]] to %[[VAL_33]] step %[[VAL_37]] {
  ! CHECK: %[[VAL_39:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<2,8>>) -> !fir.ref<!fir.array<8x!fir.char<2>>>
  ! CHECK: %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_39]], %[[VAL_38]] : (!fir.ref<!fir.array<8x!fir.char<2>>>, index) -> !fir.ref<!fir.char<2>>
  ! CHECK: fir.store %[[VAL_36]] to %[[VAL_40]] : !fir.ref<!fir.char<2>>
  ! CHECK: }
  ! CHECK: %[[VAL_41:.*]] = fir.array_amend %[[VAL_20]], %[[VAL_22]] : (!fir.array<20x!fir.char<2,8>>, !fir.ref<!fir.char<2,8>>) -> !fir.array<20x!fir.char<2,8>>
  ! CHECK: fir.result %[[VAL_41]] : !fir.array<20x!fir.char<2,8>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_10]], %[[VAL_42:.*]] to %[[VAL_3]] : !fir.array<20x!fir.char<2,8>>, !fir.array<20x!fir.char<2,8>>, !fir.ref<!fir.array<20x!fir.char<2,8>>>

  a = b
  ! CHECK: return
  ! CHECK: }
end subroutine test19b

! CHECK-LABEL: func @_QPtest19c(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<4>, %[[VAL_1:.*]]: !fir.boxchar<4>, %[[VAL_2:.*]]: !fir.ref<i32>) {
subroutine test19c(a,b,i)
  character(KIND=4, LEN=i) a(30)
  character(KIND=4, LEN=10) b(30)
  ! CHECK: %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  ! CHECK: %[[VAL_4:.*]] = constant 10 : index
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<30x!fir.char<4,10>>>
  ! CHECK: %[[VAL_6:.*]] = constant 30 : index
  ! CHECK: %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<30x!fir.char<4,?>>>
  ! CHECK: %[[VAL_10:.*]] = constant 30 : index
  ! CHECK: %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_12:.*]] = fir.array_load %[[VAL_9]](%[[VAL_11]]) typeparams %[[VAL_8]] : (!fir.ref<!fir.array<30x!fir.char<4,?>>>, !fir.shape<1>, i32) -> !fir.array<30x!fir.char<4,?>>
  ! CHECK: %[[VAL_13:.*]] = constant 30 : i64
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
  ! CHECK: %[[VAL_15:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_16:.*]] = fir.array_load %[[VAL_5]](%[[VAL_15]]) : (!fir.ref<!fir.array<30x!fir.char<4,10>>>, !fir.shape<1>) -> !fir.array<30x!fir.char<4,10>>
  ! CHECK: %[[VAL_17:.*]] = constant 1 : index
  ! CHECK: %[[VAL_18:.*]] = constant 0 : index
  ! CHECK: %[[VAL_19:.*]] = subi %[[VAL_14]], %[[VAL_17]] : index
  ! CHECK: %[[VAL_20:.*]] = fir.do_loop %[[VAL_21:.*]] = %[[VAL_18]] to %[[VAL_19]] step %[[VAL_17]] unordered iter_args(%[[VAL_22:.*]] = %[[VAL_12]]) -> (!fir.array<30x!fir.char<4,?>>) {
  ! CHECK: %[[VAL_23:.*]] = fir.array_access %[[VAL_16]], %[[VAL_21]] : (!fir.array<30x!fir.char<4,10>>, index) -> !fir.ref<!fir.char<4,10>>
  ! CHECK: %[[VAL_24:.*]] = fir.array_access %[[VAL_22]], %[[VAL_21]] typeparams %[[VAL_8]] : (!fir.array<30x!fir.char<4,?>>, index, i32) -> !fir.ref<!fir.char<4,?>>
  ! CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
  ! CHECK: %[[VAL_26:.*]] = cmpi slt, %[[VAL_25]], %[[VAL_4]] : index
  ! CHECK: %[[VAL_27:.*]] = select %[[VAL_26]], %[[VAL_25]], %[[VAL_4]] : index
  ! CHECK: %[[VAL_28:.*]] = constant 4 : i64
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (index) -> i64
  ! CHECK: %[[VAL_30:.*]] = muli %[[VAL_28]], %[[VAL_29]] : i64
  ! CHECK: %[[VAL_31:.*]] = constant false
  ! CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.char<4,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_32]], %[[VAL_33]], %[[VAL_30]], %[[VAL_31]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_34:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_35:.*]] = subi %[[VAL_8]], %[[VAL_34]] : i32
  ! CHECK: %[[VAL_36:.*]] = constant 32 : i32
  ! CHECK: %[[VAL_37:.*]] = fir.undefined !fir.char<4>
  ! CHECK: %[[VAL_38:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_36]], [0 : index] : (!fir.char<4>, i32) -> !fir.char<4>
  ! CHECK: %[[VAL_39:.*]] = constant 1 : index
  ! CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
  ! CHECK: fir.do_loop %[[VAL_41:.*]] = %[[VAL_27]] to %[[VAL_40]] step %[[VAL_39]] {
  ! CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
  ! CHECK: %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_42]], %[[VAL_41]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
  ! CHECK: fir.store %[[VAL_38]] to %[[VAL_43]] : !fir.ref<!fir.char<4>>
  ! CHECK: }
  ! CHECK: %[[VAL_44:.*]] = fir.array_amend %[[VAL_22]], %[[VAL_24]] : (!fir.array<30x!fir.char<4,?>>, !fir.ref<!fir.char<4,?>>) -> !fir.array<30x!fir.char<4,?>>
  ! CHECK: fir.result %[[VAL_44]] : !fir.array<30x!fir.char<4,?>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_12]], %[[VAL_45:.*]] to %[[VAL_9]] typeparams %[[VAL_8]] : !fir.array<30x!fir.char<4,?>>, !fir.array<30x!fir.char<4,?>>, !fir.ref<!fir.array<30x!fir.char<4,?>>>, i32

  a = b
  ! CHECK: return
  ! CHECK: }
end subroutine test19c

! CHECK-LABEL: func @_QPtest19d(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<1>, %[[VAL_1:.*]]: !fir.boxchar<1>,
! CHECK-SAME:                   %[[VAL_2:.*]]: !fir.ref<i32>,
! CHECK-SAME:                   %[[VAL_3:.*]]: !fir.ref<i32>) {
subroutine test19d(a,b,i,j)
  character(i) a(40)
  character(j) b(40)
  ! CHECK: %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<40x!fir.char<1,?>>>
  ! CHECK: %[[VAL_7:.*]] = constant 40 : index
  ! CHECK: %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<40x!fir.char<1,?>>>
  ! CHECK: %[[VAL_11:.*]] = constant 40 : index
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_13:.*]] = fir.array_load %[[VAL_6]](%[[VAL_12]]) typeparams %[[VAL_5]] : (!fir.ref<!fir.array<40x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<40x!fir.char<1,?>>
  ! CHECK: %[[VAL_14:.*]] = constant 40 : i64
  ! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
  ! CHECK: %[[VAL_16:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_17:.*]] = fir.array_load %[[VAL_10]](%[[VAL_16]]) typeparams %[[VAL_9]] : (!fir.ref<!fir.array<40x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<40x!fir.char<1,?>>
  ! CHECK: %[[VAL_18:.*]] = constant 1 : index
  ! CHECK: %[[VAL_19:.*]] = constant 0 : index
  ! CHECK: %[[VAL_20:.*]] = subi %[[VAL_15]], %[[VAL_18]] : index
  ! CHECK: %[[VAL_21:.*]] = fir.do_loop %[[VAL_22:.*]] = %[[VAL_19]] to %[[VAL_20]] step %[[VAL_18]] unordered iter_args(%[[VAL_23:.*]] = %[[VAL_13]]) -> (!fir.array<40x!fir.char<1,?>>) {
  ! CHECK: %[[VAL_24:.*]] = fir.array_access %[[VAL_17]], %[[VAL_22]] typeparams %[[VAL_9]] : (!fir.array<40x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_25:.*]] = fir.array_access %[[VAL_23]], %[[VAL_22]] typeparams %[[VAL_5]] : (!fir.array<40x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
  ! CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
  ! CHECK: %[[VAL_28:.*]] = cmpi slt, %[[VAL_26]], %[[VAL_27]] : index
  ! CHECK: %[[VAL_29:.*]] = select %[[VAL_28]], %[[VAL_26]], %[[VAL_27]] : index
  ! CHECK: %[[VAL_30:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
  ! CHECK: %[[VAL_32:.*]] = muli %[[VAL_30]], %[[VAL_31]] : i64
  ! CHECK: %[[VAL_33:.*]] = constant false
  ! CHECK: %[[VAL_34:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_34]], %[[VAL_35]], %[[VAL_32]], %[[VAL_33]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_36:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_37:.*]] = subi %[[VAL_5]], %[[VAL_36]] : i32
  ! CHECK: %[[VAL_38:.*]] = constant 32 : i8
  ! CHECK: %[[VAL_39:.*]] = fir.undefined !fir.char<1>
  ! CHECK: %[[VAL_40:.*]] = fir.insert_value %[[VAL_39]], %[[VAL_38]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: %[[VAL_41:.*]] = constant 1 : index
  ! CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_37]] : (i32) -> index
  ! CHECK: fir.do_loop %[[VAL_43:.*]] = %[[VAL_29]] to %[[VAL_42]] step %[[VAL_41]] {
  ! CHECK: %[[VAL_44:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_44]], %[[VAL_43]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: fir.store %[[VAL_40]] to %[[VAL_45]] : !fir.ref<!fir.char<1>>
  ! CHECK: }
  ! CHECK: %[[VAL_46:.*]] = fir.array_amend %[[VAL_23]], %[[VAL_25]] : (!fir.array<40x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<40x!fir.char<1,?>>
  ! CHECK: fir.result %[[VAL_46]] : !fir.array<40x!fir.char<1,?>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_13]], %[[VAL_47:.*]] to %[[VAL_6]] typeparams %[[VAL_5]] : !fir.array<40x!fir.char<1,?>>, !fir.array<40x!fir.char<1,?>>, !fir.ref<!fir.array<40x!fir.char<1,?>>>, i32

  a = b
  ! CHECK: return
  ! CHECK: }
end subroutine test19d

! CHECK-LABEL: func @_QPtest19e(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<1>, %[[VAL_1:.*]]: !fir.boxchar<1>) {
subroutine test19e(a,b)
  character(*) a(50)
  character(*) b(50)
  ! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<50x!fir.char<1,?>>>
  ! CHECK: %[[VAL_4:.*]] = constant 50 : index
  ! CHECK: %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<50x!fir.char<1,?>>>
  ! CHECK: %[[VAL_7:.*]] = constant 50 : index
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<50x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<50x!fir.char<1,?>>
  ! CHECK: %[[VAL_10:.*]] = constant 50 : i64
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_13:.*]] = fir.array_load %[[VAL_6]](%[[VAL_12]]) typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.array<50x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<50x!fir.char<1,?>>
  ! CHECK: %[[VAL_14:.*]] = constant 1 : index
  ! CHECK: %[[VAL_15:.*]] = constant 0 : index
  ! CHECK: %[[VAL_16:.*]] = subi %[[VAL_11]], %[[VAL_14]] : index
  ! CHECK: %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_9]]) -> (!fir.array<50x!fir.char<1,?>>) {
  ! CHECK: %[[VAL_20:.*]] = fir.array_access %[[VAL_13]], %[[VAL_18]] typeparams %[[VAL_5]]#1 : (!fir.array<50x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_21:.*]] = fir.array_access %[[VAL_19]], %[[VAL_18]] typeparams %[[VAL_2]]#1 : (!fir.array<50x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_22:.*]] = cmpi slt, %[[VAL_2]]#1, %[[VAL_5]]#1 : index
  ! CHECK: %[[VAL_23:.*]] = select %[[VAL_22]], %[[VAL_2]]#1, %[[VAL_5]]#1 : index
  ! CHECK: %[[VAL_24:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
  ! CHECK: %[[VAL_26:.*]] = muli %[[VAL_24]], %[[VAL_25]] : i64
  ! CHECK: %[[VAL_27:.*]] = constant false
  ! CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_28]], %[[VAL_29]], %[[VAL_26]], %[[VAL_27]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_30:.*]] = constant 1 : index
  ! CHECK: %[[VAL_31:.*]] = subi %[[VAL_2]]#1, %[[VAL_30]] : index
  ! CHECK: %[[VAL_32:.*]] = constant 32 : i8
  ! CHECK: %[[VAL_33:.*]] = fir.undefined !fir.char<1>
  ! CHECK: %[[VAL_34:.*]] = fir.insert_value %[[VAL_33]], %[[VAL_32]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: %[[VAL_35:.*]] = constant 1 : index
  ! CHECK: fir.do_loop %[[VAL_36:.*]] = %[[VAL_23]] to %[[VAL_31]] step %[[VAL_35]] {
  ! CHECK: %[[VAL_37:.*]] = fir.convert %[[VAL_21]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[VAL_38:.*]] = fir.coordinate_of %[[VAL_37]], %[[VAL_36]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: fir.store %[[VAL_34]] to %[[VAL_38]] : !fir.ref<!fir.char<1>>
  ! CHECK: }
  ! CHECK: %[[VAL_39:.*]] = fir.array_amend %[[VAL_19]], %[[VAL_21]] : (!fir.array<50x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<50x!fir.char<1,?>>
  ! CHECK: fir.result %[[VAL_39]] : !fir.array<50x!fir.char<1,?>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_9]], %[[VAL_40:.*]] to %[[VAL_3]] typeparams %[[VAL_2]]#1 : !fir.array<50x!fir.char<1,?>>, !fir.array<50x!fir.char<1,?>>, !fir.ref<!fir.array<50x!fir.char<1,?>>>, index

  a = b
  ! CHECK: return
  ! CHECK: }
end subroutine test19e

! CHECK-LABEL: func @_QPtest19f(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<1>, %[[VAL_1:.*]]: !fir.boxchar<1>) {
subroutine test19f(a,b)
  character(*) a(60)
  character(*) b(60)
  ! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<60x!fir.char<1,?>>>
  ! CHECK: %[[VAL_4:.*]] = constant 60 : index
  ! CHECK: %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<60x!fir.char<1,?>>>
  ! CHECK: %[[VAL_7:.*]] = constant 60 : index
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<60x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<60x!fir.char<1,?>>
  ! CHECK: %[[VAL_10:.*]] = constant 60 : i64
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
  ! CHECK: %[[VAL_12:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,7>>
  ! CHECK: %[[VAL_13:.*]] = constant 7 : index
  ! CHECK: %[[VAL_14:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_15:.*]] = fir.array_load %[[VAL_6]](%[[VAL_14]]) typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.array<60x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.array<60x!fir.char<1,?>>
  ! CHECK: %[[VAL_16:.*]] = constant 1 : index
  ! CHECK: %[[VAL_17:.*]] = constant 0 : index
  ! CHECK: %[[VAL_18:.*]] = subi %[[VAL_11]], %[[VAL_16]] : index
  ! CHECK: %[[VAL_19:.*]] = fir.do_loop %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_18]] step %[[VAL_16]] unordered iter_args(%[[VAL_21:.*]] = %[[VAL_9]]) -> (!fir.array<60x!fir.char<1,?>>) {
  ! CHECK: %[[VAL_22:.*]] = fir.array_access %[[VAL_15]], %[[VAL_20]] typeparams %[[VAL_5]]#1 : (!fir.array<60x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_23:.*]] = addi %[[VAL_13]], %[[VAL_5]]#1 : index
  ! CHECK: %[[VAL_24:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_23]] : index) {bindc_name = ".chrtmp"}
  ! CHECK: %[[VAL_25:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
  ! CHECK: %[[VAL_27:.*]] = muli %[[VAL_25]], %[[VAL_26]] : i64
  ! CHECK: %[[VAL_28:.*]] = constant false
  ! CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_29]], %[[VAL_30]], %[[VAL_27]], %[[VAL_28]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_31:.*]] = constant 1 : index
  ! CHECK: %[[VAL_32:.*]] = subi %[[VAL_23]], %[[VAL_31]] : index
  ! CHECK: fir.do_loop %[[VAL_33:.*]] = %[[VAL_13]] to %[[VAL_32]] step %[[VAL_31]] {
  ! CHECK: %[[VAL_34:.*]] = subi %[[VAL_33]], %[[VAL_13]] : index
  ! CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_22]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[VAL_36:.*]] = fir.coordinate_of %[[VAL_35]], %[[VAL_34]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[VAL_37:.*]] = fir.load %[[VAL_36]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[VAL_39:.*]] = fir.coordinate_of %[[VAL_38]], %[[VAL_33]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: fir.store %[[VAL_37]] to %[[VAL_39]] : !fir.ref<!fir.char<1>>
  ! CHECK: }
  ! CHECK: %[[VAL_40:.*]] = fir.array_access %[[VAL_21]], %[[VAL_20]] typeparams %[[VAL_2]]#1 : (!fir.array<60x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_41:.*]] = cmpi slt, %[[VAL_2]]#1, %[[VAL_23]] : index
  ! CHECK: %[[VAL_42:.*]] = select %[[VAL_41]], %[[VAL_2]]#1, %[[VAL_23]] : index
  ! CHECK: %[[VAL_43:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_44:.*]] = fir.convert %[[VAL_42]] : (index) -> i64
  ! CHECK: %[[VAL_45:.*]] = muli %[[VAL_43]], %[[VAL_44]] : i64
  ! CHECK: %[[VAL_46:.*]] = constant false
  ! CHECK: %[[VAL_47:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_48:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_47]], %[[VAL_48]], %[[VAL_45]], %[[VAL_46]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_49:.*]] = constant 1 : index
  ! CHECK: %[[VAL_50:.*]] = subi %[[VAL_2]]#1, %[[VAL_49]] : index
  ! CHECK: %[[VAL_51:.*]] = constant 32 : i8
  ! CHECK: %[[VAL_52:.*]] = fir.undefined !fir.char<1>
  ! CHECK: %[[VAL_53:.*]] = fir.insert_value %[[VAL_52]], %[[VAL_51]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: %[[VAL_54:.*]] = constant 1 : index
  ! CHECK: fir.do_loop %[[VAL_55:.*]] = %[[VAL_42]] to %[[VAL_50]] step %[[VAL_54]] {
  ! CHECK: %[[VAL_56:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[VAL_57:.*]] = fir.coordinate_of %[[VAL_56]], %[[VAL_55]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: fir.store %[[VAL_53]] to %[[VAL_57]] : !fir.ref<!fir.char<1>>
  ! CHECK: }
  ! CHECK: %[[VAL_58:.*]] = fir.array_amend %[[VAL_21]], %[[VAL_40]] : (!fir.array<60x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<60x!fir.char<1,?>>
  ! CHECK: fir.result %[[VAL_58]] : !fir.array<60x!fir.char<1,?>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_9]], %[[VAL_59:.*]] to %[[VAL_3]] typeparams %[[VAL_2]]#1 : !fir.array<60x!fir.char<1,?>>, !fir.array<60x!fir.char<1,?>>, !fir.ref<!fir.array<60x!fir.char<1,?>>>, index

  a = "prefix " // b
  ! CHECK: return
  ! CHECK: }
end subroutine test19f

! CHECK-LABEL: func @_QPtest19g(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<4>, %[[VAL_1:.*]]: !fir.boxchar<2>,
! CHECK-SAME:                   %[[VAL_2:.*]]: !fir.ref<i32>) {
subroutine test19g(a,b,i)
  character(kind=4,len=i) a(70)
  character(kind=2,len=13) b(140)
  ! CHECK: %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
  ! CHECK: %[[VAL_4:.*]] = constant 13 : index
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<!fir.array<140x!fir.char<2,13>>>
  ! CHECK: %[[VAL_6:.*]] = constant 140 : index
  ! CHECK: %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
  ! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<70x!fir.char<4,?>>>
  ! CHECK: %[[VAL_10:.*]] = constant 70 : index
  ! CHECK: %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_12:.*]] = fir.array_load %[[VAL_9]](%[[VAL_11]]) typeparams %[[VAL_8]] : (!fir.ref<!fir.array<70x!fir.char<4,?>>>, !fir.shape<1>, i32) -> !fir.array<70x!fir.char<4,?>>
  ! CHECK: %[[VAL_13:.*]] = constant 70 : i64
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
  ! CHECK: %[[VAL_15:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_16:.*]] = constant 140 : i64
  ! CHECK: %[[VAL_17:.*]] = constant 2 : i64
  ! CHECK: %[[VAL_18:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_19:.*]] = fir.slice %[[VAL_15]], %[[VAL_16]], %[[VAL_17]] : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[VAL_20:.*]] = fir.array_load %[[VAL_5]](%[[VAL_18]]) {{\[}}%[[VAL_19]]] : (!fir.ref<!fir.array<140x!fir.char<2,13>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<140x!fir.char<2,13>>
  ! CHECK: %[[VAL_21:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> i64
  ! CHECK: %[[VAL_23:.*]] = constant 1 : index
  ! CHECK: %[[VAL_24:.*]] = constant 0 : index
  ! CHECK: %[[VAL_25:.*]] = subi %[[VAL_14]], %[[VAL_23]] : index
  ! CHECK: %[[VAL_26:.*]] = fir.do_loop %[[VAL_27:.*]] = %[[VAL_24]] to %[[VAL_25]] step %[[VAL_23]] unordered iter_args(%[[VAL_28:.*]] = %[[VAL_12]]) -> (!fir.array<70x!fir.char<4,?>>) {
  ! CHECK: %[[VAL_29:.*]] = fir.array_access %[[VAL_20]], %[[VAL_27]] : (!fir.array<140x!fir.char<2,13>>, index) -> !fir.ref<!fir.char<2,13>>
  ! CHECK: %[[VAL_30:.*]] = constant 13 : index
  ! CHECK: %[[VAL_31:.*]] = fir.alloca !fir.char<4,?>(%[[VAL_30]] : index)
  ! CHECK: %[[VAL_32:.*]] = cmpi slt, %[[VAL_30]], %[[VAL_4]] : index
  ! CHECK: %[[VAL_33:.*]] = select %[[VAL_32]], %[[VAL_30]], %[[VAL_4]] : index
  ! CHECK: fir.char_convert %[[VAL_29]] for %[[VAL_33]] to %[[VAL_31]] : !fir.ref<!fir.char<2,13>>, index, !fir.ref<!fir.char<4,?>>
  ! CHECK: %[[VAL_34:.*]] = constant 1 : index
  ! CHECK: %[[VAL_35:.*]] = subi %[[VAL_30]], %[[VAL_34]] : index
  ! CHECK: %[[VAL_36:.*]] = constant 32 : i32
  ! CHECK: %[[VAL_37:.*]] = fir.undefined !fir.char<4>
  ! CHECK: %[[VAL_38:.*]] = fir.insert_value %[[VAL_37]], %[[VAL_36]], [0 : index] : (!fir.char<4>, i32) -> !fir.char<4>
  ! CHECK: %[[VAL_39:.*]] = constant 1 : index
  ! CHECK: fir.do_loop %[[VAL_40:.*]] = %[[VAL_33]] to %[[VAL_35]] step %[[VAL_39]] {
  ! CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
  ! CHECK: %[[VAL_42:.*]] = fir.coordinate_of %[[VAL_41]], %[[VAL_40]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
  ! CHECK: fir.store %[[VAL_38]] to %[[VAL_42]] : !fir.ref<!fir.char<4>>
  ! CHECK: }
  ! CHECK: %[[VAL_43:.*]] = fir.array_access %[[VAL_28]], %[[VAL_27]] typeparams %[[VAL_8]] : (!fir.array<70x!fir.char<4,?>>, index, i32) -> !fir.ref<!fir.char<4,?>>
  ! CHECK: %[[VAL_44:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
  ! CHECK: %[[VAL_45:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
  ! CHECK: %[[VAL_46:.*]] = cmpi slt, %[[VAL_44]], %[[VAL_45]] : index
  ! CHECK: %[[VAL_47:.*]] = select %[[VAL_46]], %[[VAL_44]], %[[VAL_45]] : index
  ! CHECK: %[[VAL_48:.*]] = constant 4 : i64
  ! CHECK: %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (index) -> i64
  ! CHECK: %[[VAL_50:.*]] = muli %[[VAL_48]], %[[VAL_49]] : i64
  ! CHECK: %[[VAL_51:.*]] = constant false
  ! CHECK: %[[VAL_52:.*]] = fir.convert %[[VAL_43]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_53:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_52]], %[[VAL_53]], %[[VAL_50]], %[[VAL_51]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_54:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_55:.*]] = subi %[[VAL_8]], %[[VAL_54]] : i32
  ! CHECK: %[[VAL_56:.*]] = constant 32 : i32
  ! CHECK: %[[VAL_57:.*]] = fir.undefined !fir.char<4>
  ! CHECK: %[[VAL_58:.*]] = fir.insert_value %[[VAL_57]], %[[VAL_56]], [0 : index] : (!fir.char<4>, i32) -> !fir.char<4>
  ! CHECK: %[[VAL_59:.*]] = constant 1 : index
  ! CHECK: %[[VAL_60:.*]] = fir.convert %[[VAL_55]] : (i32) -> index
  ! CHECK: fir.do_loop %[[VAL_61:.*]] = %[[VAL_47]] to %[[VAL_60]] step %[[VAL_59]] {
  ! CHECK: %[[VAL_62:.*]] = fir.convert %[[VAL_43]] : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<!fir.array<?x!fir.char<4>>>
  ! CHECK: %[[VAL_63:.*]] = fir.coordinate_of %[[VAL_62]], %[[VAL_61]] : (!fir.ref<!fir.array<?x!fir.char<4>>>, index) -> !fir.ref<!fir.char<4>>
  ! CHECK: fir.store %[[VAL_58]] to %[[VAL_63]] : !fir.ref<!fir.char<4>>
  ! CHECK: }
  ! CHECK: %[[VAL_64:.*]] = fir.array_amend %[[VAL_28]], %[[VAL_43]] : (!fir.array<70x!fir.char<4,?>>, !fir.ref<!fir.char<4,?>>) -> !fir.array<70x!fir.char<4,?>>
  ! CHECK: fir.result %[[VAL_64]] : !fir.array<70x!fir.char<4,?>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_12]], %[[VAL_65:.*]] to %[[VAL_9]] typeparams %[[VAL_8]] : !fir.array<70x!fir.char<4,?>>, !fir.array<70x!fir.char<4,?>>, !fir.ref<!fir.array<70x!fir.char<4,?>>>, i32

  a = b(1:140:2)
  ! CHECK: return
  ! CHECK: }
end subroutine test19g

! CHECK-LABEL: func @_QPtest19h(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.boxchar<1>, %[[VAL_1:.*]]: !fir.boxchar<1>,
! CHECK-SAME:                   %[[VAL_2:.*]]: !fir.ref<i32>,
! CHECK-SAME:                   %[[VAL_3:.*]]: !fir.ref<i32>) {
subroutine test19h(a,b,i,j)
  character(i) a(70)
  character(*) b(j)
  ! CHECK: %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<70x!fir.char<1,?>>>
  ! CHECK: %[[VAL_7:.*]] = constant 70 : index
  ! CHECK: %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[VAL_10:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> i64
  ! CHECK: %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
  ! CHECK: %[[VAL_13:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[VAL_6]](%[[VAL_13]]) typeparams %[[VAL_5]] : (!fir.ref<!fir.array<70x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<70x!fir.char<1,?>>
  ! CHECK: %[[VAL_15:.*]] = constant 70 : i64
  ! CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
  ! CHECK: %[[VAL_17:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_18:.*]] = constant 140 : i64
  ! CHECK: %[[VAL_19:.*]] = constant 2 : i64
  ! CHECK: %[[VAL_20:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_21:.*]] = fir.slice %[[VAL_17]], %[[VAL_18]], %[[VAL_19]] : (i64, i64, i64) -> !fir.slice<1>
  ! CHECK: %[[VAL_22:.*]] = fir.array_load %[[VAL_9]](%[[VAL_20]]) {{\[}}%[[VAL_21]]] typeparams %[[VAL_8]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.array<?x!fir.char<1,?>>
  ! CHECK: %[[VAL_23:.*]] = constant 1 : index
  ! CHECK: %[[VAL_24:.*]] = constant 0 : index
  ! CHECK: %[[VAL_25:.*]] = subi %[[VAL_16]], %[[VAL_23]] : index
  ! CHECK: %[[VAL_26:.*]] = fir.do_loop %[[VAL_27:.*]] = %[[VAL_24]] to %[[VAL_25]] step %[[VAL_23]] unordered iter_args(%[[VAL_28:.*]] = %[[VAL_14]]) -> (!fir.array<70x!fir.char<1,?>>) {
  ! CHECK: %[[VAL_29:.*]] = fir.array_access %[[VAL_22]], %[[VAL_27]] typeparams %[[VAL_8]]#1 : (!fir.array<?x!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_30:.*]] = fir.array_access %[[VAL_28]], %[[VAL_27]] typeparams %[[VAL_5]] : (!fir.array<70x!fir.char<1,?>>, index, i32) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
  ! CHECK: %[[VAL_32:.*]] = cmpi slt, %[[VAL_31]], %[[VAL_8]]#1 : index
  ! CHECK: %[[VAL_33:.*]] = select %[[VAL_32]], %[[VAL_31]], %[[VAL_8]]#1 : index
  ! CHECK: %[[VAL_34:.*]] = constant 1 : i64
  ! CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK: %[[VAL_36:.*]] = muli %[[VAL_34]], %[[VAL_35]] : i64
  ! CHECK: %[[VAL_37:.*]] = constant false
  ! CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_39:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[VAL_38]], %[[VAL_39]], %[[VAL_36]], %[[VAL_37]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_40:.*]] = constant 1 : i32
  ! CHECK: %[[VAL_41:.*]] = subi %[[VAL_5]], %[[VAL_40]] : i32
  ! CHECK: %[[VAL_42:.*]] = constant 32 : i8
  ! CHECK: %[[VAL_43:.*]] = fir.undefined !fir.char<1>
  ! CHECK: %[[VAL_44:.*]] = fir.insert_value %[[VAL_43]], %[[VAL_42]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: %[[VAL_45:.*]] = constant 1 : index
  ! CHECK: %[[VAL_46:.*]] = fir.convert %[[VAL_41]] : (i32) -> index
  ! CHECK: fir.do_loop %[[VAL_47:.*]] = %[[VAL_33]] to %[[VAL_46]] step %[[VAL_45]] {
  ! CHECK: %[[VAL_48:.*]] = fir.convert %[[VAL_30]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[VAL_49:.*]] = fir.coordinate_of %[[VAL_48]], %[[VAL_47]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: fir.store %[[VAL_44]] to %[[VAL_49]] : !fir.ref<!fir.char<1>>
  ! CHECK: }
  ! CHECK: %[[VAL_50:.*]] = fir.array_amend %[[VAL_28]], %[[VAL_30]] : (!fir.array<70x!fir.char<1,?>>, !fir.ref<!fir.char<1,?>>) -> !fir.array<70x!fir.char<1,?>>
  ! CHECK: fir.result %[[VAL_50]] : !fir.array<70x!fir.char<1,?>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_14]], %[[VAL_51:.*]] to %[[VAL_6]] typeparams %[[VAL_5]] : !fir.array<70x!fir.char<1,?>>, !fir.array<70x!fir.char<1,?>>, !fir.ref<!fir.array<70x!fir.char<1,?>>>, i32

  a = b(1:140:2)
  ! CHECK: return
  ! CHECK: }
end subroutine test19h

! CHECK-LABEL: func @_QPtest_elemental_character_intrinsic(
! CHECK-SAME:            %[[VAL_0:.*]]: !fir.boxchar<1>,
! CHECK-SAME:            %[[VAL_1:.*]]: !fir.boxchar<1>) {
subroutine test_elemental_character_intrinsic(c1, c2)
  ! CHECK: %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
  ! CHECK: %[[VAL_4:.*]] = constant 10 : index
  ! CHECK: %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
  ! CHECK: %[[VAL_7:.*]] = constant 2 : index
  ! CHECK: %[[VAL_8:.*]] = constant 10 : index
  ! CHECK: %[[VAL_9:.*]] = constant -1 : i32
  ! CHECK: %[[VAL_10:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_13:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_9]], %[[VAL_11]], %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_14:.*]] = constant 10 : i64
  ! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
  ! CHECK: %[[VAL_16:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_17:.*]] = fir.shape_shift %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[VAL_18:.*]] = fir.allocmem !fir.array<10xi32>
  ! CHECK: %[[VAL_19:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_20:.*]] = fir.array_load %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
  ! CHECK: %[[VAL_21:.*]] = constant 1 : index
  ! CHECK: %[[VAL_22:.*]] = constant 0 : index
  ! CHECK: %[[VAL_23:.*]] = subi %[[VAL_15]], %[[VAL_21]] : index
  ! CHECK: %[[VAL_24:.*]] = fir.do_loop %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_23]] step %[[VAL_21]] unordered iter_args(%[[VAL_26:.*]] = %[[VAL_20]]) -> (!fir.array<10xi32>) {
  ! CHECK: %[[VAL_27:.*]] = constant 1 : index
  ! CHECK: %[[VAL_28:.*]] = addi %[[VAL_25]], %[[VAL_27]] : index
  ! CHECK: %[[VAL_29:.*]] = fir.array_coor %[[VAL_3]](%[[VAL_16]]) %[[VAL_28]] typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_30:.*]] = addi %[[VAL_25]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_31:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_17]]) %[[VAL_30]] typeparams %[[VAL_5]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shapeshift<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_32:.*]] = constant false
  ! CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_34:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
  ! CHECK: %[[VAL_35:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_36:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
  ! CHECK: %[[VAL_37:.*]] = fir.call @_FortranAScan1(%[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_32]]) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  ! CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> i32
  ! CHECK: %[[VAL_39:.*]] = fir.array_update %[[VAL_26]], %[[VAL_38]], %[[VAL_25]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
  ! CHECK: fir.result %[[VAL_39]] : !fir.array<10xi32>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_20]], %[[VAL_40:.*]] to %[[VAL_18]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.heap<!fir.array<10xi32>>
  ! CHECK: %[[VAL_41:.*]] = fir.shape %[[VAL_15]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_42:.*]] = fir.embox %[[VAL_18]](%[[VAL_41]]) : (!fir.heap<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  ! CHECK: %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  ! CHECK: %[[VAL_44:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_13]], %[[VAL_43]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: fir.freemem %[[VAL_18]] : !fir.heap<!fir.array<10xi32>>
  ! CHECK: %[[VAL_45:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_13]]) : (!fir.ref<i8>) -> i32
  character(*) :: c1(10), c2(2:11)
  print *, scan(c1, c2)
  ! CHECK: return
  ! CHECK: }
end subroutine

! CHECK: func private @_QPbar(
