! Test lowering of derived type descriptors builtin data
! RUN: %bbc -emit-fir %s -o - | FileCheck %s

subroutine foo()
  real, save, target :: init_values(10, 10)
  type sometype
    integer :: num = 42
    real, pointer :: values(:, :) => init_values
  end type
  type(sometype), allocatable, save :: x(:)
end subroutine

! CHECK-DAG: fir.global internal @_QFfooE.n.num("num") : !fir.char<1,3>
! CHECK-DAG: fir.global internal @_QFfooE.n.values("values") : !fir.char<1,6>
! CHECK-DAG: fir.global internal @_QFfooE.di.sometype.num : i32
! CHECK-DAG: fir.global internal @_QFfooE.n.sometype("sometype") : !fir.char<1,8>

! CHECK-LABEL: fir.global internal @_QFfooE.c.sometype {{.*}} {
  ! CHECK: fir.address_of(@_QFfooE.n.num)
  ! CHECK: fir.address_of(@_QFfooE.di.sometype.num) : !fir.ref<i32>
  ! CHECK: fir.address_of(@_QFfooE.n.values)
  ! CHECK: fir.address_of(@_QFfooEinit_values)
! CHECK: }

! CHECK-LABEL: fir.global internal @_QFfooE.dt.sometype {{.*}} {
  !CHECK: fir.address_of(@_QFfooE.n.sometype)
  !CHECK: fir.address_of(@_QFfooE.c.sometype)
! CHECK:}

subroutine char_comp_init()
  implicit none  
  type t
     character(8) :: name='Empty'
  end type t
  type(t) :: a
end subroutine

! CHECK-LABEL: fir.global internal @_QFchar_comp_initE.di.t.name("Empty   ") : !fir.char<1,8>
! CHECK-LABEL: fir.global internal @_QFchar_comp_initE.c.t : {{.*}} {
  ! CHECK: fir.address_of(@_QFchar_comp_initE.di.t.name) : !fir.ref<!fir.char<1,8>>
! CHECK: }
