! (C) Copyright 2013-2014 ECMWF.


! -----------------------------------------------------------------------------
! Field routines

function atlas_Field__create(params) result(field)
  type(atlas_Field) :: field
  class(atlas_Config), intent(in) :: params
  field%cpp_object_ptr = atlas__Field__create(params%cpp_object_ptr)
end function

function atlas_Field__create_arrayspec(shape,kind,name) result(field)
  type(atlas_Field) :: field
  integer, intent(in) :: shape(:)
  integer, intent(in), optional :: kind
  character(len=*), intent(in), optional :: name

  type(atlas_Config) :: params

  params = atlas_Config()
  call params%set("creator","ArraySpec")
  call params%set("shape",shape)
  call params%set("fortran",.True.)
  if( present(kind) ) call params%set("data_type",atlas_data_type(kind))
  if( present(name) ) call params%set("name",name)

  field%cpp_object_ptr = atlas__Field__create(params%cpp_object_ptr)

  call atlas_delete(params)
end function

subroutine atlas_Field__delete(this)
  class(atlas_Field), intent(inout) :: this
  if ( c_associated(this%cpp_object_ptr) ) then
    call atlas__Field__delete(this%cpp_object_ptr)
  end if
  this%cpp_object_ptr = c_null_ptr
end subroutine

function Field__name(this) result(field_name)
  class(atlas_Field), intent(in) :: this
  character(len=:), allocatable :: field_name
  type(c_ptr) :: field_name_c_str
  field_name_c_str = atlas__Field__name(this%cpp_object_ptr)
  field_name = c_to_f_string_cptr(field_name_c_str)
end function Field__name

function Field__datatype(this) result(datatype)
  class(atlas_Field), intent(in) :: this
  character(len=:), allocatable :: datatype
  type(c_ptr) :: datatype_cptr
  integer(c_int) :: datatype_size
  integer(c_int) :: datatype_allocated
  call atlas__Field__datatype(this%cpp_object_ptr,datatype_cptr,datatype_size,datatype_allocated)
  allocate(character(len=datatype_size) :: datatype )
  datatype= c_to_f_string_cptr(datatype_cptr)
  if( datatype_allocated == 1 ) call atlas_free(datatype_cptr)
end function Field__datatype

function Field__size(this) result(size)
  class(atlas_Field), intent(in) :: this
  integer :: size
  size = atlas__Field__size(this%cpp_object_ptr)
end function Field__size

function Field__bytes(this) result(bytes)
  class(atlas_Field), intent(in) :: this
  real(c_double) :: bytes
  bytes = atlas__Field__bytes(this%cpp_object_ptr)
end function Field__bytes

function Field__metadata(this) result(metadata)
  class(atlas_Field), intent(in) :: this
  type(atlas_Metadata) :: Metadata
  metadata%cpp_object_ptr = atlas__Field__metadata(this%cpp_object_ptr)
end function Field__metadata

function Field__function_space(this) result(function_space)
  class(atlas_Field), intent(in) :: this
  type(atlas_FunctionSpace) :: function_space
  function_space%cpp_object_ptr = atlas__Field__function_space(this%cpp_object_ptr)
end function Field__function_space


function Field__shape(this) result(shape)
  class(atlas_Field), intent(in) :: this
  integer, pointer :: shape(:)
  type(c_ptr) :: shape_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__shapef(this%cpp_object_ptr, shape_c_ptr, field_rank)
  call C_F_POINTER ( shape_c_ptr , shape , (/field_rank/) )
end function Field__shape


subroutine Field__access_data1_int32(this, field)
  class(atlas_Field), intent(in) :: this
  integer, pointer, intent(out) :: field(:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  call atlas__Field__data_shapef_int(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , (/field_size/) )
end subroutine Field__access_data1_int32

subroutine Field__access_data2_int32(this, field)
  class(atlas_Field), intent(in) :: this
  integer, pointer, intent(out) :: field(:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_int(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-1:field_rank) )
end subroutine Field__access_data2_int32

subroutine Field__access_data3_int32(this, field)
  class(atlas_Field), intent(in) :: this
  integer, pointer, intent(out) :: field(:,:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_int(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-2:field_rank) )
end subroutine Field__access_data3_int32

subroutine Field__access_data1_int64(this, field)
  class(atlas_Field), intent(in) :: this
  integer(c_long), pointer, intent(out) :: field(:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  call atlas__Field__data_shapef_long(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , (/field_size/) )
end subroutine Field__access_data1_int64

subroutine Field__access_data2_int64(this, field)
  class(atlas_Field), intent(in) :: this
  integer(c_long), pointer, intent(out) :: field(:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_long(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-1:field_rank) )
end subroutine Field__access_data2_int64

subroutine Field__access_data3_int64(this, field)
  class(atlas_Field), intent(in) :: this
  integer(c_long), pointer, intent(out) :: field(:,:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_long(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-2:field_rank) )
end subroutine Field__access_data3_int64

subroutine Field__access_data1_real32(this, field)
  class(atlas_Field), intent(in) :: this
  real(c_float), pointer, intent(out) :: field(:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  call atlas__Field__data_shapef_float(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , (/field_size/) )
end subroutine Field__access_data1_real32

subroutine Field__access_data2_real32(this, field)
  class(atlas_Field), intent(in) :: this
  real(c_float), pointer, intent(out) :: field(:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_float(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-1:field_rank) )
end subroutine Field__access_data2_real32

subroutine Field__access_data3_real32(this, field)
  class(atlas_Field), intent(in) :: this
  real(c_float), pointer, intent(out) :: field(:,:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_float(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-2:field_rank) )
end subroutine Field__access_data3_real32

subroutine Field__access_data1_real64(this, field)
  class(atlas_Field), intent(in) :: this
  real(c_double), pointer, intent(out) :: field(:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , (/field_size/) )
end subroutine Field__access_data1_real64

subroutine Field__access_data2_real64(this, field)
  class(atlas_Field), intent(in) :: this
  real(c_double), pointer, intent(out) :: field(:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-1:field_rank) )
end subroutine Field__access_data2_real64

subroutine Field__access_data3_real64(this, field)
  class(atlas_Field), intent(in) :: this
  real(c_double), pointer, intent(out) :: field(:,:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-2:field_rank) )
end subroutine Field__access_data3_real64

subroutine Field__access_data2_real64_bounds(this, field, field_bounds)
  class(atlas_Field), intent(in) :: this
  real(c_double), pointer, intent(out) :: field(:,:)
  integer, intent(in) :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_c_ptr , field , field_bounds )
end subroutine Field__access_data2_real64_bounds

subroutine Field__access_data3_real64_bounds(this, field, field_bounds)
  class(atlas_Field), intent(in) :: this
  real(c_double), pointer, intent(out) :: field(:,:,:)
  integer, intent(in) :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_c_ptr , field , field_bounds )
end subroutine Field__access_data3_real64_bounds

subroutine Field__access_data4_real64_bounds(this, field, field_bounds)
  class(atlas_Field), intent(in) :: this
  real(c_double), pointer, intent(out) :: field(:,:,:,:)
  integer, intent(in) :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  call C_F_POINTER ( field_c_ptr , field , field_bounds )
end subroutine Field__access_data4_real64_bounds

function Field__data1_wp(this) result(field)
  class(atlas_Field), intent(in) :: this
  real(wp), pointer :: field(:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  if( wp == c_double ) then
    call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  else if (wp == c_float ) then
    call atlas__Field__data_shapef_float(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  end if
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , (/field_size/) )
end function Field__data1_wp

function Field__data2_wp(this) result(field)
  class(atlas_Field), intent(in) :: this
  real(wp), pointer :: field(:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  if( wp == c_double ) then
    call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  else if (wp == c_float ) then
    call atlas__Field__data_shapef_float(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  end if
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  if (size(field_bounds) < 2) then
    write(0,*) "Cannot access field """,this%name(),""" with rank",field_rank," as rank 2"
    write(0,*) 'call abort()'
  end if
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , field_bounds(field_rank-1:field_rank) )
  if( size(field) /= field_size ) then
    write(0,*) "Requested bounds of field ", this%name(), "[", field_bounds(1:2), &
     & "] do not cover the entire field of size ", field_size
    write(0,*) 'call abort()'
  end if
end function Field__data2_wp

function Field__data3_wp(this) result(field)
  class(atlas_Field), intent(in) :: this
  real(wp), pointer :: field(:,:,:)
  integer, pointer :: field_bounds(:)
  type(c_ptr) :: field_c_ptr
  type(c_ptr) :: field_bounds_c_ptr
  integer(c_int) :: field_rank
  integer :: field_size, jbound
  if( wp == c_double ) then
    call atlas__Field__data_shapef_double(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  else if (wp == c_float ) then
    call atlas__Field__data_shapef_float(this%cpp_object_ptr, field_c_ptr, field_bounds_c_ptr, field_rank)
  end if
  call C_F_POINTER ( field_bounds_c_ptr , field_bounds , (/field_rank/) )
  if (size(field_bounds) < 3) then
    write(0,*) "Cannot access field """,this%name(),""" with rank",field_rank," as rank 3"
    write(0,*) 'call abort()'
  end if
  field_size = 1
  do jbound=1,field_rank
    field_size = field_size * field_bounds(jbound)
  end do
  call C_F_POINTER ( field_c_ptr , field , field_bounds(1:3) )
  if( size(field) /= field_size ) then
    write(0,*) "Requested bounds of field ", field_bounds(1:3), " do not cover the entire field of size ", field_size
    write(0,*) 'call abort()'
  end if
end function Field__data3_wp
