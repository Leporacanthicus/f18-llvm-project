get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FIRRuntime
  Character.cpp
  Derived.cpp
  Numeric.cpp
  Reduction.cpp
  Transformational.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  FIRBuilder
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRSupport
  FIRBuilder
  ${dialect_libs}
)
