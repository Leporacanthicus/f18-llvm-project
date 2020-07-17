
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(FortranLower
  Bridge.cpp
  CallInterface.cpp
  CharacterExpr.cpp
  CharacterRuntime.cpp
  Coarray.cpp
  ComplexExpr.cpp
  ConvertExpr.cpp
  ConvertType.cpp
  ConvertExpr.cpp
  DoLoopHelper.cpp
  FIRBuilder.cpp
  IntrinsicCall.cpp
  IO.cpp
  Mangler.cpp
  OpenACC.cpp
  OpenMP.cpp
  PFTBuilder.cpp
  Runtime.cpp

  DEPENDS
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  omp_gen
  acc_gen

  LINK_LIBS
  FIRDialect
  FIRSupport
  FIRTransforms
  ${dialect_libs}
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  MLIRAffineToStandard
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandard

  LINK_COMPONENTS
  Support
)
