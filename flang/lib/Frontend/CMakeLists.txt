get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  TextDiagnosticPrinter.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnostic.cpp

  DEPENDS
  clangBasic

  LINK_LIBS
  FortranParser
  FortranSemantics
  FortranEvaluate
  FortranCommon
  FortranLower
  clangBasic
  clangDriver
  FIRDialect
  FIRAnalysis
  FIRSupport
  FIRTransforms
  FIRBuilder
  FIRRuntime
  ${dialect_libs}
  MLIRAffineToStandard
  MLIRSCFToStandard

  LINK_COMPONENTS
  Option
  Support
  FrontendOpenACC
  FrontendOpenMP
)
