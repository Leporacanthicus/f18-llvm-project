// RUN: mlir-opt %s -test-transform-dialect-erase-schedule -one-shot-bufferize="bufferize-function-boundaries" --test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

// RUN: mlir-opt %s -test-transform-dialect-erase-schedule -linalg-flatten-vectors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs" --test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(tensor<*xf32>)

func.func @conv_1d_nwc_wc(%arg0: tensor<3x8x1xf32>, %arg1: tensor<3x1xf32>, %arg2: tensor<3x6x1xf32>) -> tensor<3x6x1xf32> {
  %res = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>,
                         strides = dense<1> : tensor<1xi64>}
     ins (%arg0, %arg1: tensor<3x8x1xf32>, tensor<3x1xf32>)
    outs (%arg2: tensor<3x6x1xf32>) -> (tensor<3x6x1xf32>)
  return %res : tensor<3x6x1xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_1d_nwc_wc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
}

func.func @main() {
  %filter1D_nw = arith.constant dense<[[2.00000e+00], [2.00000e+00], [2.00000e+00]]> : tensor<3x1xf32>

  %in1D_nwc = arith.constant dense<[
	    [[2.00000e+00], [2.00000e+00], [2.00000e+00], [10.00000e+00],
	     [2.00000e+00], [2.00000e+00], [2.00000e+00], [2.00000e+00]],
	    [[2.00000e+00], [2.00000e+00], [2.00000e+00], [2.00000e+00],
	     [2.00000e+00], [2.00000e+00], [2.00000e+00], [2.00000e+00]],
	    [[2.00000e+00], [2.00000e+00], [2.00000e+00], [2.00000e+00],
	     [2.00000e+00], [2.00000e+00], [2.00000e+00], [2.00000e+00]]
	  ]> : tensor<3x8x1xf32>
  %out1D_nwc = arith.constant dense<0.0> : tensor<3x6x1xf32>

  %res = call @conv_1d_nwc_wc(%in1D_nwc, %filter1D_nw, %out1D_nwc) : (tensor<3x8x1xf32>, tensor<3x1xf32>, tensor<3x6x1xf32>) -> (tensor<3x6x1xf32>)
  %out1D_nwc_ = tensor.cast %res : tensor<3x6x1xf32> to tensor<*xf32>
  call @printMemrefF32(%out1D_nwc_): (tensor<*xf32>) -> ()

  return
}

// CHECK:       Unranked Memref {{.*}}
// CHECK-NEXT:  [
// CHECK-SAME:   [
// CHECK-SAME:    [12],
// CHECK-COUNT-3: [28],
// CHECK-NEXT:    [12],
// CHECK-NEXT:    [12]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-5: [12],
// CHECK-NEXT:    [12]
// CHECK-SAME:   ],
// CHECK-NEXT:   [
// CHECK-COUNT-5: [12],
// CHECK-NEXT:    [12]
// CHECK-SAME:   ]
// CHECK-SAME:  ]
