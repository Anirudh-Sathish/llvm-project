// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(tosa-to-linalg,tosa-to-linalg-named,tosa-to-arith)) " | \
// RUN: mlir-opt -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline | \
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-linalg-to-affine-loops))" | \
// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(custom-affine-loop-fusion))" | \
// RUN: mlir-opt -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>
  %b = arith.constant dense<[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]> : tensor<6xf32>
  %c = tosa.add %a, %b : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %d = tosa.add %c, %b : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %c_unranked = tensor.cast %c : tensor<6xf32> to tensor<*xf32>
  %d_unranked = tensor.cast %d : tensor<6xf32> to tensor<*xf32>
  call @printMemrefF32(%c_unranked) : (tensor<*xf32>) -> ()
  call @printMemrefF32(%d_unranked) : (tensor<*xf32>) -> ()
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [6] strides = [1] data =
// CHECK-NEXT: [7,  7,  7,  7,  7,  7]
// CHECK-NEXT: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [6] strides = [1] data =
// CHECK-NEXT: [13,  12,  11,  10,  9,  8]