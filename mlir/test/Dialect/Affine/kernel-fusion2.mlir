// RUN: mlir-opt %s --custom-affine-loop-fusion --split-input-file | FileCheck %s

module {
    func.func @main() {
    %a = memref.alloc() : memref<6xf32>
    %b = memref.alloc() : memref<6xf32>
    %c = memref.alloc() : memref<6xf32>
    %d = memref.alloc() : memref<6xf32>
    %e = memref.alloc() : memref<6xf32>
    %f = memref.alloc() : memref<6xf32>
    %N = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    affine.for %t = %c0 to %N{
        affine.for %i = %c0 to %N {
            %ai = affine.load %a[%i] : memref<6xf32>
            %bi = affine.load %b[%i] : memref<6xf32>
            %sum = arith.addf %ai, %bi : f32
            affine.store %sum, %c[%i] : memref<6xf32>
        }
        affine.for %j = %c0 to %N {
            %cj = affine.load %c[%j] : memref<6xf32>
            %bj = affine.load %b[%j] : memref<6xf32>
            %sum2 = arith.addf %cj, %bj : f32
            affine.store %sum2, %d[%j] : memref<6xf32>
        }
    }
    return
    }
}

// CHECK-LABEL: func.func @main() {
// CHECK-NEXT:   %alloc = memref.alloc() : memref<6xf32>
// CHECK-NEXT:   %alloc_0 = memref.alloc() : memref<6xf32>
// CHECK-NEXT:   %alloc_1 = memref.alloc() : memref<6xf32>
// CHECK-NEXT:   %alloc_2 = memref.alloc() : memref<6xf32>
// CHECK-NEXT:   %alloc_3 = memref.alloc() : memref<6xf32>
// CHECK-NEXT:   %alloc_4 = memref.alloc() : memref<6xf32>
// CHECK-NEXT:   %c6 = arith.constant 6 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   affine.for %arg0 = %c0 to %c6 {
// CHECK-NEXT:     affine.for %arg1 = %c0 to %c6 {
// CHECK-NEXT:       %0 = affine.load %alloc[%arg1] : memref<6xf32>
// CHECK-NEXT:       %1 = affine.load %alloc_0[%arg1] : memref<6xf32>
// CHECK-NEXT:       %2 = arith.addf %0, %1 : f32
// CHECK-NEXT:       affine.store %2, %alloc_1[%arg1] : memref<6xf32>
// CHECK-NEXT:       %3 = affine.load %alloc_1[%arg1] : memref<6xf32>
// CHECK-NEXT:       %4 = affine.load %alloc_0[%arg1] : memref<6xf32>
// CHECK-NEXT:       %5 = arith.addf %3, %4 : f32
// CHECK-NEXT:       affine.store %5, %alloc_2[%arg1] : memref<6xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
