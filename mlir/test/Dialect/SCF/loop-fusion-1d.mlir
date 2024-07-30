// RUN: mlir-opt %s --scf-loop-fusion --split-input-file | FileCheck %s
module {
  memref.global "private" constant @__constant_6xf32_0 : memref<6xf32> = dense<[6.000000e+00, 5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_6xf32 : memref<6xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_6xf32 : memref<6xf32>
    %1 = memref.get_global @__constant_6xf32_0 : memref<6xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<6xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<6xf32>
    scf.for %arg0 = %c0 to %c6 step %c1 {
      %2 = memref.load %0[%arg0] : memref<6xf32>
      %3 = memref.load %1[%arg0] : memref<6xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %alloc[%arg0] : memref<6xf32>
    }
    scf.for %arg0 = %c0 to %c6 step %c1 {
      %2 = memref.load %alloc[%arg0] : memref<6xf32>
      %3 = memref.load %1[%arg0] : memref<6xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %alloc_0[%arg0] : memref<6xf32>
    }
    %cast = memref.cast %alloc : memref<6xf32> to memref<*xf32>
    %cast_1 = memref.cast %alloc_0 : memref<6xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    call @printMemrefF32(%cast_1) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<6xf32>
    memref.dealloc %alloc_0 : memref<6xf32>
    return
  }
}

//CHECK-LABEL: func.func @main() 
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c6 = arith.constant 6 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %0 = memref.get_global @__constant_6xf32 : memref<6xf32>
//CHECK-NEXT: %1 = memref.get_global @__constant_6xf32_0 : memref<6xf32>
//CHECK-NEXT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<6xf32>
//CHECK-NEXT: %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<6xf32>
//CHECK-NEXT: scf.for %arg0 = %c0 to %c6 step %c1 {
//CHECK-NEXT:   %2 = memref.load %0[%arg0] : memref<6xf32>
//CHECK-NEXT:   %3 = memref.load %1[%arg0] : memref<6xf32>
//CHECK-NEXT:   %4 = arith.addf %2, %3 : f32
//CHECK-NEXT:   memref.store %4, %alloc[%arg0] : memref<6xf32>
//CHECK-NEXT:   %5 = memref.load %alloc[%arg0] : memref<6xf32>
//CHECK-NEXT:   %6 = memref.load %1[%arg0] : memref<6xf32>
//CHECK-NEXT:   %7 = arith.addf %5, %6 : f32
//CHECK-NEXT:   memref.store %7, %alloc_0[%arg0] : memref<6xf32>
//CHECK-NEXT: }