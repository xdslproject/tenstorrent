# RUN: python3.13 %s | filecheck %s
from tenstorrent.frontend import tt


@tt.data_in
def advanced():
    a = 27 + 9.3
    b = 45

    for i in range(5, 23):
        if b < 12:
            for j in range(9, 17, 4):
                c = j + 1

                for k in range(4, 20):
                    d = tt.cb_pages_reservable_at_back(b, k)

        elif b > 35:
            print("Hello world")

        else:
            tt.cb_wait_front(b, b)


# CHECK:      builtin.module {
# CHECK-NEXT:   builtin.module attributes {kernel_type = "data_in"} {
# CHECK-NEXT:     func.func @kernel_main() {
# CHECK-NEXT:       %0 = arith.constant 27 : i32
# CHECK-NEXT:       %1 = arith.constant 9.300000e+00 : f32
# CHECK-NEXT:       %2 = arith.sitofp %0 : i32 to f32
# CHECK-NEXT:       %3 = arith.addf %2, %1 : f32
# CHECK-NEXT:       %a = memref.alloc() : memref<f32>
# CHECK-NEXT:       memref.store %3, %a[] : memref<f32>
# CHECK-NEXT:       %4 = arith.constant 45 : i32
# CHECK-NEXT:       %b = memref.alloc() : memref<i32>
# CHECK-NEXT:       memref.store %4, %b[] : memref<i32>
# CHECK-NEXT:       %c = memref.alloc() : memref<i32>
# CHECK-NEXT:       %d = memref.alloc() : memref<i1>
# CHECK-NEXT:       %5 = arith.constant 5 : i32
# CHECK-NEXT:       %6 = arith.constant 23 : i32
# CHECK-NEXT:       %7 = arith.constant 1 : i32
# CHECK-NEXT:       %i = memref.alloc() : memref<i32>
# CHECK-NEXT:       scf.for %8 = %5 to %6 step %7  : i32 {
# CHECK-NEXT:         memref.store %8, %i[] : memref<i32>
# CHECK-NEXT:         %9 = memref.load %b[] : memref<i32>
# CHECK-NEXT:         %10 = arith.constant 12 : i32
# CHECK-NEXT:         %11 = arith.cmpi ult, %9, %10 : i32
# CHECK-NEXT:         scf.if %11 {
# CHECK-NEXT:           %12 = arith.constant 9 : i32
# CHECK-NEXT:           %13 = arith.constant 17 : i32
# CHECK-NEXT:           %14 = arith.constant 1 : i32
# CHECK-NEXT:           %j = memref.alloc() : memref<i32>
# CHECK-NEXT:           scf.for %15 = %12 to %13 step %14  : i32 {
# CHECK-NEXT:             memref.store %15, %j[] : memref<i32>
# CHECK-NEXT:             %16 = memref.load %j[] : memref<i32>
# CHECK-NEXT:             %17 = arith.constant 1 : i32
# CHECK-NEXT:             %18 = arith.addi %16, %17 : i32
# CHECK-NEXT:             memref.store %18, %c[] : memref<i32>
# CHECK-NEXT:             %19 = arith.constant 4 : i32
# CHECK-NEXT:             %20 = arith.constant 20 : i32
# CHECK-NEXT:             %21 = arith.constant 1 : i32
# CHECK-NEXT:             %k = memref.alloc() : memref<i32>
# CHECK-NEXT:             scf.for %22 = %19 to %20 step %21  : i32 {
# CHECK-NEXT:               memref.store %22, %k[] : memref<i32>
# CHECK-NEXT:               %23 = memref.load %b[] : memref<i32>
# CHECK-NEXT:               %24 = memref.load %k[] : memref<i32>
# CHECK-NEXT:               %25 = "cb.pages_reservable_at_back"(%23, %24) : (i32, i32) -> i1
# CHECK-NEXT:               memref.store %25, %d[] : memref<i1>
# CHECK-NEXT:             }
# CHECK-NEXT:           }
# CHECK-NEXT:         } else {
# CHECK-NEXT:           %26 = memref.load %b[] : memref<i32>
# CHECK-NEXT:           %27 = arith.constant 35 : i32
# CHECK-NEXT:           %28 = arith.cmpi ugt, %26, %27 : i32
# CHECK-NEXT:           scf.if %28 {
# CHECK-NEXT:             printf.print_format "Hello world"
# CHECK-NEXT:           } else {
# CHECK-NEXT:             %29 = memref.load %b[] : memref<i32>
# CHECK-NEXT:             %30 = memref.load %b[] : memref<i32>
# CHECK-NEXT:             "cb.wait_front"(%29, %30) : (i32, i32) -> ()
# CHECK-NEXT:           }
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:       func.return
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
