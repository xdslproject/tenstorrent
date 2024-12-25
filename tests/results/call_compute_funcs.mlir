builtin.module attributes  {"kernel_type" = "compute"} {
  func.func @call_compute_funcs() {
    %0 = arith.constant 0 : i32
    %1 = memref.alloc() : memref<i32>
    memref.store %0, %1[] : memref<i32>
    %2 = arith.constant 1 : i32
    %3 = memref.alloc() : memref<i32>
    memref.store %2, %3[] : memref<i32>
    %4 = arith.constant 2 : i32
    %5 = memref.alloc() : memref<i32>
    memref.store %4, %5[] : memref<i32>
    %6 = arith.constant 3 : i32
    %7 = memref.alloc() : memref<i32>
    memref.store %6, %7[] : memref<i32>
    %8 = arith.constant 4 : i32
    %9 = memref.alloc() : memref<i32>
    memref.store %8, %9[] : memref<i32>
    %10 = arith.constant 5 : i32
    %11 = memref.alloc() : memref<i32>
    memref.store %10, %11[] : memref<i32>
    %12 = arith.constant 6 : i32
    %13 = memref.alloc() : memref<i32>
    memref.store %12, %13[] : memref<i32>
    %14 = arith.constant 7 : i32
    %15 = memref.alloc() : memref<i32>
    memref.store %14, %15[] : memref<i32>
    %16 = arith.constant 8 : i32
    %17 = memref.alloc() : memref<i32>
    memref.store %16, %17[] : memref<i32>
    %18 = arith.constant false
    %19 = memref.alloc() : memref<i1>
    memref.store %18, %19[] : memref<i1>
    %20 = arith.constant 9 : i32
    %21 = memref.alloc() : memref<i32>
    memref.store %20, %21[] : memref<i32>
    %22 = arith.constant 10 : i32
    %23 = memref.alloc() : memref<i32>
    memref.store %22, %23[] : memref<i32>
    %24 = arith.constant 11 : i32
    %25 = memref.alloc() : memref<i32>
    memref.store %24, %25[] : memref<i32>
    %26 = arith.constant 12 : i32
    %27 = memref.alloc() : memref<i32>
    memref.store %26, %27[] : memref<i32>
    %28 = arith.constant 13 : i32
    %29 = memref.alloc() : memref<i32>
    memref.store %28, %29[] : memref<i32>
    %30 = arith.constant 14 : i32
    %31 = memref.alloc() : memref<i32>
    memref.store %30, %31[] : memref<i32>
    %32 = arith.constant true
    %33 = memref.alloc() : memref<i1>
    memref.store %32, %33[] : memref<i1>
    %34 = arith.constant 15 : i32
    %35 = memref.alloc() : memref<i32>
    memref.store %34, %35[] : memref<i32>
    %36 = arith.constant 16 : i32
    %37 = memref.alloc() : memref<i32>
    memref.store %36, %37[] : memref<i32>
    %38 = arith.constant 17 : i32
    %39 = memref.alloc() : memref<i32>
    memref.store %38, %39[] : memref<i32>
    %40 = arith.constant 18 : i32
    %41 = memref.alloc() : memref<i32>
    memref.store %40, %41[] : memref<i32>
    %42 = arith.constant 19 : i32
    %43 = memref.alloc() : memref<i32>
    memref.store %42, %43[] : memref<i32>
    %44 = arith.constant 20 : i32
    %45 = memref.alloc() : memref<i32>
    memref.store %44, %45[] : memref<i32>
    %46 = arith.constant 21 : i32
    %47 = memref.alloc() : memref<i32>
    memref.store %46, %47[] : memref<i32>
    %48 = arith.constant 22 : i32
    %49 = memref.alloc() : memref<i32>
    memref.store %48, %49[] : memref<i32>
    %50 = arith.constant 23 : i32
    %51 = memref.alloc() : memref<i32>
    memref.store %50, %51[] : memref<i32>
    %52 = arith.constant 24 : i32
    %53 = memref.alloc() : memref<i32>
    memref.store %52, %53[] : memref<i32>
    %54 = arith.constant 25 : i32
    %55 = memref.alloc() : memref<i32>
    memref.store %54, %55[] : memref<i32>
    %56 = arith.constant 26 : i32
    %57 = memref.alloc() : memref<i32>
    memref.store %56, %57[] : memref<i32>
    %58 = memref.load %1[] : memref<i32>
    %59 = memref.load %3[] : memref<i32>
    %60 = memref.load %5[] : memref<i32>
    "comp.copy_tile"(%58, %59, %60) : (i32, i32, i32) -> ()
    %61 = memref.load %7[] : memref<i32>
    %62 = memref.load %9[] : memref<i32>
    %63 = memref.load %11[] : memref<i32>
    "comp.copy_tile_to_dst_init_short_with_dt"(%61, %62, %63) : (i32, i32, i32) -> ()
    %64 = memref.load %1[] : memref<i32>
    %65 = memref.load %11[] : memref<i32>
    "comp.copy_tile_to_dst_init_short"(%64, %65) : (i32, i32) -> ()
    "comp.copy_tile_init"() : () -> ()
    "comp.acquire_dst"() : () -> ()
    "comp.release_dst"() : () -> ()
    "comp.tile_regs_acquire"() : () -> ()
    "comp.tile_regs_wait"() : () -> ()
    "comp.tile_regs_commit"() : () -> ()
    "comp.tile_regs_release"() : () -> ()
    "comp.abs_tile_init"() : () -> ()
    %66 = memref.load %13[] : memref<i32>
    "comp.abs_tile"(%66) : (i32) -> ()
    "comp.add_tiles_init_nof"() : () -> ()
    %67 = memref.load %15[] : memref<i32>
    %68 = memref.load %17[] : memref<i32>
    %69 = memref.load %19[] : memref<i1>
    "comp.add_tiles_init"(%67, %68, %69) : (i32, i32, i1) -> ()
    %70 = memref.load %15[] : memref<i32>
    %71 = memref.load %17[] : memref<i32>
    %72 = memref.load %21[] : memref<i32>
    %73 = memref.load %23[] : memref<i32>
    %74 = memref.load %13[] : memref<i32>
    "comp.add_tiles"(%70, %71, %72, %73, %74) : (i32, i32, i32, i32, i32) -> ()
    "comp.sub_tiles_init_nof"() : () -> ()
    %75 = memref.load %15[] : memref<i32>
    %76 = memref.load %17[] : memref<i32>
    %77 = memref.load %19[] : memref<i1>
    "comp.sub_tiles_init"(%75, %76, %77) : (i32, i32, i1) -> ()
    %78 = memref.load %15[] : memref<i32>
    %79 = memref.load %17[] : memref<i32>
    %80 = memref.load %21[] : memref<i32>
    %81 = memref.load %23[] : memref<i32>
    %82 = memref.load %13[] : memref<i32>
    "comp.sub_tiles"(%78, %79, %80, %81, %82) : (i32, i32, i32, i32, i32) -> ()
    "comp.mul_tiles_init_f"() : () -> ()
    %83 = memref.load %15[] : memref<i32>
    %84 = memref.load %17[] : memref<i32>
    "comp.mul_tiles_init"(%83, %84) : (i32, i32) -> ()
    %85 = memref.load %15[] : memref<i32>
    %86 = memref.load %17[] : memref<i32>
    %87 = memref.load %21[] : memref<i32>
    %88 = memref.load %23[] : memref<i32>
    %89 = memref.load %13[] : memref<i32>
    "comp.mul_tiles"(%85, %86, %87, %88, %89) : (i32, i32, i32, i32, i32) -> ()
    %90 = memref.load %15[] : memref<i32>
    %91 = memref.load %17[] : memref<i32>
    "comp.add_bcast_cols_init_short"(%90, %91) : (i32, i32) -> ()
    %92 = memref.load %15[] : memref<i32>
    %93 = memref.load %17[] : memref<i32>
    "comp.add_bcast_rows_init_short"(%92, %93) : (i32, i32) -> ()
    %94 = memref.load %15[] : memref<i32>
    %95 = memref.load %17[] : memref<i32>
    %96 = memref.load %21[] : memref<i32>
    %97 = memref.load %23[] : memref<i32>
    %98 = memref.load %13[] : memref<i32>
    "comp.add_tiles_bcast"(%94, %95, %96, %97, %98) : (i32, i32, i32, i32, i32) -> ()
    %99 = memref.load %15[] : memref<i32>
    %100 = memref.load %17[] : memref<i32>
    "comp.sub_bcast_cols_init_short"(%99, %100) : (i32, i32) -> ()
    %101 = memref.load %15[] : memref<i32>
    %102 = memref.load %17[] : memref<i32>
    %103 = memref.load %21[] : memref<i32>
    %104 = memref.load %23[] : memref<i32>
    %105 = memref.load %13[] : memref<i32>
    "comp.sub_tiles_bcast"(%101, %102, %103, %104, %105) : (i32, i32, i32, i32, i32) -> ()
    %106 = memref.load %15[] : memref<i32>
    %107 = memref.load %17[] : memref<i32>
    "comp.mul_bcast_cols_init_short"(%106, %107) : (i32, i32) -> ()
    %108 = memref.load %15[] : memref<i32>
    %109 = memref.load %17[] : memref<i32>
    "comp.mul_bcast_rows_init_short"(%108, %109) : (i32, i32) -> ()
    %110 = memref.load %15[] : memref<i32>
    %111 = memref.load %17[] : memref<i32>
    %112 = memref.load %21[] : memref<i32>
    %113 = memref.load %23[] : memref<i32>
    %114 = memref.load %13[] : memref<i32>
    "comp.mul_tiles_bcast"(%110, %111, %112, %113, %114) : (i32, i32, i32, i32, i32) -> ()
    %115 = memref.load %15[] : memref<i32>
    %116 = memref.load %17[] : memref<i32>
    "comp.mul_tiles_bcast_scalar_init_short"(%115, %116) : (i32, i32) -> ()
    %117 = memref.load %15[] : memref<i32>
    %118 = memref.load %17[] : memref<i32>
    %119 = memref.load %21[] : memref<i32>
    %120 = memref.load %23[] : memref<i32>
    %121 = memref.load %13[] : memref<i32>
    "comp.mul_tiles_bcast_scalar"(%117, %118, %119, %120, %121) : (i32, i32, i32, i32, i32) -> ()
    %122 = memref.load %15[] : memref<i32>
    %123 = memref.load %17[] : memref<i32>
    %124 = memref.load %13[] : memref<i32>
    %125 = memref.load %11[] : memref<i32>
    "comp.mm_init"(%122, %123, %124, %125) : (i32, i32, i32, i32) -> ()
    %126 = memref.load %15[] : memref<i32>
    %127 = memref.load %17[] : memref<i32>
    %128 = memref.load %13[] : memref<i32>
    %129 = memref.load %11[] : memref<i32>
    "comp.mm_init_short_with_dt"(%126, %127, %128, %129) : (i32, i32, i32, i32) -> ()
    %130 = memref.load %15[] : memref<i32>
    %131 = memref.load %17[] : memref<i32>
    %132 = memref.load %13[] : memref<i32>
    "comp.mm_init_short"(%130, %131, %132) : (i32, i32, i32) -> ()
    %133 = memref.load %15[] : memref<i32>
    %134 = memref.load %17[] : memref<i32>
    %135 = memref.load %21[] : memref<i32>
    %136 = memref.load %23[] : memref<i32>
    %137 = memref.load %13[] : memref<i32>
    %138 = memref.load %11[] : memref<i32>
    "comp.matmul_tiles"(%133, %134, %135, %136, %137, %138) : (i32, i32, i32, i32, i32, i32) -> ()
    %139 = memref.load %15[] : memref<i32>
    %140 = memref.load %17[] : memref<i32>
    %141 = memref.load %13[] : memref<i32>
    %142 = memref.load %11[] : memref<i32>
    %143 = memref.load %25[] : memref<i32>
    %144 = memref.load %27[] : memref<i32>
    %145 = memref.load %29[] : memref<i32>
    "comp.mm_block_init"(%139, %140, %141, %142, %143, %144, %145) : (i32, i32, i32, i32, i32, i32, i32) -> ()
    %146 = memref.load %15[] : memref<i32>
    %147 = memref.load %17[] : memref<i32>
    %148 = memref.load %11[] : memref<i32>
    %149 = memref.load %25[] : memref<i32>
    %150 = memref.load %27[] : memref<i32>
    %151 = memref.load %29[] : memref<i32>
    "comp.mm_block_init_short"(%146, %147, %148, %149, %150, %151) : (i32, i32, i32, i32, i32, i32) -> ()
    %152 = memref.load %15[] : memref<i32>
    %153 = memref.load %17[] : memref<i32>
    %154 = memref.load %31[] : memref<i32>
    %155 = memref.load %25[] : memref<i32>
    %156 = memref.load %27[] : memref<i32>
    %157 = memref.load %29[] : memref<i32>
    "comp.mm_block_init_short_with_dt"(%152, %153, %154, %155, %156, %157) : (i32, i32, i32, i32, i32, i32) -> ()
    %158 = memref.load %15[] : memref<i32>
    %159 = memref.load %17[] : memref<i32>
    %160 = memref.load %21[] : memref<i32>
    %161 = memref.load %23[] : memref<i32>
    %162 = memref.load %13[] : memref<i32>
    %163 = memref.load %33[] : memref<i1>
    %164 = memref.load %25[] : memref<i32>
    %165 = memref.load %27[] : memref<i32>
    %166 = memref.load %29[] : memref<i32>
    "comp.matmul_block"(%158, %159, %160, %161, %162, %163, %164, %165, %166) : (i32, i32, i32, i32, i32, i1, i32, i32, i32) -> ()
    "comp.exp_tile_init"() <{"fast_and_approx" = false}> : () -> ()
    %167 = memref.load %13[] : memref<i32>
    "comp.exp_tile"(%167) <{"fast_and_approx" = true}> : (i32) -> ()
    "comp.exp2_tile_init"() : () -> ()
    %168 = memref.load %13[] : memref<i32>
    "comp.exp2_tile"(%168) : (i32) -> ()
    "comp.expm1_tile_init"() : () -> ()
    %169 = memref.load %13[] : memref<i32>
    "comp.expm1_tile"(%169) : (i32) -> ()
    "comp.relu_tile_init"() : () -> ()
    %170 = memref.load %13[] : memref<i32>
    "comp.relu_tile"(%170) : (i32) -> ()
    "comp.relu_max_tile_init"() : () -> ()
    %171 = memref.load %13[] : memref<i32>
    %172 = memref.load %35[] : memref<i32>
    "comp.relu_max_tile"(%171, %172) : (i32, i32) -> ()
    "comp.relu_min_tile_init"() : () -> ()
    %173 = memref.load %13[] : memref<i32>
    %174 = memref.load %37[] : memref<i32>
    "comp.relu_min_tile"(%173, %174) : (i32, i32) -> ()
    %175 = memref.load %13[] : memref<i32>
    %176 = memref.load %39[] : memref<i32>
    "comp.leaky_relu_tile_init"(%175, %176) : (i32, i32) -> ()
    "comp.elu_tile_init"() : () -> ()
    %177 = memref.load %13[] : memref<i32>
    %178 = memref.load %39[] : memref<i32>
    "comp.elu_tile"(%177, %178) : (i32, i32) -> ()
    "comp.erf_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %179 = memref.load %13[] : memref<i32>
    "comp.erf_tile"(%179) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.erfc_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %180 = memref.load %13[] : memref<i32>
    "comp.erfc_tile"(%180) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.erfinv_tile_init"() : () -> ()
    %181 = memref.load %13[] : memref<i32>
    "comp.erfinv_tile"(%181) : (i32) -> ()
    "comp.gelu_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %182 = memref.load %13[] : memref<i32>
    "comp.gelu_tile"(%182) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.heaviside_tile_init"() : () -> ()
    %183 = memref.load %41[] : memref<i32>
    "comp.heaviside_tile"(%183) : (i32) -> ()
    "comp.isinf_tile_init"() : () -> ()
    %184 = memref.load %13[] : memref<i32>
    "comp.isinf_tile"(%184) : (i32) -> ()
    "comp.isposinf_tile_init"() : () -> ()
    %185 = memref.load %13[] : memref<i32>
    "comp.isposinf_tile"(%185) : (i32) -> ()
    "comp.isneginf_tile_init"() : () -> ()
    %186 = memref.load %13[] : memref<i32>
    "comp.isneginf_tile"(%186) : (i32) -> ()
    "comp.isfinite_tile_init"() : () -> ()
    %187 = memref.load %13[] : memref<i32>
    "comp.isfinite_tile"(%187) : (i32) -> ()
    %188 = memref.load %13[] : memref<i32>
    "comp.isnan_tile"(%188) : (i32) -> ()
    "comp.i0_tile_init"() : () -> ()
    %189 = memref.load %13[] : memref<i32>
    "comp.i0_tile"(%189) : (i32) -> ()
    "comp.logical_not_unary_tile_init"() : () -> ()
    %190 = memref.load %13[] : memref<i32>
    "comp.logical_not_unary_tile"(%190) : (i32) -> ()
    "comp.recip_tile_init"() : () -> ()
    %191 = memref.load %13[] : memref<i32>
    "comp.recip_tile"(%191) : (i32) -> ()
    "comp.sign_tile_init"() : () -> ()
    %192 = memref.load %13[] : memref<i32>
    "comp.sign_tile"(%192) : (i32) -> ()
    "comp.sqrt_tile_init"() : () -> ()
    %193 = memref.load %13[] : memref<i32>
    "comp.sqrt_tile"(%193) : (i32) -> ()
    "comp.rsqrt_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %194 = memref.load %13[] : memref<i32>
    "comp.rsqrt_tile"(%194) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.sigmoid_tile_init"() : () -> ()
    %195 = memref.load %13[] : memref<i32>
    "comp.sigmoid_tile"(%195) : (i32) -> ()
    "comp.log_tile_init"() : () -> ()
    %196 = memref.load %13[] : memref<i32>
    "comp.log_tile"(%196) : (i32) -> ()
    "comp.log_with_base_tile_init"() : () -> ()
    %197 = memref.load %13[] : memref<i32>
    %198 = memref.load %43[] : memref<i32>
    "comp.log_with_base_tile"(%197, %198) : (i32, i32) -> ()
    "comp.power_tile_init"() : () -> ()
    %199 = memref.load %13[] : memref<i32>
    %200 = memref.load %45[] : memref<i32>
    "comp.power_tile"(%199, %200) : (i32, i32) -> ()
    "comp.rsub_tile_init"() : () -> ()
    %201 = memref.load %13[] : memref<i32>
    %202 = memref.load %41[] : memref<i32>
    "comp.rsub_tile"(%201, %202) : (i32, i32) -> ()
    "comp.signbit_tile_init"() : () -> ()
    %203 = memref.load %13[] : memref<i32>
    "comp.signbit_tile"(%203) : (i32) -> ()
    "comp.square_tile_init"() : () -> ()
    %204 = memref.load %13[] : memref<i32>
    "comp.square_tile"(%204) : (i32) -> ()
    %205 = memref.load %15[] : memref<i32>
    %206 = memref.load %17[] : memref<i32>
    %207 = memref.load %21[] : memref<i32>
    %208 = memref.load %23[] : memref<i32>
    %209 = memref.load %13[] : memref<i32>
    "comp.reduce_tile"(%205, %206, %207, %208, %209) : (i32, i32, i32, i32, i32) -> ()
    %210 = memref.load %49[] : memref<i32>
    %211 = memref.load %53[] : memref<i32>
    "comp.transpose_wh_tile_init"(%210, %211) : (i32, i32) -> ()
    %212 = memref.load %1[] : memref<i32>
    %213 = memref.load %47[] : memref<i32>
    %214 = memref.load %13[] : memref<i32>
    "comp.transpose_wh_tile"(%212, %213, %214) : (i32, i32, i32) -> ()
    "comp.tanh_tile_init"() : () -> ()
    %215 = memref.load %13[] : memref<i32>
    "comp.tanh_tile"(%215) : (i32) -> ()
    "comp.tan_tile_init"() : () -> ()
    %216 = memref.load %13[] : memref<i32>
    "comp.tan_tile"(%216) : (i32) -> ()
    "comp.sin_tile_init"() : () -> ()
    %217 = memref.load %13[] : memref<i32>
    "comp.sin_tile"(%217) : (i32) -> ()
    "comp.cos_tile_init"() : () -> ()
    %218 = memref.load %13[] : memref<i32>
    "comp.cos_tile"(%218) : (i32) -> ()
    "comp.asin_tile_init"() : () -> ()
    %219 = memref.load %13[] : memref<i32>
    "comp.asin_tile"(%219) : (i32) -> ()
    "comp.atan_tile_init"() : () -> ()
    %220 = memref.load %13[] : memref<i32>
    "comp.atan_tile"(%220) : (i32) -> ()
    "comp.acos_tile_init"() : () -> ()
    %221 = memref.load %13[] : memref<i32>
    "comp.acos_tile"(%221) : (i32) -> ()
    "comp.ltz_tile_init"() : () -> ()
    %222 = memref.load %13[] : memref<i32>
    "comp.ltz_tile"(%222) : (i32) -> ()
    "comp.eqz_tile_init"() : () -> ()
    %223 = memref.load %13[] : memref<i32>
    "comp.eqz_tile"(%223) : (i32) -> ()
    "comp.lez_tile_init"() : () -> ()
    %224 = memref.load %13[] : memref<i32>
    "comp.lez_tile"(%224) : (i32) -> ()
    "comp.gtz_tile_init"() : () -> ()
    %225 = memref.load %13[] : memref<i32>
    "comp.gtz_tile"(%225) : (i32) -> ()
    "comp.gez_tile_init"() : () -> ()
    %226 = memref.load %13[] : memref<i32>
    "comp.gez_tile"(%226) : (i32) -> ()
    "comp.nez_tile_init"() : () -> ()
    %227 = memref.load %13[] : memref<i32>
    "comp.nez_tile"(%227) : (i32) -> ()
    "comp.unary_ne_tile_init"() : () -> ()
    %228 = memref.load %13[] : memref<i32>
    %229 = memref.load %41[] : memref<i32>
    "comp.unary_ne_tile"(%228, %229) : (i32, i32) -> ()
    "comp.unary_gt_tile_init"() : () -> ()
    %230 = memref.load %13[] : memref<i32>
    %231 = memref.load %41[] : memref<i32>
    "comp.unary_gt_tile"(%230, %231) : (i32, i32) -> ()
    "comp.unary_lt_tile_init"() : () -> ()
    %232 = memref.load %13[] : memref<i32>
    %233 = memref.load %41[] : memref<i32>
    "comp.unary_lt_tile"(%232, %233) : (i32, i32) -> ()
    %234 = memref.load %49[] : memref<i32>
    %235 = memref.load %51[] : memref<i32>
    %236 = memref.load %53[] : memref<i32>
    "comp.tilize_init"(%234, %235, %236) : (i32, i32, i32) -> ()
    %237 = memref.load %49[] : memref<i32>
    %238 = memref.load %51[] : memref<i32>
    %239 = memref.load %53[] : memref<i32>
    "comp.tilize_init_short"(%237, %238, %239) : (i32, i32, i32) -> ()
    %240 = memref.load %55[] : memref<i32>
    %241 = memref.load %57[] : memref<i32>
    %242 = memref.load %51[] : memref<i32>
    %243 = memref.load %53[] : memref<i32>
    "comp.tilize_init_short_with_dt"(%240, %241, %242, %243) : (i32, i32, i32, i32) -> ()
    %244 = memref.load %49[] : memref<i32>
    %245 = memref.load %51[] : memref<i32>
    %246 = memref.load %53[] : memref<i32>
    "comp.tilize_block"(%244, %245, %246) : (i32, i32, i32) -> ()
    %247 = memref.load %49[] : memref<i32>
    %248 = memref.load %53[] : memref<i32>
    "comp.tilize_uninit"(%247, %248) : (i32, i32) -> ()
    %249 = memref.load %55[] : memref<i32>
    %250 = memref.load %57[] : memref<i32>
    %251 = memref.load %53[] : memref<i32>
    "comp.tilize_uninit_with_dt"(%249, %250, %251) : (i32, i32, i32) -> ()
    %252 = memref.load %49[] : memref<i32>
    %253 = memref.load %53[] : memref<i32>
    "comp.untilize_init"(%252, %253) : (i32, i32) -> ()
    %254 = memref.load %49[] : memref<i32>
    "comp.untilize_init_short"(%254) : (i32) -> ()
    %255 = memref.load %49[] : memref<i32>
    %256 = memref.load %51[] : memref<i32>
    %257 = memref.load %53[] : memref<i32>
    "comp.untilize_block"(%255, %256, %257) <{"n" = 27 : i32}> : (i32, i32, i32) -> ()
    %258 = memref.load %49[] : memref<i32>
    "comp.untilize_uninit"(%258) : (i32) -> ()
    func.return
  }
}
