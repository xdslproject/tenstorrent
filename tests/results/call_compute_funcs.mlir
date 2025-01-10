builtin.module attributes  {"kernel_type" = "compute"} {
  func.func @call_compute_funcs() {
    %0 = arith.constant 0 : i32
    %cb = memref.alloc() : memref<i32>
    memref.store %0, %cb[] : memref<i32>
    %1 = arith.constant 1 : i32
    %in_tile_index = memref.alloc() : memref<i32>
    memref.store %1, %in_tile_index[] : memref<i32>
    %2 = arith.constant 2 : i32
    %dst_tile_index = memref.alloc() : memref<i32>
    memref.store %2, %dst_tile_index[] : memref<i32>
    %3 = arith.constant 3 : i32
    %old_cb = memref.alloc() : memref<i32>
    memref.store %3, %old_cb[] : memref<i32>
    %4 = arith.constant 4 : i32
    %new_cb = memref.alloc() : memref<i32>
    memref.store %4, %new_cb[] : memref<i32>
    %5 = arith.constant 5 : i32
    %transpose = memref.alloc() : memref<i32>
    memref.store %5, %transpose[] : memref<i32>
    %6 = arith.constant 6 : i32
    %dst = memref.alloc() : memref<i32>
    memref.store %6, %dst[] : memref<i32>
    %7 = arith.constant 7 : i32
    %cb0 = memref.alloc() : memref<i32>
    memref.store %7, %cb0[] : memref<i32>
    %8 = arith.constant 8 : i32
    %cb1 = memref.alloc() : memref<i32>
    memref.store %8, %cb1[] : memref<i32>
    %9 = arith.constant false
    %acc_to_dest = memref.alloc() : memref<i1>
    memref.store %9, %acc_to_dest[] : memref<i1>
    %10 = arith.constant 9 : i32
    %tile0 = memref.alloc() : memref<i32>
    memref.store %10, %tile0[] : memref<i32>
    %11 = arith.constant 10 : i32
    %tile1 = memref.alloc() : memref<i32>
    memref.store %11, %tile1[] : memref<i32>
    %12 = arith.constant 11 : i32
    %out_cols = memref.alloc() : memref<i32>
    memref.store %12, %out_cols[] : memref<i32>
    %13 = arith.constant 12 : i32
    %out_rows = memref.alloc() : memref<i32>
    memref.store %13, %out_rows[] : memref<i32>
    %14 = arith.constant 13 : i32
    %kt_dim = memref.alloc() : memref<i32>
    memref.store %14, %kt_dim[] : memref<i32>
    %15 = arith.constant 14 : i32
    %prev_cb1 = memref.alloc() : memref<i32>
    memref.store %15, %prev_cb1[] : memref<i32>
    %16 = arith.constant true
    %transpose_bool = memref.alloc() : memref<i1>
    memref.store %16, %transpose_bool[] : memref<i1>
    %17 = arith.constant 15 : i32
    %upper_limit = memref.alloc() : memref<i32>
    memref.store %17, %upper_limit[] : memref<i32>
    %18 = arith.constant 16 : i32
    %lower_limit = memref.alloc() : memref<i32>
    memref.store %18, %lower_limit[] : memref<i32>
    %19 = arith.constant 17 : i32
    %slope = memref.alloc() : memref<i32>
    memref.store %19, %slope[] : memref<i32>
    %20 = arith.constant 18 : i32
    %param = memref.alloc() : memref<i32>
    memref.store %20, %param[] : memref<i32>
    %21 = arith.constant 19 : i32
    %log_base = memref.alloc() : memref<i32>
    memref.store %21, %log_base[] : memref<i32>
    %22 = arith.constant 20 : i32
    %power_ = memref.alloc() : memref<i32>
    memref.store %22, %power_[] : memref<i32>
    %23 = arith.constant 21 : i32
    %tile = memref.alloc() : memref<i32>
    memref.store %23, %tile[] : memref<i32>
    %24 = arith.constant 22 : i32
    %in_cb = memref.alloc() : memref<i32>
    memref.store %24, %in_cb[] : memref<i32>
    %25 = arith.constant 23 : i32
    %block = memref.alloc() : memref<i32>
    memref.store %25, %block[] : memref<i32>
    %26 = arith.constant 24 : i32
    %out_cb = memref.alloc() : memref<i32>
    memref.store %26, %out_cb[] : memref<i32>
    %27 = arith.constant 25 : i32
    %old_in_cb = memref.alloc() : memref<i32>
    memref.store %27, %old_in_cb[] : memref<i32>
    %28 = arith.constant 26 : i32
    %new_in_cb = memref.alloc() : memref<i32>
    memref.store %28, %new_in_cb[] : memref<i32>
    %29 = memref.load %cb[] : memref<i32>
    %30 = memref.load %in_tile_index[] : memref<i32>
    %31 = memref.load %dst_tile_index[] : memref<i32>
    "comp.copy_tile"(%29, %30, %31) : (i32, i32, i32) -> ()
    %32 = memref.load %old_cb[] : memref<i32>
    %33 = memref.load %new_cb[] : memref<i32>
    %34 = memref.load %transpose[] : memref<i32>
    "comp.copy_tile_to_dst_init_short_with_dt"(%32, %33, %34) : (i32, i32, i32) -> ()
    %35 = memref.load %cb[] : memref<i32>
    %36 = memref.load %transpose[] : memref<i32>
    "comp.copy_tile_to_dst_init_short"(%35, %36) : (i32, i32) -> ()
    "comp.copy_tile_init"() : () -> ()
    "comp.acquire_dst"() : () -> ()
    "comp.release_dst"() : () -> ()
    "comp.tile_regs_acquire"() : () -> ()
    "comp.tile_regs_wait"() : () -> ()
    "comp.tile_regs_commit"() : () -> ()
    "comp.tile_regs_release"() : () -> ()
    "comp.abs_tile_init"() : () -> ()
    %37 = memref.load %dst[] : memref<i32>
    "comp.abs_tile"(%37) : (i32) -> ()
    "comp.add_tiles_init_nof"() : () -> ()
    %38 = memref.load %cb0[] : memref<i32>
    %39 = memref.load %cb1[] : memref<i32>
    %40 = memref.load %acc_to_dest[] : memref<i1>
    "comp.add_tiles_init"(%38, %39, %40) : (i32, i32, i1) -> ()
    %41 = memref.load %cb0[] : memref<i32>
    %42 = memref.load %cb1[] : memref<i32>
    %43 = memref.load %tile0[] : memref<i32>
    %44 = memref.load %tile1[] : memref<i32>
    %45 = memref.load %dst[] : memref<i32>
    "comp.add_tiles"(%41, %42, %43, %44, %45) : (i32, i32, i32, i32, i32) -> ()
    "comp.sub_tiles_init_nof"() : () -> ()
    %46 = memref.load %cb0[] : memref<i32>
    %47 = memref.load %cb1[] : memref<i32>
    %48 = memref.load %acc_to_dest[] : memref<i1>
    "comp.sub_tiles_init"(%46, %47, %48) : (i32, i32, i1) -> ()
    %49 = memref.load %cb0[] : memref<i32>
    %50 = memref.load %cb1[] : memref<i32>
    %51 = memref.load %tile0[] : memref<i32>
    %52 = memref.load %tile1[] : memref<i32>
    %53 = memref.load %dst[] : memref<i32>
    "comp.sub_tiles"(%49, %50, %51, %52, %53) : (i32, i32, i32, i32, i32) -> ()
    "comp.mul_tiles_init_f"() : () -> ()
    %54 = memref.load %cb0[] : memref<i32>
    %55 = memref.load %cb1[] : memref<i32>
    "comp.mul_tiles_init"(%54, %55) : (i32, i32) -> ()
    %56 = memref.load %cb0[] : memref<i32>
    %57 = memref.load %cb1[] : memref<i32>
    %58 = memref.load %tile0[] : memref<i32>
    %59 = memref.load %tile1[] : memref<i32>
    %60 = memref.load %dst[] : memref<i32>
    "comp.mul_tiles"(%56, %57, %58, %59, %60) : (i32, i32, i32, i32, i32) -> ()
    %61 = memref.load %cb0[] : memref<i32>
    %62 = memref.load %cb1[] : memref<i32>
    "comp.add_bcast_cols_init_short"(%61, %62) : (i32, i32) -> ()
    %63 = memref.load %cb0[] : memref<i32>
    %64 = memref.load %cb1[] : memref<i32>
    "comp.add_bcast_rows_init_short"(%63, %64) : (i32, i32) -> ()
    %65 = memref.load %cb0[] : memref<i32>
    %66 = memref.load %cb1[] : memref<i32>
    %67 = memref.load %tile0[] : memref<i32>
    %68 = memref.load %tile1[] : memref<i32>
    %69 = memref.load %dst[] : memref<i32>
    "comp.add_tiles_bcast"(%65, %66, %67, %68, %69) : (i32, i32, i32, i32, i32) -> ()
    %70 = memref.load %cb0[] : memref<i32>
    %71 = memref.load %cb1[] : memref<i32>
    "comp.sub_bcast_cols_init_short"(%70, %71) : (i32, i32) -> ()
    %72 = memref.load %cb0[] : memref<i32>
    %73 = memref.load %cb1[] : memref<i32>
    %74 = memref.load %tile0[] : memref<i32>
    %75 = memref.load %tile1[] : memref<i32>
    %76 = memref.load %dst[] : memref<i32>
    "comp.sub_tiles_bcast"(%72, %73, %74, %75, %76) : (i32, i32, i32, i32, i32) -> ()
    %77 = memref.load %cb0[] : memref<i32>
    %78 = memref.load %cb1[] : memref<i32>
    "comp.mul_bcast_cols_init_short"(%77, %78) : (i32, i32) -> ()
    %79 = memref.load %cb0[] : memref<i32>
    %80 = memref.load %cb1[] : memref<i32>
    "comp.mul_bcast_rows_init_short"(%79, %80) : (i32, i32) -> ()
    %81 = memref.load %cb0[] : memref<i32>
    %82 = memref.load %cb1[] : memref<i32>
    %83 = memref.load %tile0[] : memref<i32>
    %84 = memref.load %tile1[] : memref<i32>
    %85 = memref.load %dst[] : memref<i32>
    "comp.mul_tiles_bcast"(%81, %82, %83, %84, %85) : (i32, i32, i32, i32, i32) -> ()
    %86 = memref.load %cb0[] : memref<i32>
    %87 = memref.load %cb1[] : memref<i32>
    "comp.mul_tiles_bcast_scalar_init_short"(%86, %87) : (i32, i32) -> ()
    %88 = memref.load %cb0[] : memref<i32>
    %89 = memref.load %cb1[] : memref<i32>
    %90 = memref.load %tile0[] : memref<i32>
    %91 = memref.load %tile1[] : memref<i32>
    %92 = memref.load %dst[] : memref<i32>
    "comp.mul_tiles_bcast_scalar"(%88, %89, %90, %91, %92) : (i32, i32, i32, i32, i32) -> ()
    %93 = memref.load %cb0[] : memref<i32>
    %94 = memref.load %cb1[] : memref<i32>
    %95 = memref.load %dst[] : memref<i32>
    %96 = memref.load %transpose[] : memref<i32>
    "comp.mm_init"(%93, %94, %95, %96) : (i32, i32, i32, i32) -> ()
    %97 = memref.load %cb0[] : memref<i32>
    %98 = memref.load %cb1[] : memref<i32>
    %99 = memref.load %dst[] : memref<i32>
    %100 = memref.load %transpose[] : memref<i32>
    "comp.mm_init_short_with_dt"(%97, %98, %99, %100) : (i32, i32, i32, i32) -> ()
    %101 = memref.load %cb0[] : memref<i32>
    %102 = memref.load %cb1[] : memref<i32>
    %103 = memref.load %dst[] : memref<i32>
    "comp.mm_init_short"(%101, %102, %103) : (i32, i32, i32) -> ()
    %104 = memref.load %cb0[] : memref<i32>
    %105 = memref.load %cb1[] : memref<i32>
    %106 = memref.load %tile0[] : memref<i32>
    %107 = memref.load %tile1[] : memref<i32>
    %108 = memref.load %dst[] : memref<i32>
    %109 = memref.load %transpose[] : memref<i32>
    "comp.matmul_tiles"(%104, %105, %106, %107, %108, %109) : (i32, i32, i32, i32, i32, i32) -> ()
    %110 = memref.load %cb0[] : memref<i32>
    %111 = memref.load %cb1[] : memref<i32>
    %112 = memref.load %dst[] : memref<i32>
    %113 = memref.load %transpose[] : memref<i32>
    %114 = memref.load %out_cols[] : memref<i32>
    %115 = memref.load %out_rows[] : memref<i32>
    %116 = memref.load %kt_dim[] : memref<i32>
    "comp.mm_block_init"(%110, %111, %112, %113, %114, %115, %116) : (i32, i32, i32, i32, i32, i32, i32) -> ()
    %117 = memref.load %cb0[] : memref<i32>
    %118 = memref.load %cb1[] : memref<i32>
    %119 = memref.load %transpose[] : memref<i32>
    %120 = memref.load %out_cols[] : memref<i32>
    %121 = memref.load %out_rows[] : memref<i32>
    %122 = memref.load %kt_dim[] : memref<i32>
    "comp.mm_block_init_short"(%117, %118, %119, %120, %121, %122) : (i32, i32, i32, i32, i32, i32) -> ()
    %123 = memref.load %cb0[] : memref<i32>
    %124 = memref.load %cb1[] : memref<i32>
    %125 = memref.load %prev_cb1[] : memref<i32>
    %126 = memref.load %out_cols[] : memref<i32>
    %127 = memref.load %out_rows[] : memref<i32>
    %128 = memref.load %kt_dim[] : memref<i32>
    "comp.mm_block_init_short_with_dt"(%123, %124, %125, %126, %127, %128) : (i32, i32, i32, i32, i32, i32) -> ()
    %129 = memref.load %cb0[] : memref<i32>
    %130 = memref.load %cb1[] : memref<i32>
    %131 = memref.load %tile0[] : memref<i32>
    %132 = memref.load %tile1[] : memref<i32>
    %133 = memref.load %dst[] : memref<i32>
    %134 = memref.load %transpose_bool[] : memref<i1>
    %135 = memref.load %out_cols[] : memref<i32>
    %136 = memref.load %out_rows[] : memref<i32>
    %137 = memref.load %kt_dim[] : memref<i32>
    "comp.matmul_block"(%129, %130, %131, %132, %133, %134, %135, %136, %137) : (i32, i32, i32, i32, i32, i1, i32, i32, i32) -> ()
    "comp.exp_tile_init"() <{"fast_and_approx" = false}> : () -> ()
    %138 = memref.load %dst[] : memref<i32>
    "comp.exp_tile"(%138) <{"fast_and_approx" = true}> : (i32) -> ()
    "comp.exp2_tile_init"() : () -> ()
    %139 = memref.load %dst[] : memref<i32>
    "comp.exp2_tile"(%139) : (i32) -> ()
    "comp.expm1_tile_init"() : () -> ()
    %140 = memref.load %dst[] : memref<i32>
    "comp.expm1_tile"(%140) : (i32) -> ()
    "comp.relu_tile_init"() : () -> ()
    %141 = memref.load %dst[] : memref<i32>
    "comp.relu_tile"(%141) : (i32) -> ()
    "comp.relu_max_tile_init"() : () -> ()
    %142 = memref.load %dst[] : memref<i32>
    %143 = memref.load %upper_limit[] : memref<i32>
    "comp.relu_max_tile"(%142, %143) : (i32, i32) -> ()
    "comp.relu_min_tile_init"() : () -> ()
    %144 = memref.load %dst[] : memref<i32>
    %145 = memref.load %lower_limit[] : memref<i32>
    "comp.relu_min_tile"(%144, %145) : (i32, i32) -> ()
    %146 = memref.load %dst[] : memref<i32>
    %147 = memref.load %slope[] : memref<i32>
    "comp.leaky_relu_tile_init"(%146, %147) : (i32, i32) -> ()
    "comp.elu_tile_init"() : () -> ()
    %148 = memref.load %dst[] : memref<i32>
    %149 = memref.load %slope[] : memref<i32>
    "comp.elu_tile"(%148, %149) : (i32, i32) -> ()
    "comp.erf_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %150 = memref.load %dst[] : memref<i32>
    "comp.erf_tile"(%150) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.erfc_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %151 = memref.load %dst[] : memref<i32>
    "comp.erfc_tile"(%151) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.erfinv_tile_init"() : () -> ()
    %152 = memref.load %dst[] : memref<i32>
    "comp.erfinv_tile"(%152) : (i32) -> ()
    "comp.gelu_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %153 = memref.load %dst[] : memref<i32>
    "comp.gelu_tile"(%153) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.heaviside_tile_init"() : () -> ()
    %154 = memref.load %param[] : memref<i32>
    "comp.heaviside_tile"(%154) : (i32) -> ()
    "comp.isinf_tile_init"() : () -> ()
    %155 = memref.load %dst[] : memref<i32>
    "comp.isinf_tile"(%155) : (i32) -> ()
    "comp.isposinf_tile_init"() : () -> ()
    %156 = memref.load %dst[] : memref<i32>
    "comp.isposinf_tile"(%156) : (i32) -> ()
    "comp.isneginf_tile_init"() : () -> ()
    %157 = memref.load %dst[] : memref<i32>
    "comp.isneginf_tile"(%157) : (i32) -> ()
    "comp.isfinite_tile_init"() : () -> ()
    %158 = memref.load %dst[] : memref<i32>
    "comp.isfinite_tile"(%158) : (i32) -> ()
    %159 = memref.load %dst[] : memref<i32>
    "comp.isnan_tile"(%159) : (i32) -> ()
    "comp.i0_tile_init"() : () -> ()
    %160 = memref.load %dst[] : memref<i32>
    "comp.i0_tile"(%160) : (i32) -> ()
    "comp.logical_not_unary_tile_init"() : () -> ()
    %161 = memref.load %dst[] : memref<i32>
    "comp.logical_not_unary_tile"(%161) : (i32) -> ()
    "comp.recip_tile_init"() : () -> ()
    %162 = memref.load %dst[] : memref<i32>
    "comp.recip_tile"(%162) : (i32) -> ()
    "comp.sign_tile_init"() : () -> ()
    %163 = memref.load %dst[] : memref<i32>
    "comp.sign_tile"(%163) : (i32) -> ()
    "comp.sqrt_tile_init"() : () -> ()
    %164 = memref.load %dst[] : memref<i32>
    "comp.sqrt_tile"(%164) : (i32) -> ()
    "comp.rsqrt_tile_init"() <{"fast_and_approx" = true}> : () -> ()
    %165 = memref.load %dst[] : memref<i32>
    "comp.rsqrt_tile"(%165) <{"fast_and_approx" = false}> : (i32) -> ()
    "comp.sigmoid_tile_init"() : () -> ()
    %166 = memref.load %dst[] : memref<i32>
    "comp.sigmoid_tile"(%166) : (i32) -> ()
    "comp.log_tile_init"() : () -> ()
    %167 = memref.load %dst[] : memref<i32>
    "comp.log_tile"(%167) : (i32) -> ()
    "comp.log_with_base_tile_init"() : () -> ()
    %168 = memref.load %dst[] : memref<i32>
    %169 = memref.load %log_base[] : memref<i32>
    "comp.log_with_base_tile"(%168, %169) : (i32, i32) -> ()
    "comp.power_tile_init"() : () -> ()
    %170 = memref.load %dst[] : memref<i32>
    %171 = memref.load %power_[] : memref<i32>
    "comp.power_tile"(%170, %171) : (i32, i32) -> ()
    "comp.rsub_tile_init"() : () -> ()
    %172 = memref.load %dst[] : memref<i32>
    %173 = memref.load %param[] : memref<i32>
    "comp.rsub_tile"(%172, %173) : (i32, i32) -> ()
    "comp.signbit_tile_init"() : () -> ()
    %174 = memref.load %dst[] : memref<i32>
    "comp.signbit_tile"(%174) : (i32) -> ()
    "comp.square_tile_init"() : () -> ()
    %175 = memref.load %dst[] : memref<i32>
    "comp.square_tile"(%175) : (i32) -> ()
    %176 = memref.load %cb0[] : memref<i32>
    %177 = memref.load %cb1[] : memref<i32>
    %178 = memref.load %tile0[] : memref<i32>
    %179 = memref.load %tile1[] : memref<i32>
    %180 = memref.load %dst[] : memref<i32>
    "comp.reduce_tile"(%176, %177, %178, %179, %180) : (i32, i32, i32, i32, i32) -> ()
    %181 = memref.load %in_cb[] : memref<i32>
    %182 = memref.load %out_cb[] : memref<i32>
    "comp.transpose_wh_tile_init"(%181, %182) : (i32, i32) -> ()
    %183 = memref.load %cb[] : memref<i32>
    %184 = memref.load %tile[] : memref<i32>
    %185 = memref.load %dst[] : memref<i32>
    "comp.transpose_wh_tile"(%183, %184, %185) : (i32, i32, i32) -> ()
    "comp.tanh_tile_init"() : () -> ()
    %186 = memref.load %dst[] : memref<i32>
    "comp.tanh_tile"(%186) : (i32) -> ()
    "comp.tan_tile_init"() : () -> ()
    %187 = memref.load %dst[] : memref<i32>
    "comp.tan_tile"(%187) : (i32) -> ()
    "comp.sin_tile_init"() : () -> ()
    %188 = memref.load %dst[] : memref<i32>
    "comp.sin_tile"(%188) : (i32) -> ()
    "comp.cos_tile_init"() : () -> ()
    %189 = memref.load %dst[] : memref<i32>
    "comp.cos_tile"(%189) : (i32) -> ()
    "comp.asin_tile_init"() : () -> ()
    %190 = memref.load %dst[] : memref<i32>
    "comp.asin_tile"(%190) : (i32) -> ()
    "comp.atan_tile_init"() : () -> ()
    %191 = memref.load %dst[] : memref<i32>
    "comp.atan_tile"(%191) : (i32) -> ()
    "comp.acos_tile_init"() : () -> ()
    %192 = memref.load %dst[] : memref<i32>
    "comp.acos_tile"(%192) : (i32) -> ()
    "comp.ltz_tile_init"() : () -> ()
    %193 = memref.load %dst[] : memref<i32>
    "comp.ltz_tile"(%193) : (i32) -> ()
    "comp.eqz_tile_init"() : () -> ()
    %194 = memref.load %dst[] : memref<i32>
    "comp.eqz_tile"(%194) : (i32) -> ()
    "comp.lez_tile_init"() : () -> ()
    %195 = memref.load %dst[] : memref<i32>
    "comp.lez_tile"(%195) : (i32) -> ()
    "comp.gtz_tile_init"() : () -> ()
    %196 = memref.load %dst[] : memref<i32>
    "comp.gtz_tile"(%196) : (i32) -> ()
    "comp.gez_tile_init"() : () -> ()
    %197 = memref.load %dst[] : memref<i32>
    "comp.gez_tile"(%197) : (i32) -> ()
    "comp.nez_tile_init"() : () -> ()
    %198 = memref.load %dst[] : memref<i32>
    "comp.nez_tile"(%198) : (i32) -> ()
    "comp.unary_ne_tile_init"() : () -> ()
    %199 = memref.load %dst[] : memref<i32>
    %200 = memref.load %param[] : memref<i32>
    "comp.unary_ne_tile"(%199, %200) : (i32, i32) -> ()
    "comp.unary_gt_tile_init"() : () -> ()
    %201 = memref.load %dst[] : memref<i32>
    %202 = memref.load %param[] : memref<i32>
    "comp.unary_gt_tile"(%201, %202) : (i32, i32) -> ()
    "comp.unary_lt_tile_init"() : () -> ()
    %203 = memref.load %dst[] : memref<i32>
    %204 = memref.load %param[] : memref<i32>
    "comp.unary_lt_tile"(%203, %204) : (i32, i32) -> ()
    %205 = memref.load %in_cb[] : memref<i32>
    %206 = memref.load %block[] : memref<i32>
    %207 = memref.load %out_cb[] : memref<i32>
    "comp.tilize_init"(%205, %206, %207) : (i32, i32, i32) -> ()
    %208 = memref.load %in_cb[] : memref<i32>
    %209 = memref.load %block[] : memref<i32>
    %210 = memref.load %out_cb[] : memref<i32>
    "comp.tilize_init_short"(%208, %209, %210) : (i32, i32, i32) -> ()
    %211 = memref.load %old_in_cb[] : memref<i32>
    %212 = memref.load %new_in_cb[] : memref<i32>
    %213 = memref.load %block[] : memref<i32>
    %214 = memref.load %out_cb[] : memref<i32>
    "comp.tilize_init_short_with_dt"(%211, %212, %213, %214) : (i32, i32, i32, i32) -> ()
    %215 = memref.load %in_cb[] : memref<i32>
    %216 = memref.load %block[] : memref<i32>
    %217 = memref.load %out_cb[] : memref<i32>
    "comp.tilize_block"(%215, %216, %217) : (i32, i32, i32) -> ()
    %218 = memref.load %in_cb[] : memref<i32>
    %219 = memref.load %out_cb[] : memref<i32>
    "comp.tilize_uninit"(%218, %219) : (i32, i32) -> ()
    %220 = memref.load %old_in_cb[] : memref<i32>
    %221 = memref.load %new_in_cb[] : memref<i32>
    %222 = memref.load %out_cb[] : memref<i32>
    "comp.tilize_uninit_with_dt"(%220, %221, %222) : (i32, i32, i32) -> ()
    %223 = memref.load %in_cb[] : memref<i32>
    %224 = memref.load %out_cb[] : memref<i32>
    "comp.untilize_init"(%223, %224) : (i32, i32) -> ()
    %225 = memref.load %in_cb[] : memref<i32>
    "comp.untilize_init_short"(%225) : (i32) -> ()
    %226 = memref.load %in_cb[] : memref<i32>
    %227 = memref.load %block[] : memref<i32>
    %228 = memref.load %out_cb[] : memref<i32>
    "comp.untilize_block"(%226, %227, %228) <{"n" = 27 : i32}> : (i32, i32, i32) -> ()
    %229 = memref.load %in_cb[] : memref<i32>
    "comp.untilize_uninit"(%229) : (i32) -> ()
    func.return
  }
}
