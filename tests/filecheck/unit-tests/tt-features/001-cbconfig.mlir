// RUN: python3.13 tenstorrent/tools/tt-opt %s -t tt-metalium | filecheck %s

builtin.module {
  builtin.module attributes {kernel_type = "host", vis = "external"} {
    func.func @cb_configs() {
      %0 = arith.constant 1 : i32
      %1 = "tthost.create_cb_configuration"(%0, %0, %0) <{data_type = "float"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      %2 = "tthost.create_cb_configuration"(%0, %0, %0) <{data_type = "int"}> : (i32, i32, i32) -> !tthost.circular_buffer_config
      func.return
    }
  }
}

// CHECK: CircularBufferConfig cb_config_0 = CircularBufferConfig(1 * 1, {{[{][{]}}1, tt::DataFormat::Float32{{[}][}]}}).set_page_size(1, 1);
// CHECK: CircularBufferConfig cb_config_1 = CircularBufferConfig(1 * 1, {{[{][{]}}1, tt::DataFormat::Int32{{[}][}]}}).set_page_size(1, 1);