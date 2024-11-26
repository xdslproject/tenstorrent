void for_loop_use_i() {
    std::int32_t a;
    std::int32_t i;
    for (i = 3; i < 5; i += 1) {
        i = i;
        a = i;
    }
}
