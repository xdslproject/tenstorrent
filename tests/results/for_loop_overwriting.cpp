void for_loop_overwriting() {
    std::int32_t a;
    a = 0;
    std::int32_t i;
    for (i = 3; i < 5; i += 1) {
        i = i;
        a = i;
    }
}
