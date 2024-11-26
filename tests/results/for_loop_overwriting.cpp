void for_loop_overwriting() {
    std::uint32_t a;
    a = 0;
    std::uint32_t i;
    for (i = 3; i < 5; i += 1) {
        i = i;
        a = i;
    }
}
