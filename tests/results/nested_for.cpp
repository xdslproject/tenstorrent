void nested_for() {
    std::uint32_t a;
    std::uint32_t i;
    for (i = 3; i < 5; i += 1) {
        i = i;
        std::uint32_t i1;
        for (i1 = 7; i1 < 9; i1 += 1) {
            i1 = i1;
            a = 10;
        }
    }
}
