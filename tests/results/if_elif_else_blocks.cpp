void if_elif_else_blocks() {
    std::uint32_t a;
    a = 5;
    std::uint32_t a1;
    if (a == 7) {
        a = 2;
        a1 = 3;
    } else {
        if (a == 5) {
            a = 1;
            a1 = 2;
        } else {
            a = 10;
            a1 = 9;
        }
    }
}
