void if_elif() {
    std::uint32_t a;
    a = 5;
    if (a == 7) {
        a = 2;
    } else {
        if (a == 5) {
            a = 1;
        }
    }
}