void kernel_main() {
    std::int32_t a = 5;
    std::int32_t b;
    if (a == 7) {
        a = 2;
        b = 3;
    } else {
        if (a == 5) {
            a = 1;
            b = 2;
        } else {
            a = 10;
            b = 9;
        }
    }
}
