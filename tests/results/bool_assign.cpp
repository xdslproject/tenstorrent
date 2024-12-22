void kernel_main() {
    bool a;
    a = true;
    bool a1;
    a1 = false;
    bool a2;
    a2 = a && a1;
    if (a2) {
        a = false;
    }
    std::int32_t a3;
    a3 = 5;
    bool a4;
    a4 = a3 < 10 || a2;
}
