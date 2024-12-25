void kernel_main() {
    bool a;
    a = true;
    bool b;
    b = false;
    bool c;
    c = a && b;
    if (c) {
        a = false;
    }
    std::int32_t d;
    d = 5;
    bool e;
    e = d < 10 || c;
}
