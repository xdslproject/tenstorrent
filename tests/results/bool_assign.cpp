void kernel_main() {
    bool a = true;
    bool b = false;
    bool c = a && b;
    if (c) {
        a = false;
    }
    std::int32_t d = 5;
    bool e = d < 10 || c;
}
