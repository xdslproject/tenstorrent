void kernel_main() {
    std::int32_t a;
    a = 27 + 3;
    bool d;
    std::int32_t i;
    for (i = 8; i < 20; i += 1) {
        i = i;
        d = cb_pages_reservable_at_back(a, i);
    }
}
