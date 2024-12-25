void kernel_main() {
    std::int32_t a = 1;
    std::int32_t b = 2;
    std::int32_t c = 3;
    std::int32_t d = 4;
    std::int32_t e = 5;
    bool f = false;
    bool g = true;
    noc_async_write_multicast(a, b, c, d, f, g, e);
    noc_semaphore_set_multicast(a, b, c, f, g, d);
    noc_async_write(a, b, c, d);
    noc_async_read(a, b, c, d);
    noc_semaphore_set(a, b);
    noc_semaphore_wait(a, b);
    noc_semaphore_inc(a, b, c);
    noc_async_read_barrier(a);
    noc_async_write_barrier(a);
    get_noc_addr_from_bank_id<true>(a, b, c);
}
