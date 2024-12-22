void kernel_main() {
    std::int32_t a;
    a = 1;
    std::int32_t a1;
    a1 = 2;
    std::int32_t a2;
    a2 = 3;
    std::int32_t a3;
    a3 = 4;
    std::int32_t a4;
    a4 = 5;
    bool a5;
    a5 = false;
    bool a6;
    a6 = true;
    noc_async_write_multicast(a, a1, a2, a3, a5, a6, a4);
    noc_semaphore_set_multicast(a, a1, a2, a5, a6, a3);
    noc_async_write(a, a1, a2, a3);
    noc_async_read(a, a1, a2, a3);
    noc_semaphore_set(a, a1);
    noc_semaphore_wait(a, a1);
    noc_semaphore_inc(a, a1, a2);
    noc_async_read_barrier(a);
    noc_async_write_barrier(a);
    get_noc_addr_from_bank_id<true>(a, a1, 3);
}
