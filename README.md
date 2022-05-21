# HIP-Optimization-Bug
---

In this branch, we have tested this idea that maybe there is a reorder between multiple assembly statements in 
one function that cause the bug we have seen so far in O3 mode.

But the result was unexpected and we can see that in O3 mode of multiple statements in one function, the 
tests are passed but when we combine statements to one block, the tests are failed.

Please note that in the code we are using a flag to be able to test each one of the two cases as you can
see in the below code.( Note the `#if defined(MULTI_ASM)` line)
```c++
DEVICE inline uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
#if defined(MULTI_ASM)
  asm volatile("v_mul_lo_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
  asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(r): "v"(c): "vcc");
#else
    asm volatile("v_mul_lo_u32 %0, %1, %2;\r\n"
                 "v_addc_co_u32 %0, vcc, %0, %3, vcc;\r\n" 
                  : "+v"(r): "v"(a), "v"(b), "v"(c): "vcc");
#endif
    return r;
}
```

### Test multiple statements
For this test you should run the shell script `run_multi_statement.sh` in the directory.
```bash
$ ./run_multi_statement.sh
```

The result:

```



```

### Test for the other one(combine statements to one block)

Result for this call:
```
g1_add_test-> failed
g1_double_test-> failed
g1_add_mixed_test-> success
g2_add_test-> failed
g2_add_mixed_test-> failed
g2_double_test-> failed
```