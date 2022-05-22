
Using [this branch](https://github.com/ramin-raeisi/HIP-Optimization-Bug/tree/vcc-correction), we have tested the idea that maybe there is a
reordering between multiple assembly statements in a function that
causes the bugs we have seen so far in O3 mode.

The result was unexpected, and we can see that even after combining
statements into one block, the tests still fail. We believe
that this is due to optimization changing the order of assembly statements.

In this code, a flag is used to allow us to test each of the two cases. 
These two cases are:

- Multiple assembly statements in one block(the default case)
```c++
  asm volatile("v_mul_lo_u32 %0, %1, %2" : "+v"(r): "v"(a), "v"(b));
  asm volatile("v_addc_co_u32 %0, vcc, %0, %1, vcc" : "+v"(r): "v"(c): "vcc");
```
- Merged assembly statements into one block(the case we are testing to stop the reorder):
```c++
asm volatile("v_mul_lo_u32 %0, %1, %2;\r\n"
             "v_addc_co_u32 %0, vcc, %0, %3, vcc;\r\n" 
              : "+v"(r): "v"(a), "v"(b), "v"(c): "vcc");
```
So as an example we can see the `madloc_cc` function.
(Note the `#if defined(MULTI_ASM)` in the code)
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

### 1.Test Multiple assembly statements
For this test you should run the shell script `run_multi_statement.sh` in the directory.
```bash
$ ./run_multi_statement.sh
```

The result:

```
g1_add_test-> failed
g1_double_test-> failed
g1_add_mixed_test-> success
g2_add_test-> failed
g2_add_mixed_test-> failed
g2_double_test-> failed
```

### 2.Test one block assembly statements
Run in the terminal:
```bash
$ ./run_single_statement.sh
```
Result for this call will be:
```
g1_add_test-> failed
g1_double_test-> failed
g1_add_mixed_test-> success
g2_add_test-> failed
g2_add_mixed_test-> failed
g2_double_test-> failed
```

# Compare with Cuda execution model:
Also, we checked how the cuda behaves with the carry flag when executing in parallel.
As you can see in this
[link](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions)
:
> The condition code register is not preserved 
across calls and is mainly intended for use in straight-line code 
sequences for computing extended-precision integer addition, subtraction,
and multiplication. 

Therefore, the carry flag, though it applies to each thread, will not be
preserved across calls. In hip execution, we have seen that the carry flag
is preserved across calls and we use a workaround to avoid this issue.

We need some clarification about some matters in order to make further progress, such as:
- Is Hip using the carry flag like the cuda is?
- In Cuda we have a method of adding two uint32 without any effect on the carry flag,
similar to the function V_ADD_U32 in hip that we founded in the MI100 ISA(
you can get access to it via [this link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjWnKzjlfH3AhXYSPEDHdKQAVMQFnoECAsQAQ&url=https%3A%2F%2Fdeveloper.amd.com%2Fwp-content%2Fresources%2FCDNA1_Shader_ISA_14December2020.pdf&usg=AOvVaw00eAcCsorzW_bWmk_OG_3y)
)however, when we use it, the 
compiler says that the instruction isn't supported on this architecture.
- Using an extra vcc as the carry flag seems to be a bad idea since we have
a huge gap between the cuda run-time and the hip, and it is worth to mention that
we were not able to test this idea since we had the below issue in `bellperson` and it
was in our previous report:
```
Caused by:
process didn't exit successfully: `/home/c9/ws_masoomi/bellperson/target/debug/build/bellperson-bef831d80a5c9934/build-script-build` (exit status: 101)
--- stdout
cargo:rustc-link-arg=-Wl,-rpath=/opt/rocm/lib/

--- stderr
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
error: ran out of registers during register allocation
fatal error: error in backend: no registers from class available to allocate
clang-15: error: clang frontend command failed with exit code 70 (use -v to see invocation)
clang version 15.0.0 (ssh://chfang@git.amd.com:29418/lightning/ec/llvm-project 5c271fb43e6e8030d659c6d48a003be01ddb50dd)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/c9/amd/llvm-050622/bin
clang-15: note: diagnostic msg:
  ********************

PLEASE ATTACH THE FOLLOWING FILES TO THE BUG REPORT:
Preprocessed source(s) and associated run script(s) are located at:
clang-15: note: diagnostic msg: /tmp/f9719c2d8e79c5c60c227b26ddb6400c8a2a103e4aebc494cbface265bd6f6c8-6d85d9/f9719c2d8e79c5c60c227b26ddb6400c8a2a103e4aebc494cbface265bd6f6c8-gfx908.cu
clang-15: note: diagnostic msg: /tmp/f9719c2d8e79c5c60c227b26ddb6400c8a2a103e4aebc494cbface265bd6f6c8-6d85d9/f9719c2d8e79c5c60c227b26ddb6400c8a2a103e4aebc494cbface265bd6f6c8-gfx908.sh
clang-15: note: diagnostic msg:

  ********************
thread 'main' panicked at 'hipcc failed. See the kernel source at /home/c9/ws_masoomi/bellperson/target/debug/build/bellperson-33c4cc1f0eea85ea/out/f9719c2d8e79c5c60c227b26ddb6400c8a2a103e4aebc494cbface265bd6f6c8.cxx', build.rs:159:13
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

