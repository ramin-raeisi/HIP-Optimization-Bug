# HIP-Optimization-Bug
---

The code comes from the Lotus project, which can be found at https://github.com/filecoin-project/lotus

LICENSE-MIT and LICENSE-APACHE are two of the lotus licenses.

---
The GPU code was extracted from the `bellperson` project, which is a `sub-project` of the Lotus project.
In this code, I've removed all unnecessary code in order to test the `g1_add` function in HIP/ROCm.
The GPU code can be found in `g1_add.cu` source.

## Intro
The [hip version code](https://github.com/kkHuang-amd/HIP-Optimization-Bug/tree/fix-O3-error) was added to the [more_tests_failed](https://github.com/ramin-raeisi/HIP-Optimization-Bug/tree/more_tests_failed) branch,
resulting in the [non_assembly_more_test](https://github.com/ramin-raeisi/HIP-Optimization-Bug/tree/non_assembly_more_tests) branch. 

In this branch, we can see that all tests pass, however, using a 
global variable(`__host__ __device__ uint32_t vcc = 0;`) for kernel will make the upper level tests in the
`bellperson` project fail, as in there, each function will call a
kernel simultaneously. Due to this problem, we need some kind of
protection over this global variable or use local variables for
that part(In this case, it's not applicable since we need to carry in or carry out from functions), that needs further assessment for performance issues,
which is one of our primary concern at the present time.

Below is a list of the tests that failed 
(the program is compiled with the O0 flag).

```
failures:
    domain::tests::gpu_fft3_consistency
    domain::tests::gpu_fft_consistency
    groth16::proof::test_with_bls12_381::serialization
    multiexp::gpu_multiexp_consistency
```

