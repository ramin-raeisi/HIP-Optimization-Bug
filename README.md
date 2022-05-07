# HIP-Optimization-Bug
---

The code comes from the Lotus project, which can be found at https://github.com/filecoin-project/lotus

LICENSE-MIT and LICENSE-APACHE are two of the lotus licenses.

The GPU code was extracted from the `bellperson` project, which is a `sub-project` of the Lotus project.

---
## Introduction to the problem
Having fixed all errors with the newly installed compiler on our server, we tested our **full source code** but encountered a new problem.
The problem can be seen in the following:

```
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
clang-15: note: diagnostic msg: /tmp/source-de1f58/source-gfx908.cu
clang-15: note: diagnostic msg: /tmp/source-de1f58/source-gfx908.sh
clang-15: note: diagnostic msg: 

********************
```

`source-gfx908.cu` and `source-gfx908.sh` are provided in the branch For your assessment( in report folder ).

We have also checked the source code to see if removing the `G1_bellman_multiexp` and `G2_bellman_multiexp` code will solve the problem.
The code compiles in about 2 seconds by commenting out these two functions.

With `G1_bellman_multiexp`, we can see that the code is compiled within a few minutes.
When we add `G2_bellman_multiexp`, however, we see that the code does not compile and after 40 minutes, the compiler crashes.

---
## does the previous compiler has the same problem?
No, the previous compiler does not have the same problem. We checked it 
by **not** using the below export:

    export HIP_CLANG_PATH=/home/c9/amd/llvm-050622/bin/
and the code compiles without any error.