# HIP-Optimization-Bug
---

The code comes from the Lotus project, which can be found at https://github.com/filecoin-project/lotus

LICENSE-MIT and LICENSE-APACHE are two of the lotus licenses.

---
The GPU code was extracted from the `bellperson` project, which is a `sub-project` of the Lotus project.
In this code, I've removed all unnecessary code in order to test the `g1_add` function in HIP/ROCm.
The GPU code can be found in `g1_add.cu` source.

The main problem is that when we use optimization flags like `-O3`, the result differs 
significantly from when we use the `-O0` flag, This error will occur on one system with `gfx908` gpu,
but it will not occur on another system with `gfx900`. The results are displayed below.
(the optimization flags can be found in Makefile)

## Test1(gfx908) :heavy_multiplication_x:
- gpu : gfx908
- os: ubuntu 20.04
- hip_version: 5.0.0(different versions tested, but always results in the same error)

### -O0 flag

    $ ./run.sh
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O0 -I/opt/include/ --amdgpu-target=gfx908 -o g1_add g1_add.cu
    4170345375, 2012830284, 1503711250, 570769679, 2836923000, 704975099, 1227040403, 3178475093, 275088504, 3980011365, 1748720341, 393178949,
    45540860, 2989434092, 3885112365, 372465729, 3694980533, 1270039660, 2171494759, 628615972, 3887561082, 2949947987, 2478314731, 3490317013,
    1068862307, 755154257, 1513188381, 1201411001, 2901736584, 1458952506, 1414772649, 1985525671, 4013457955, 2331417879, 1159900396, 3582112287,
### -O3 flag

    $ ./run.sh
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O3 -I/opt/include/ --amdgpu-target=gfx908 -o g1_add g1_add.cu
    3641899738, 1797061044, 1119247772, 3376253473, 2085976603, 683446392, 2884480335, 2395190299, 3319398775, 339766621, 2301350609, 1939640236,
    4027138783, 1688965632, 2662134587, 1586485807, 1792799139, 2899095012, 3678715337, 2198583921, 4028042761, 3984195621, 2991183175, 827875079,
    1068862307, 755154257, 1513188381, 1201411001, 2901736584, 1458952506, 1414772649, 1985525671, 4013457955, 2331417879, 1159900396, 3582112287,

## Test2(gfx900) :heavy_check_mark:
- gpu : gfx900
- os: ubuntu 20.04
- hip_version: 5.0.0
### -O0 flag
    $ ./run.sh
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O0 -I/opt/include/ --amdgpu-target=gfx900 -o g1_add g1_add.cu
    4170345375, 2012830284, 1503711250, 570769679, 2836923000, 704975099, 1227040403, 3178475093, 275088504, 3980011365, 1748720341, 393178949,
    45540860, 2989434092, 3885112365, 372465729, 3694980533, 1270039660, 2171494759, 628615972, 3887561082, 2949947987, 2478314731, 3490317013,
    1068862307, 755154257, 1513188381, 1201411001, 2901736584, 1458952506, 1414772649, 1985525671, 4013457955, 2331417879, 1159900396, 3582112287,
### -O3 flag
    $ ./run.sh
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O3 -I/opt/include/ --amdgpu-target=gfx900 -o g1_add g1_add.cu
    4170389065, 66804300, 4143501331, 3836559633, 3149272623, 1537436601, 1645805844, 4102362955, 2311989961, 1459804152, 4114320801, 3815590768,
    45540860, 2989434092, 3885112365, 372465729, 3694980533, 1270039660, 2171494759, 628615972, 3887561082, 2949947987, 2478314731, 3490317013,
    1068862307, 755154257, 1513188381, 1201411001, 2901736584, 1458952506, 1414772649, 1985525671, 4013457955, 2331417879, 1159900396, 3582112287, 