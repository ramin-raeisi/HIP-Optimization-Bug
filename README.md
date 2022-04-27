# HIP-Optimization-Bug
---

The code comes from the Lotus project, which can be found at https://github.com/filecoin-project/lotus

LICENSE-MIT and LICENSE-APACHE are two of the lotus licenses.

---
The GPU code was extracted from the `bellperson` project, which is a `sub-project` of the Lotus project.
In this code, I've removed all unnecessary code in order to test the `g1_add` function in HIP/ROCm.
The GPU code can be found in `g1_add.cu` source.

The main problem is that when we use optimization flags like `-O3`, the result differs
significantly from when we use the `-O0` flag, This error will occur on two system with `gfx908` and `gfx900` gpus. The results are displayed below.
(the optimization flags can be found in Makefile)

I have also tested the code by cuda compiler and the result is the same with the `-O0` flag.

### note
Using the `[[clang::optnone]]` attribute for the `G1_add` function, we can see that the test results are correct even when compiled with O3 flag.
This is not ideal, but you can test it in the `optnone` branch.

## Test1(gfx908)
- gpu : gfx908
- os: ubuntu 20.04
- hip_version: 5.0.0(different versions tested, but always results in the same error)

### -O0 flag

    $ ./run.sh 
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O0 -I/opt/include/ -o g1_add g1_add.cu
    x: 695163763, 838428337, 867136025, 3916970060, 1083605276, 2882035772, 3603006931, 2269309842, 422274527, 1169772790, 1990394245, 416975321, 
    y: 1229022948, 3366429108, 670218974, 1658335027, 392632874, 1379067484, 798160530, 3656524164, 3793686573, 2144155088, 2721370348, 298035558, 
    z: 413203031, 3318893592, 1282426328, 1145762026, 1542369093, 485346739, 1679000480, 4026228341, 2371190916, 3558967189, 3094593878, 414846589,
### -O3 flag

    $ ./run.sh 
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O3 -I/opt/include/ -o g1_add g1_add.cu
    x: 2219282510, 760277030, 550604824, 1084982058, 1598340764, 4070768988, 58429962, 1029107186, 416340485, 2067845827, 3739158037, 1359784916, 
    y: 2110372024, 922872633, 3951822721, 2291919632, 395581724, 1245577012, 1515893760, 1633137805, 375254496, 1642521037, 2226945294, 2153458039, 
    z: 1774849172, 2702311364, 79269099, 3819870691, 506309, 3711088651, 2420102526, 2732203651, 134185010, 763995165, 1644979634, 372489431,

## Test2(gfx900)
- gpu : gfx900
- os: ubuntu 20.04
- hip_version: 5.0.0
### -O0 flag

    $ ./run.sh 
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O0 -I/opt/include/ --amdgpu-target=gfx900 -o g1_add g1_add.cu
    x: 695163763, 838428337, 867136025, 3916970060, 1083605276, 2882035772, 3603006931, 2269309842, 422274527, 1169772790, 1990394245, 416975321, 
    y: 1229022948, 3366429108, 670218974, 1658335027, 392632874, 1379067484, 798160530, 3656524164, 3793686573, 2144155088, 2721370348, 298035558, 
    z: 413203031, 3318893592, 1282426328, 1145762026, 1542369093, 485346739, 1679000480, 4026228341, 2371190916, 3558967189, 3094593878, 414846589,
### -O3 flag

    $ ./run.sh 
    rm -f g1_add *.o
    /opt/rocm/hip/bin/hipcc -std=c++14 -O3 -I/opt/include/ --amdgpu-target=gfx900 -o g1_add g1_add.cu
    x: 2929414586, 969768673, 2320176390, 299604284, 1872350046, 1842943764, 2580129601, 3246407952, 3262431993, 3573956922, 451610209, 160000686, 
    y: 1803897690, 2447850291, 4070635830, 3392081673, 3269585030, 1599467931, 3607413871, 1287758637, 1597882672, 3461576140, 1052015481, 3190781428, 
    z: 2716997885, 450547341, 2764350271, 4292641511, 3738188316, 2435605332, 3300623324, 1860722296, 426561640, 209336877, 1524356693, 114052808, 