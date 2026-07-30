**[README](../README.md)** » **Triton Distrubuted Tutorial User Guide**

# Installation

To run triton distributed examples, you need to install the following packages:
1. `RocSHMEM`: a parallel programming interface based on OpenSHMEM
2. `PyRocSHMEM`: a python bindings that expose RocSHMEM API to python API
3. `hip-python`: a hip-python API (required by PyRocSHMEM implementations)

For local build, please refer below installation guide for each package

## RocSHMEM

```bash
git clone http://42.228.13.241:10068/dcutoolkit/deeplearing/rocshmem.git
cd rocshmem
ROCSHMEM_INSTALL_PREFIX=/root/rocshmem bash compile.sh
cd build
make package
dpkg -i /workspace/rocshmem/build/rocshmem_3.5.0_amd64.deb
```

## PyRocSHMEM

```bash
git clone http://192.168.140.14:9080/ROCm/pyrocshmem.git
cd pyrocshmem
bash das-build.sh
python3.10 -m pip install --force-reinstall dist/*.whl
```

## hip-python

local build
```bash
git clone ssh://git@10.65.42.70:8022/ROCm/hip-python.git
cd hip-python
git checkout dev
./build.sh --hip --cuda --post-clean --no-venv -j <MAX_JOBS>

# hip-python and hip-python-as-cuda will be installed
```

# Run tutorials

Current we only support intra-node test, so keep `--nnodes=1`

```bash
cd python/tutorials/dist
torchrun --master-port=29501 --node_rank=0 --nproc_per_node=<specify number of GPUs you want to test> --nnodes=1 ./01-intra-node-gemm-rs-fused-sequential.py
```


# Tutorials

In the tutorials/dist, we have implemented different patterns of compute/communication overlap to accelerate Gemm-ReduceScatter(gemm-rs), Gemm-AllReduce(gemm-ar), Allgather-Gemm(ag-gemm), and AlltoALL (dispatch) cases.

The patterns are listed below:
1. Unfused Patterns (use more than 1 kernels)
    - Bulk-Synchronization
    - Producer-Consumer
2. Fused Patterns
    - Sequential: each workgroup compute one tile 
    - Producer-Consumer
        - Workgroup Specialization
        - Wave Specialization

For Producer-Consumer, it has 2 modes:
1. Push Mode: each rank push the data to peer rank once it is ready
2. Pull Mode: each rank only store the data locally and set the signal, the peer checks the signal and pulls the data when it needs

For Producer-Consumer

## 01-intra-node-gemm-rs-fused-sequential.py

This is Gemm-ReduceScatter using Fused Gemm-Scatter Patterns, Producer-Consumer, Push-Mode
The 2 kernels are not overlapped, however, the `kernel_gemm_rs_producer_fuse_scatter` directly push the data to the peer ranks using Regiters, which saved the time of data copy from Local memory to Global memory.

- kernel `kernel_gemm_rs_producer_fuse_scatter` pushes result to scatter buffer of each peer rank
- kernel `barrier_all_ipc` sync all the ranks to make sure the scatter result is ready for each rank
- kernel `kernel_consumer_reduce` does the local reduce

Performance for M=2048,N=3584,K=14336
```bash
# TFLOPS
triton #0 530.3981536254756
torch #0 563.3122264118182
```

## 02-intra-node-gemm-rs-producer-consumer.py

This is Gemm-ReduceScatter using Unfused Patterns, Producer-Consumer, Pull-Mode
The 2 kernels are overlapped

- kernel `kernel_gemm_rs_producer_persistent` stores the result to local scatter buffer and set signal (m_per_rank)
- kernel `copyp2p_kernel` and `add_continuous_kernel` waits the signal and do the p2p copy and add using RingReuce Algorithm

Performance for M=2048,N=3584,K=14336
```bash
# TFLOPS
triton #0 443.04878572716126
torch #0 563.4806961765307
```

## 03-intra-node-ag-gemm-producer-consumer.py

This is AllGather-Gemm using Unfused Patterns, Multi-Stream Producer-Consumer, Push-Mode

Performance for M=2048,N=3584,K=14336
```bash
# TFLOPS
triton #0 162.64591367269185
torch #0 224.3256896274446
```

## 04-intra-node-gemm-allreduce-fused-oneshot.py

This is Gemm-AllReduce using Fused Sequential Patterns, Using Only one Kernel

```bash
# TFLOPS
triton #0 250.0825652504355
torch #0 450.5031583838216
```

## 05-intra-node-gemm-allreduce-persist-ring.py

This is Gemm-AllReduce using Unfused Patterns, Multi-Stream with Persistent Reduce Kernel
This results shows that Persistent Reduce Kernel affect the Gemm Kernel Performance

Performance for M=2048,N=3584,K=14336
```bash
# ms
triton #0 1.2470881462097168
torch #0 0.46596808433532716
```

## 06-intra-node-gemm-allreduce-unfused-producer-consumer-ringreduce

This is Gemm-AllReduce using Unfused Patterns, Multi-Stream with Ring Reduce Kernel

Performance for M=2048,N=3584,K=14336
```bash
# us
triton #0 1.4593762397766112
torch #0 0.46665611267089846
```