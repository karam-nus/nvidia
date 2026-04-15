---
title: "Chapter 2 — NVIDIA GPU Microarchitecture"
---

[← Back to Table of Contents](./README.md)

# Chapter 2: NVIDIA GPU Microarchitecture
## Deep Dive into Silicon

Understanding GPU microarchitecture is fundamental to writing high-performance CUDA code. While high-level abstractions make GPU programming accessible, knowing **what** happens at the hardware level, **why** design decisions were made, **which** components process your code, **when** bottlenecks occur, **who** (what unit) handles specific operations, and **where** data resides determines whether your kernel achieves 10% or 95% of theoretical peak performance.

This chapter dissects NVIDIA GPU architecture from the **Streaming Multiprocessor (SM)** — the fundamental compute unit — through the **memory hierarchy**, **execution model**, and **architectural evolution** from Pascal to Blackwell. We'll explore how hardware schedulers dispatch warps, how Tensor Cores accelerate matrix operations, why memory access patterns matter, and how modern features like Thread Block Clusters enable new programming paradigms.

---

## 1. Streaming Multiprocessor (SM) — The Heart of GPU Compute

### What is a Streaming Multiprocessor?

The **Streaming Multiprocessor (SM)** is the fundamental processing unit of an NVIDIA GPU. A modern GPU contains tens to hundreds of SMs, each capable of executing thousands of concurrent threads. Think of the SM as a SIMT (Single Instruction, Multiple Thread) processor that schedules and executes groups of 32 threads called **warps**.

<div class="diagram">
<div class="diagram-title">Streaming Multiprocessor Architecture (Ampere GA100)</div>
<div class="flow">
<div class="flow-node accent wide">
<strong>Streaming Multiprocessor (SM)</strong><br>
├─ <span class="green">4× Warp Schedulers</span> (dispatch 1 instruction/warp/cycle)<br>
├─ <span class="cyan">64× FP32 CUDA Cores</span> (single-precision floating-point)<br>
├─ <span class="cyan">32× FP64 CUDA Cores</span> (double-precision floating-point)<br>
├─ <span class="purple">64× INT32 Units</span> (integer arithmetic)<br>
├─ <span class="orange">4× Tensor Cores</span> (matrix multiply-accumulate)<br>
├─ <span class="teal">32× Load/Store Units (LD/ST)</span><br>
├─ <span class="pink">16× Special Function Units (SFU)</span> (sin, cos, exp, log)<br>
├─ <span class="yellow">192 KB Shared Memory / L1 Cache</span> (configurable split)<br>
├─ <span class="yellow">256 KB Register File</span> (65,536 × 32-bit registers)<br>
└─ <span class="green">Texture Units</span> (4× per SM)
</div>
</div>
</div>

### Why This Architecture?

**Throughput over Latency**: Unlike CPUs that optimize for single-thread performance with out-of-order execution, branch prediction, and large caches, GPUs maximize throughput by executing many simple threads in parallel. When one warp stalls on memory access, the scheduler immediately switches to another ready warp — hiding latency through thread-level parallelism.

**SIMT Efficiency**: By grouping 32 threads into warps that execute the same instruction, the GPU amortizes instruction fetch/decode costs across 32 execution units. This delivers massive compute density at lower power than independent thread execution.

### Internal Components Deep Dive

#### Warp Schedulers

Each SM contains **2-4 warp schedulers** (architecture dependent). Each scheduler:
- Selects an eligible warp from its pool each cycle
- Dispatches one instruction from that warp to execution units
- Manages warp state (active, stalled on memory, barrier, etc.)

**Volta/Ampere/Hopper**: 4 warp schedulers per SM
**Pascal/Turing**: 2-4 warp schedulers per SM (varies by product)

Modern schedulers support **independent thread scheduling** (Volta+), allowing threads within a warp to diverge and reconverge without lockstep execution constraints.

#### CUDA Cores

**CUDA Cores** are the FP32/FP64/INT32 arithmetic units:

- **FP32 Cores**: Single-precision floating-point (32-bit). Most compute kernels use FP32.
- **FP64 Cores**: Double-precision floating-point (64-bit). Scientific computing, simulations. Fewer units than FP32 (typically 1:2 or 1:32 ratio depending on product line).
- **INT32 Units**: 32-bit integer operations. Address calculations, loop counters, bitwise operations.

**Hopper Innovation**: 128 FP32 cores per SM (doubled from Ampere), enabling simultaneous FP32 and Tensor Core operations.

#### Load/Store Units (LD/ST)

Handle memory transactions between register file and memory hierarchy:
- Global memory loads/stores
- Shared memory loads/stores
- Constant cache loads
- Texture cache loads

**32 LD/ST units** per SM (typical) allow up to 32 memory operations per cycle. Coalesced memory accesses maximize bandwidth utilization.

#### Special Function Units (SFU)

Compute transcendental functions:
- Trigonometric: `sin`, `cos`, `tan`
- Exponential: `exp`, `exp2`, `log`, `log2`
- Reciprocals: `rsqrt`, `rcp`
- Others: `pow`, `atan`, `erf`

SFUs operate at **lower precision** (23 bits for FP32) but higher throughput than standard CUDA cores for these operations.

### Occupancy — Maximizing SM Utilization

**Occupancy** measures how effectively your kernel uses SM resources:

$$
\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}
$$

**Maximum Warps per SM**:
- Pascal/Volta/Turing: **64 warps**
- Ampere/Hopper: **64 warps**

**Maximum Thread Blocks per SM**:
- Pascal/Volta/Turing: **32 blocks**
- Ampere/Hopper: **32 blocks**

#### Occupancy Limiters

Three resources constrain occupancy:

1. **Registers**: Each SM has a fixed register file (65,536 × 32-bit registers). If your kernel uses 128 registers/thread with 256 threads/block:
   - Registers per block = $128 \times 256 = 32,768$
   - Blocks per SM = $\lfloor 65,536 / 32,768 \rfloor = 2$
   - Occupancy = $(2 \times 256) / (64 \times 32) = 512 / 2048 = 25\%$

2. **Shared Memory**: If your kernel uses 48 KB shared memory/block and SM has 96 KB total:
   - Blocks per SM = $\lfloor 96 / 48 \rfloor = 2$

3. **Thread Block Limit**: Maximum 32 blocks per SM regardless of other resources.

#### Occupancy Calculation Example

```cpp
// Kernel using 64 registers/thread, 32 KB shared memory/block
__global__ void matmul_kernel(float* C, float* A, float* B, int N) {
    __shared__ float As[32][32];  // 4 KB
    __shared__ float Bs[32][32];  // 4 KB (8 KB total)
    
    // Register usage: ~64 registers/thread (compiler-determined)
    // ...
}

// Launch configuration
dim3 block(256);  // 256 threads/block = 8 warps
dim3 grid((N + 15) / 16, (N + 15) / 16);

// Occupancy calculation (Ampere A100):
// - Registers: 256 threads × 64 regs = 16,384 regs/block
//   → 65,536 / 16,384 = 4 blocks/SM (register-limited)
// - Shared memory: 8 KB/block
//   → 164 KB / 8 KB = 20 blocks/SM
// - Block limit: 32 blocks/SM
// Limiting factor: registers → 4 blocks/SM
// Active warps: 4 blocks × 8 warps = 32 warps
// Occupancy: 32 / 64 = 50%
```

#### Occupancy vs Performance Trade-off

**Higher occupancy ≠ higher performance**. Consider:

- **Latency hiding**: 30-50% occupancy often suffices to hide memory latency.
- **Resource intensity**: Compute-bound kernels benefit less from high occupancy.
- **Cache utilization**: Fewer concurrent blocks may improve L1/L2 hit rates.

**When to optimize for higher occupancy**:
- Memory-bound kernels with irregular access patterns
- Kernels with high latency operations (global memory, atomics)

**When to accept lower occupancy**:
- Compute-bound kernels with high arithmetic intensity
- Kernels benefiting from larger shared memory working sets
- Register-heavy kernels where reducing registers hurts ILP (instruction-level parallelism)

<div class="diagram-grid cols-3">
<div class="diagram-card green">
<div class="card-icon">📊</div>
<div class="card-title">Compute-Bound</div>
<div class="card-desc">30-50% occupancy often optimal. Focus on ILP, instruction throughput, and minimizing divergence.</div>
</div>
<div class="diagram-card cyan">
<div class="card-icon">💾</div>
<div class="card-title">Memory-Bound</div>
<div class="card-desc">60-100% occupancy helps hide latency. Optimize memory coalescing and access patterns first.</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">⚖️</div>
<div class="card-title">Balanced</div>
<div class="card-desc">Profile to identify bottleneck. Occupancy is a tuning parameter, not a primary goal.</div>
</div>
</div>

---

## 2. Tensor Cores — Specialized Matrix Engines

### What Are Tensor Cores?

**Tensor Cores** are specialized hardware units for matrix multiply-accumulate (MMA) operations, designed to accelerate deep learning workloads. A single Tensor Core executes matrix multiplication on small tiles (4×4, 8×8, or 16×16 depending on precision) in a single operation.

Basic operation:
$$
\mathbf{D} = \mathbf{A} \times \mathbf{B} + \mathbf{C}
$$

Where $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $\mathbf{D}$ are small matrices (tiles). This replaces hundreds of scalar multiply-add instructions with a single hardware operation.

### Why Tensor Cores?

**Performance**: Tensor Cores deliver **8-20×** higher throughput than CUDA cores for matrix operations:
- **A100 Tensor Core Peak**: 312 TFLOPS (FP16)
- **A100 CUDA Core Peak**: 19.5 TFLOPS (FP32)

**Energy Efficiency**: Specialized hardware reduces power consumption per operation by 5-10× compared to general-purpose cores.

**AI/ML Dominance**: Training and inference for neural networks are dominated by matrix multiplications (GEMM operations). Tensor Cores directly target this workload.

### Tensor Core Evolution

<div class="timeline">
<div class="timeline-item">
<div class="timeline-year accent">2017 — Volta V100</div>
<div class="timeline-title green">1st Generation Tensor Cores</div>
<div class="timeline-desc">
<strong>8 Tensor Cores/SM</strong><br>
• FP16 input/output, FP32 accumulation<br>
• 4×4×4 matrix operation (D = A×B + C)<br>
• 125 TFLOPS (FP16) on V100<br>
• WMMA (Warp Matrix Multiply-Accumulate) API
</div>
</div>

<div class="timeline-item">
<div class="timeline-year accent">2018 — Turing TU102</div>
<div class="timeline-title cyan">2nd Generation Tensor Cores</div>
<div class="timeline-desc">
<strong>8 Tensor Cores/SM</strong><br>
• FP16, INT8, INT4, INT1 support<br>
• INT8: 254 TOPS (GeForce RTX 2080 Ti)<br>
• Enabled inference optimization<br>
• Still 4×4×4 matrix tiles
</div>
</div>

<div class="timeline-item">
<div class="timeline-year accent">2020 — Ampere A100</div>
<div class="timeline-title purple">3rd Generation Tensor Cores</div>
<div class="timeline-desc">
<strong>4 Tensor Cores/SM</strong><br>
• <strong>TF32 (TensorFloat-32)</strong>: FP32 range, FP16 precision (19-bit total)<br>
• BF16 (bfloat16): Google's format, better FP32 range than FP16<br>
• <strong>Structured Sparsity (2:4)</strong>: 2× throughput for sparse matrices<br>
• FP64 Tensor Cores: 19.5 TFLOPS (scientific computing)<br>
• 312 TFLOPS (FP16/BF16), 156 TFLOPS (TF32)<br>
• 8×8×4 matrix tiles (FP16/BF16)
</div>
</div>

<div class="timeline-item">
<div class="timeline-year accent">2022 — Hopper H100</div>
<div class="timeline-title orange">4th Generation Tensor Cores</div>
<div class="timeline-desc">
<strong>4 Tensor Cores/SM</strong><br>
• <strong>FP8 (E4M3, E5M2)</strong>: 8-bit floating-point formats<br>
• <strong>Transformer Engine</strong>: Automatic FP8/FP16 scaling<br>
• 989 TFLOPS (FP8), 494 TFLOPS (FP16)<br>
• 67 TFLOPS (FP64) — 3× A100 FP64 performance<br>
• <strong>Thread Block Clusters</strong> enable distributed shared memory<br>
• <strong>TMA (Tensor Memory Accelerator)</strong>: Asynchronous global→shared copies<br>
• <strong>DPX Instructions</strong>: Dynamic programming acceleration
</div>
</div>

<div class="timeline-item">
<div class="timeline-year accent">2024 — Blackwell B200</div>
<div class="timeline-title pink">5th Generation Tensor Cores</div>
<div class="timeline-desc">
<strong>4 Tensor Cores/SM</strong><br>
• <strong>FP4 (4-bit floating-point)</strong>: Ultra-low precision inference<br>
• 20 PFLOPS (FP4) — 20,000 TFLOPS<br>
• 10 PFLOPS (FP8), 2.5 PFLOPS (FP16)<br>
• <strong>2nd Gen Transformer Engine</strong>: Enhanced FP4/FP8 support<br>
• <strong>Secure AI</strong>: Confidential computing for AI workloads<br>
• 208 SMs (GB100), 128 FP32 cores/SM
</div>
</div>
</div>

### Precision Modes — Which Format for What?

<table class="table-comparison">
<thead>
<tr>
<th>Format</th>
<th>Bits</th>
<th>Range (Approx)</th>
<th>Precision</th>
<th>Use Case</th>
<th>Throughput vs FP16</th>
</tr>
</thead>
<tbody>
<tr class="accent">
<td><strong>FP64</strong></td>
<td>64</td>
<td>±1.8×10<sup>308</sup></td>
<td>~15 decimal digits</td>
<td>Scientific computing, simulations</td>
<td>0.0625× (1/16)</td>
</tr>
<tr>
<td><strong>FP32</strong></td>
<td>32</td>
<td>±3.4×10<sup>38</sup></td>
<td>~7 decimal digits</td>
<td>General compute, graphics</td>
<td>0.5× (1/2)</td>
</tr>
<tr class="green">
<td><strong>TF32</strong></td>
<td>19 (internal)</td>
<td>Same as FP32</td>
<td>~3 decimal digits</td>
<td>DL training (drop-in FP32 replacement)</td>
<td>4×</td>
</tr>
<tr>
<td><strong>BF16</strong></td>
<td>16</td>
<td>Same as FP32</td>
<td>~2 decimal digits</td>
<td>DL training (better range than FP16)</td>
<td>1×</td>
</tr>
<tr>
<td><strong>FP16</strong></td>
<td>16</td>
<td>±6.5×10<sup>4</sup></td>
<td>~3 decimal digits</td>
<td>DL training/inference</td>
<td>1× (baseline)</td>
</tr>
<tr class="cyan">
<td><strong>FP8 (E4M3)</strong></td>
<td>8</td>
<td>±448</td>
<td>~2 decimal digits</td>
<td>DL inference, some training</td>
<td>2×</td>
</tr>
<tr>
<td><strong>FP8 (E5M2)</strong></td>
<td>8</td>
<td>±57,344</td>
<td>~1 decimal digit</td>
<td>DL gradients (needs range)</td>
<td>2×</td>
</tr>
<tr class="orange">
<td><strong>INT8</strong></td>
<td>8</td>
<td>-128 to 127</td>
<td>Integer</td>
<td>DL inference (post-training quant)</td>
<td>2×</td>
</tr>
<tr class="pink">
<td><strong>FP4</strong></td>
<td>4</td>
<td>Limited</td>
<td>~1 decimal digit</td>
<td>DL inference (extreme compression)</td>
<td>4×</td>
</tr>
<tr>
<td><strong>INT4</strong></td>
<td>4</td>
<td>-8 to 7</td>
<td>Integer</td>
<td>DL inference (extreme quant)</td>
<td>4×</td>
</tr>
</tbody>
</table>

**TF32 Explained**: Introduced in Ampere, TensorFloat-32 uses:
- 1 sign bit
- 8 exponent bits (same as FP32, preserving range)
- 10 mantissa bits (reduced from FP32's 23 bits)

This gives FP32 range with reduced precision, enabling **automatic acceleration** of existing FP32 code without modifications. PyTorch and TensorFlow automatically use TF32 on Ampere+ GPUs.

**FP8 Formats**:
- **E4M3**: 1 sign + 4 exponent + 3 mantissa bits. Better precision, limited range. For activations/weights.
- **E5M2**: 1 sign + 5 exponent + 2 mantissa bits. Better range, less precision. For gradients.

### Programming Tensor Cores

#### WMMA API (CUDA C++)

Warp Matrix Multiply-Accumulate operations via `<mma.h>`:

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmma_matmul(half* A, half* B, float* C, int M, int N, int K) {
    // Declare fragments (register tiles)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);
    
    // Load tiles from global memory into fragments
    load_matrix_sync(a_frag, A + offset_a, N);
    load_matrix_sync(b_frag, B + offset_b, K);
    
    // Perform matrix multiply-accumulate: C = A × B + C
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result back to global memory
    store_matrix_sync(C + offset_c, c_frag, N, mem_row_major);
}
```

**Fragment**: A distributed view of matrix data across warp threads. Each thread holds a portion of the matrix in registers.

**Supported Shapes** (Ampere+):
- 16×16×16 (M×N×K)
- 8×8×4 (specific precisions)
- 16×8×8, 16×8×16, etc.

#### PTX MMA Instructions (Low-Level)

For maximum control, use PTX assembly:

```cpp
__device__ void mma_fp16(uint32_t* d, uint32_t* a, uint32_t* b, uint32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}
```

### Structured Sparsity (2:4 Sparsity)

**Ampere A100** introduced hardware support for **2:4 structured sparsity**: In every group of 4 values, exactly 2 are non-zero.

**Example sparse matrix**:
```
Dense:  [1.2, 0.0, 3.4, 0.0]
        [0.0, 2.1, 0.0, 4.5]

Sparse (2:4): [1.2, 3.4] + metadata [0b1010] (positions 0,2 are non-zero)
              [2.1, 4.5] + metadata [0b0101] (positions 1,3 are non-zero)
```

**Throughput Boost**: 2× Tensor Core throughput for sparse operations (e.g., 312 → 624 TFLOPS FP16 on A100).

**Use Cases**:
- Neural network pruning (remove 50% of weights with minimal accuracy loss)
- Attention mechanisms (sparse attention patterns)

**CUTLASS Library**: NVIDIA's CUTLASS provides efficient sparse GEMM templates with automatic metadata handling.

---

## 3. Memory Hierarchy — The Performance Pyramid

GPU performance is often limited by memory bandwidth rather than compute throughput. Understanding the memory hierarchy — where data lives, access latency, and bandwidth — is critical.

<div class="diagram">
<div class="diagram-title">GPU Memory Hierarchy (Latency & Bandwidth)</div>
<div class="flow">
<div class="flow-node green" style="width:20%; margin: 0 auto;">Registers<br>1 cycle, 128+ TB/s</div>
<div class="flow-arrow accent"></div>
<div class="flow-node cyan" style="width:40%; margin: 0 auto;">Shared Memory / L1<br>~20-30 cycles, 15-20 TB/s</div>
<div class="flow-arrow accent"></div>
<div class="flow-node purple" style="width:60%; margin: 0 auto;">L2 Cache<br>~200 cycles, 5-7 TB/s</div>
<div class="flow-arrow accent"></div>
<div class="flow-node orange" style="width:80%; margin: 0 auto;">HBM (Global Memory)<br>~400-600 cycles, 1.5-3.3 TB/s</div>
<div class="flow-arrow accent"></div>
<div class="flow-node pink" style="width:100%; margin: 0 auto;">Host Memory (PCIe/NVLink)<br>~10,000+ cycles, 32-900 GB/s</div>
</div>
</div>

### Memory Hierarchy Deep Dive Table

<table class="table-comparison">
<thead>
<tr>
<th>Level</th>
<th>Size (Hopper H100)</th>
<th>Latency (cycles)</th>
<th>Bandwidth</th>
<th>Scope</th>
<th>Programmer Control</th>
</tr>
</thead>
<tbody>
<tr class="green">
<td><strong>Registers</strong></td>
<td>256 KB/SM<br>(65,536 × 32-bit)</td>
<td>1</td>
<td>&gt;128 TB/s</td>
<td>Per-thread</td>
<td>Automatic (compiler-managed)</td>
</tr>
<tr class="cyan">
<td><strong>Shared Memory</strong></td>
<td>0-228 KB/SM<br>(configurable)</td>
<td>20-30</td>
<td>15-20 TB/s</td>
<td>Per-block</td>
<td>Explicit (__shared__)</td>
</tr>
<tr class="teal">
<td><strong>L1 Cache</strong></td>
<td>0-228 KB/SM<br>(shared with SMEM)</td>
<td>25-35</td>
<td>15-20 TB/s</td>
<td>Per-SM</td>
<td>Hints (load directives)</td>
</tr>
<tr class="purple">
<td><strong>L2 Cache</strong></td>
<td>60 MB (H100)<br>50 MB (H100 PCIe)</td>
<td>~200</td>
<td>5-7 TB/s</td>
<td>GPU-wide</td>
<td>Limited (cache policies)</td>
</tr>
<tr class="orange">
<td><strong>HBM</strong></td>
<td>80 GB (H100 SXM)<br>141 GB (H200)</td>
<td>400-600</td>
<td>3.35 TB/s (H100)<br>4.8 TB/s (H200)</td>
<td>GPU-wide</td>
<td>Allocation, access patterns</td>
</tr>
<tr class="pink">
<td><strong>Host Memory</strong></td>
<td>System RAM</td>
<td>10,000+</td>
<td>32 GB/s (PCIe 4.0 x16)<br>900 GB/s (NVLink 4.0)</td>
<td>Host-device</td>
<td>Explicit transfers</td>
</tr>
</tbody>
</table>

### 1. Registers — Fastest Storage

**What**: Private 32-bit storage per thread. Fastest memory tier.

**Size**: 65,536 registers × 32 bits = 256 KB per SM (Hopper).

**Latency**: 1 cycle (essentially zero latency).

**Bandwidth**: Theoretical peak &gt;128 TB/s (limited by ALU throughput, not memory).

**When to Use**: Automatic. Compiler allocates local variables to registers.

**Limitations**:
- **Register pressure**: Using too many registers reduces occupancy.
- **Spilling**: If kernel exceeds register limit, compiler "spills" to local memory (global memory, slow!).

**Check register usage**:
```bash
nvcc --ptxas-options=-v kernel.cu
# Output: ptxas info : Used 64 registers, 4096 bytes shared memory
```

**Optimization**: Reduce register usage via `-maxrregcount=N` or `__launch_bounds__`:

```cpp
// Limit to 64 registers/thread, minimum 8 blocks/SM
__global__ void __launch_bounds__(256, 8) my_kernel() {
    // Kernel code
}
```

### 2. Shared Memory — Programmable Cache

**What**: Fast, low-latency memory shared among all threads in a block. Explicitly managed by programmer via `__shared__` qualifier.

**Size (Configurable)**:
- **Pascal/Volta**: 96 KB max (48 KB default, configurable to 96 KB)
- **Turing**: 64 KB
- **Ampere GA100**: 164 KB max
- **Hopper GH100**: 228 KB max

**Latency**: ~20-30 cycles (30× faster than global memory).

**Bandwidth**: ~15-20 TB/s per SM.

**Configuration**: Shared memory and L1 cache share the same physical memory. Configure split:

```cpp
// Set 96 KB shared memory, 32 KB L1 (Hopper)
cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);

// Or globally
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);  // Prefer shared memory
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);      // Prefer L1 cache
cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);   // 50/50 split
```

#### Bank Conflicts

Shared memory is organized into **32 banks** (4-byte wide). Simultaneous accesses to the same bank (except broadcast) serialize, reducing effective bandwidth.

**What Causes Bank Conflicts**:
```cpp
__shared__ float data[32];

// CONFLICT: All threads in warp access same bank
// Thread 0 accesses data[0] → bank 0
// Thread 1 accesses data[1] → bank 1
// ...
// Thread 0 accesses data[32] → bank 0 (conflicts with data[0])
float value = data[threadIdx.x + 32 * warpId];
```

**Solution — Padding**:
```cpp
__shared__ float data[32 + 1];  // Extra element pads to 33, avoiding stride-32 conflicts
```

**No Conflict (Broadcast)**:
```cpp
// All threads access same element → broadcast (no conflict)
float value = data[0];
```

**No Conflict (Sequential)**:
```cpp
// Sequential access: thread i accesses element i
float value = data[threadIdx.x];  // Perfect pattern
```

**When to Worry**: Only for strided patterns with strides that are multiples of 32.

### 3. L1 Cache — Automatic Caching

**What**: Transparent cache for global/local memory accesses. Shares physical memory with shared memory.

**Size**: 0-228 KB/SM (architecture dependent, configurable split with shared memory).

**Latency**: ~25-35 cycles.

**Cache Line**: 128 bytes (32 × 4-byte words).

**Policy**: Varies by architecture. Volta+ have better L1 hit rates for global loads.

**Control**:
```cpp
// Prefer L1 cache over shared memory
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

// Load with caching hint (PTX)
asm("ld.global.ca.f32 %0, [%1];" : "=f"(value) : "l"(ptr));  // Cache all levels
asm("ld.global.cg.f32 %0, [%1];" : "=f"(value) : "l"(ptr));  // Cache global (L2 only)
```

### 4. L2 Cache — GPU-Wide Cache

**What**: Unified cache shared across all SMs.

**Size Evolution**:
- Pascal P100: 4 MB
- Volta V100: 6 MB
- Turing TU102: 6 MB
- Ampere A100: 40 MB
- Hopper H100 SXM: **60 MB**
- Hopper H100 PCIe: 50 MB

**Latency**: ~200 cycles.

**Bandwidth**: 5-7 TB/s.

**Persistence**: Ampere introduced **L2 cache residency controls**:

```cpp
// Set L2 persistence for a memory region
cudaMemAccessDesc accessDesc = {};
accessDesc.location.type = cudaMemLocationTypeDevice;
accessDesc.location.id = deviceId;
accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

cudaMemPool_t mempool;
cudaDeviceGetDefaultMemPool(&mempool, deviceId);
cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

// Allocate with L2 persistence
float* d_data;
cudaMallocAsync(&d_data, size, stream);
cudaMemAdvise(d_data, size, cudaMemAdviseSetAccessedBy, deviceId);
```

### 5. HBM (High Bandwidth Memory)

**What**: GPU main memory. High bandwidth, high latency.

**Evolution**:

| Generation | Bandwidth/Stack | Bandwidth (H100) | Capacity (H100) | Tech Node |
|------------|-----------------|------------------|-----------------|-----------|
| **HBM** | 128 GB/s | — | — | 2013 |
| **HBM2** | 256 GB/s | — | 16 GB (V100) | 2016 |
| **HBM2e** | 307-410 GB/s | 1.6-2.0 TB/s | 40 GB (A100) | 2018 |
| **HBM3** | 600-819 GB/s | **3.35 TB/s** | **80 GB** | 2022 |
| **HBM3e** | 1.15 TB/s | **4.8 TB/s** | **141 GB** (H200) | 2024 |

**Latency**: 400-600 cycles (vs L2: ~200 cycles).

**Bandwidth Calculation (H100 SXM)**:
- 5 HBM3 stacks × 1024-bit interface = 5120-bit bus width
- Clock: 5.2 Gbps (effective, with DDR)
- Bandwidth = $(5120 \text{ bits} \times 5.2 \text{ Gbps}) / 8 = 3328 \text{ GB/s} = 3.35 \text{ TB/s}$

**Optimizing HBM Access**:
1. **Coalescing**: Warp threads access contiguous memory → combine into fewer transactions.
2. **Alignment**: Align to 128-byte cache lines.
3. **Prefetching**: Asynchronous memory operations (Ampere+).
4. **Compression**: Use tensor cores with FP8/INT8 to reduce memory traffic.

### 6. Asynchronous Memory Operations (Ampere+)

**Asynchronous Copy (memcpy_async)**:

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void async_copy_kernel(float* dst, const float* src, int N) {
    __shared__ float smem[256];
    auto block = cooperative_groups::this_thread_block();
    
    // Create barrier for synchronization
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (threadIdx.x == 0) {
        init(&barrier, blockDim.x);
    }
    block.sync();
    
    // Asynchronous copy from global to shared memory
    cuda::memcpy_async(smem, src + blockIdx.x * 256, sizeof(float) * 256, barrier);
    
    // Wait for copy to complete
    barrier.arrive_and_wait();
    
    // Use data in shared memory
    dst[threadIdx.x] = smem[threadIdx.x] * 2.0f;
}
```

**Benefits**:
- Overlap copy with computation
- Hardware-accelerated (bypass L1 cache)
- Higher throughput for large transfers

---

## 4. Warp Execution Model — SIMT in Detail

### What is a Warp?

A **warp** is a group of 32 threads that execute the same instruction in lockstep (SIMT: Single Instruction, Multiple Thread). The warp is the fundamental scheduling unit.

**Why 32 Threads?**:
- Historical: Matches 32-bit memory transaction alignment
- Efficiency: Amortizes instruction fetch/decode across 32 ALUs
- Hardware: Simplifies scheduler logic

### SIMT Execution

All threads in a warp execute the same instruction at the same program counter. Each thread can operate on different data (SIMD: Single Instruction, Multiple Data).

```cpp
__global__ void vector_add(float* C, float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];  // All 32 threads execute this add simultaneously
    }
}
```

**Instruction Issue**: Warp scheduler issues one instruction to all 32 threads per cycle.

### Warp Divergence — The Performance Killer

**What**: When threads in a warp take different execution paths (e.g., different sides of an if/else branch), the warp serializes execution.

**Example**:
```cpp
__global__ void divergent_kernel(float* data, int N) {
    int i = threadIdx.x;
    
    if (i % 2 == 0) {
        // Path A: Even threads
        data[i] = expensive_computation_A(data[i]);  // 100 instructions
    } else {
        // Path B: Odd threads
        data[i] = expensive_computation_B(data[i]);  // 100 instructions
    }
}
```

**Execution**:
1. Warp scheduler evaluates condition for all 32 threads
2. **First pass**: Execute Path A for threads 0, 2, 4, ..., 30 (threads 1, 3, 5, ..., 31 are **masked/disabled**)
3. **Second pass**: Execute Path B for threads 1, 3, 5, ..., 31 (threads 0, 2, 4, ..., 30 are masked)

**Cost**: $100 + 100 = 200$ instructions instead of potential 100 if all threads took the same path.

#### Predication — Hardware Optimization

For **short branches**, the compiler uses **predication** instead of branching:

```cpp
// Original code
if (condition) {
    y = a * x + b;
} else {
    y = c * x + d;
}

// Compiled (predicated)
float y1 = a * x + b;
float y2 = c * x + d;
y = condition ? y1 : y2;  // Select based on predicate
```

Both paths execute, but results are conditionally stored. Efficient for ≤4-8 instructions per branch.

#### Independent Thread Scheduling (Volta+)

**Pre-Volta**: Warps executed in lockstep. Threads couldn't diverge and reconverge flexibly.

**Volta+ (Compute Capability 7.0+)**: Threads can diverge and reconverge independently. Hardware maintains a **program counter per thread** (not just per warp).

**Benefits**:
- Fine-grained synchronization primitives
- More efficient intra-warp communication
- Better performance for complex control flow

**Example** (Volta+):
```cpp
__global__ void independent_scheduling_example() {
    int lane = threadIdx.x % 32;
    
    // Threads can now safely diverge and sync independently
    if (lane < 16) {
        __syncwarp(0xFFFF);  // Sync only first 16 threads
        // ...
    } else {
        __syncwarp(0xFFFF0000);  // Sync only last 16 threads
        // ...
    }
}
```

**Warning**: Independent scheduling can introduce deadlocks if not careful. Always use `__syncwarp()` for warp-level synchronization (not implicit).

### Thread Block to Warp Mapping

**Thread Block**: User-defined group of threads (e.g., 256 threads).

**Warp Division**: Block divided into warps of 32 threads in **row-major order** (threadIdx.x varies fastest).

**Example**: `dim3 block(64, 4, 1)` = 256 threads
- Warp 0: threadIdx.x ∈ [0, 31], threadIdx.y = 0, threadIdx.z = 0
- Warp 1: threadIdx.x ∈ [32, 63], threadIdx.y = 0, threadIdx.z = 0
- Warp 2: threadIdx.x ∈ [0, 31], threadIdx.y = 1, threadIdx.z = 0
- ...

**Best Practice**: Make blockDim.x a multiple of 32 to avoid **partial warps** (warps with &lt;32 active threads waste resources).

### Performance Impact of Divergence

**Benchmark** (A100):

| Kernel Type | Time (μs) | Speedup | Divergence |
|-------------|-----------|---------|------------|
| **Fully convergent** | 45 | 1.0× | None |
| **50% divergence** (half threads take each path) | 78 | 0.58× | High |
| **Fine-grained divergence** (adjacent threads diverge) | 82 | 0.55× | Extreme |
| **Predicated (short branches)** | 48 | 0.94× | Minimal |

**Mitigation Strategies**:
1. **Refactor**: Make all threads in a warp take the same path.
2. **Thread specialization**: Assign different warps to different tasks.
3. **Data reordering**: Sort data so similar items are processed by the same warp.
4. **Early exit**: Use `__syncthreads()` to exit entire blocks early.

```cpp
// BAD: Fine-grained divergence
if (data[i] > threshold) {
    process_high(data[i]);
} else {
    process_low(data[i]);
}

// GOOD: Partition data, process in separate warps
__shared__ float high_queue[256];
__shared__ float low_queue[256];
__shared__ int high_count, low_count;

if (threadIdx.x == 0) { high_count = low_count = 0; }
__syncthreads();

// Partition
if (data[i] > threshold) {
    int pos = atomicAdd(&high_count, 1);
    high_queue[pos] = data[i];
} else {
    int pos = atomicAdd(&low_count, 1);
    low_queue[pos] = data[i];
}
__syncthreads();

// Process partitions (warps converge within each partition)
if (threadIdx.x < high_count) {
    process_high(high_queue[threadIdx.x]);
} else if (threadIdx.x - high_count < low_count) {
    process_low(low_queue[threadIdx.x - high_count]);
}
```

---

## 5. SM Evolution Across Architectures

NVIDIA GPUs have evolved rapidly, with each generation introducing new features, improved throughput, and expanded capabilities. This section details the SM architecture for each generation from Pascal to Blackwell.

<div class="diagram">
<div class="diagram-title">SM Evolution Timeline (2016-2024)</div>
<div class="flow">
<div class="flow-node green wide">Pascal GP100 (2016)<br>SM 6.0 • 64 FP32 • 32 FP64</div>
<div class="flow-arrow accent"></div>
<div class="flow-node cyan wide">Volta GV100 (2017)<br>SM 7.0 • 64 FP32 • 32 FP64 • 8 Tensor Cores (Gen 1)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node purple wide">Turing TU102 (2018)<br>SM 7.5 • 64 FP32 • RT Cores • 8 Tensor Cores (Gen 2)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node orange wide">Ampere GA100 (2020)<br>SM 8.0 • 64 FP32 • 32 FP64 • 4 Tensor Cores (Gen 3) • Sparsity</div>
<div class="flow-arrow accent"></div>
<div class="flow-node pink wide">Hopper GH100 (2022)<br>SM 9.0 • 128 FP32 • 4 Tensor Cores (Gen 4) • FP8 • TMA</div>
<div class="flow-arrow accent"></div>
<div class="flow-node teal wide">Blackwell GB100 (2024)<br>SM 10.0 • 128 FP32 • 4 Tensor Cores (Gen 5) • FP4</div>
</div>
</div>

### Pascal GP100 (2016) — Compute Capability 6.0

**Product**: Tesla P100

**Manufacturing**: TSMC 16nm FinFET

**SM Count**: 56 SMs (full GP100), 60 SMs (GP100 die)

**SM Architecture**:
- **64 FP32 CUDA cores** per SM
- **32 FP64 CUDA cores** per SM (1:2 ratio, excellent for scientific computing)
- **64 INT32 units**
- **2 warp schedulers** (can issue 2 independent warps/cycle)
- **32 LD/ST units**
- **16 SFUs**
- **64 KB shared memory** (max, configurable with L1)
- **256 KB register file** (65,536 × 32-bit registers)

**Key Features**:
- **HBM2 memory**: 732 GB/s bandwidth (16 GB)
- **NVLink 1.0**: 160 GB/s bidirectional
- **Unified memory** improvements
- **Preemption**: Task-level and instruction-level

**Performance** (FP64-optimized):
- FP64: 5.3 TFLOPS
- FP32: 10.6 TFLOPS

**Why It Mattered**: First HBM2 GPU. Exceptional FP64 performance for scientific computing (weather, molecular dynamics, computational fluid dynamics).

### Volta GV100 (2017) — Compute Capability 7.0

**Product**: Tesla V100

**Manufacturing**: TSMC 12nm FFN

**SM Count**: 80 SMs (full GV100), 84 SMs (GV100 die)

**SM Architecture**:
- **64 FP32 CUDA cores** per SM
- **32 FP64 CUDA cores** per SM
- **64 INT32 units** (can execute concurrently with FP32)
- **8 Tensor Cores (1st generation)** — FP16 input, FP32 accumulate
- **4 warp schedulers** (2× Pascal)
- **32 LD/ST units**
- **16 SFUs**
- **96 KB shared memory** (configurable)
- **256 KB register file**

**Key Innovations**:
- **Tensor Cores**: 125 TFLOPS (FP16) — 10× improvement for deep learning
- **Independent Thread Scheduling**: Program counter per thread, not per warp
- **Enhanced L1 cache**: Unified data cache
- **HBM2**: 900 GB/s (16 GB or 32 GB)
- **NVLink 2.0**: 300 GB/s bidirectional

**Performance**:
- FP64: 7.8 TFLOPS
- FP32: 15.7 TFLOPS
- **Tensor (FP16)**: 125 TFLOPS

**Impact**: Revolutionized deep learning training. ResNet-50 training: 2.5× faster than Pascal P100. Enabled large language models (BERT, GPT-2).

### Turing TU102 (2018) — Compute Capability 7.5

**Products**: GeForce RTX 2080 Ti, Quadro RTX 6000, Tesla T4

**Manufacturing**: TSMC 12nm FFN

**SM Count**: 72 SMs (RTX 2080 Ti), 68 SMs (TU102 full die)

**SM Architecture**:
- **64 FP32 CUDA cores** per SM
- **64 INT32 units** (concurrent execution)
- **8 Tensor Cores (2nd generation)** — FP16, INT8, INT4 support
- **1 RT Core** per SM (ray tracing acceleration)
- **4 warp schedulers**
- **32 LD/ST units**
- **16 SFUs**
- **64 KB shared memory** (configurable)
- **256 KB register file**

**Key Innovations**:
- **RT Cores**: Hardware ray tracing (bounding volume hierarchy traversal, ray-triangle intersection)
- **INT8/INT4 Tensor Cores**: Inference optimization (254 TOPS INT8)
- **Concurrent FP32 + INT32**: Independent datapaths
- **GDDR6 memory**: 616 GB/s (RTX 2080 Ti)

**Performance** (RTX 2080 Ti):
- FP32: 13.4 TFLOPS
- Tensor (FP16): 107 TFLOPS
- Tensor (INT8): 214 TOPS

**Market**: First consumer GPUs with ray tracing. Gaming + inference focus.

### Ampere GA100 (2020) — Compute Capability 8.0

**Product**: A100 (SXM, PCIe), A30, A10

**Manufacturing**: TSMC 7nm

**SM Count**: 108 SMs (full GA100), 128 SMs (GA100 die)

**SM Architecture**:
- **64 FP32 CUDA cores** per SM
- **32 FP64 CUDA cores** per SM (data center SKU)
- **64 INT32 units**
- **4 Tensor Cores (3rd generation)** — TF32, BF16, FP64, INT8, sparsity
- **4 warp schedulers**
- **32 LD/ST units**
- **16 SFUs**
- **164 KB shared memory** (up to 164 KB)
- **256 KB register file**

**Key Innovations**:
- **TensorFloat-32 (TF32)**: Automatic FP32 acceleration (up to 10× vs V100)
- **BF16 support**: Better range than FP16 for training
- **Structured sparsity (2:4)**: 2× Tensor Core throughput for sparse models
- **FP64 Tensor Cores**: 19.5 TFLOPS (scientific computing)
- **Asynchronous copy** (`memcpy_async`): Overlap data movement with compute
- **Multi-Instance GPU (MIG)**: Partition single GPU into up to 7 instances
- **HBM2e**: 1.6 TB/s (40 GB) or 2.0 TB/s (80 GB)
- **NVLink 3.0**: 600 GB/s bidirectional

**Performance** (A100 80GB):
- FP64: 9.7 TFLOPS (19.5 with Tensor Cores)
- FP32: 19.5 TFLOPS
- TF32: 156 TFLOPS
- **Tensor (FP16/BF16)**: 312 TFLOPS
- **Tensor (INT8)**: 624 TOPS
- **Sparse (FP16)**: 624 TFLOPS

**Impact**: Dominated data center AI training 2020-2023. GPT-3 (175B parameters) trained on A100 clusters.

### Hopper GH100 (2022) — Compute Capability 9.0

**Products**: H100 (SXM5, PCIe), H200

**Manufacturing**: TSMC 4N (custom 4nm)

**SM Count**: 132 SMs (H100 SXM), 114 SMs (H100 PCIe)

**SM Architecture**:
- **128 FP32 CUDA cores** per SM (2× Ampere!)
- **64 FP64 CUDA cores** per SM (2× Ampere)
- **64 INT32 units**
- **4 Tensor Cores (4th generation)** — **FP8** (E4M3, E5M2), TF32, FP16, BF16, INT8
- **4 warp schedulers**
- **32 LD/ST units**
- **16 SFUs**
- **228 KB shared memory** (max, configurable)
- **256 KB register file**

**Key Innovations**:
- **FP8 Tensor Cores**: 2× throughput vs FP16. Transformer Engine auto-scales between FP8/FP16.
- **Transformer Engine**: Automatic precision management for attention layers
- **Thread Block Clusters**: Distributed shared memory across up to 8 SMs
- **TMA (Tensor Memory Accelerator)**: Asynchronous global→shared bulk transfers (hardware-managed)
- **DPX Instructions**: Dynamic programming algorithms (Smith-Waterman, Floyd-Warshall) — 7× speedup
- **Confidential Computing**: AES-256 encryption for data in use
- **HBM3**: 3.35 TB/s (80 GB) — H200: 4.8 TB/s (141 GB HBM3e)
- **NVLink 4.0**: 900 GB/s bidirectional (18× NVLink connections)
- **PCIe Gen 5**: 128 GB/s

**Performance** (H100 SXM):
- FP64: 34 TFLOPS (67 with Tensor Cores)
- FP32: 67 TFLOPS
- TF32: 494 TFLOPS (989 with sparsity)
- **Tensor (FP16/BF16)**: 989 TFLOPS
- **Tensor (FP8)**: 1,979 TFLOPS (3,958 with sparsity)
- **Tensor (INT8)**: 1,979 TOPS

**Impact**: Powers GPT-4, Gemini, Llama 3 training. 3× faster than A100 for LLM training. H200 (2024) with HBM3e: 1.8× A100 inference throughput.

### Blackwell GB100/GB200 (2024) — Compute Capability 10.0

**Products**: B100, B200, GB200 NVL72 (Grace-Blackwell superchip)

**Manufacturing**: TSMC 4NP (enhanced 4nm)

**SM Count**: 208 SMs (GB100 estimated)

**SM Architecture** (official specs pending):
- **128 FP32 CUDA cores** per SM
- **64 FP64 CUDA cores** per SM (estimated)
- **4 Tensor Cores (5th generation)** — **FP4**, FP8, FP16, BF16, TF32
- **4 warp schedulers**
- Enhanced LD/ST units
- Enhanced SFUs
- **288 KB shared memory** (rumored increase)
- **256 KB register file**

**Key Innovations**:
- **FP4 Tensor Cores**: 2× FP8 throughput. 4-bit floating-point for extreme inference acceleration.
- **2nd Gen Transformer Engine**: Automatic FP4/FP8/FP16 precision selection
- **Double Tensor Cores**: 2× Tensor Core count per chip (via chiplet design)
- **HBM3e**: 8 TB/s (192 GB) — 2.4× H100
- **NVLink 5.0**: 1.8 TB/s bidirectional
- **Secure AI**: Enhanced confidential computing
- **Decompression Engine**: On-the-fly model decompression

**Performance** (B200 estimated/announced):
- FP64: ~60 TFLOPS
- **Tensor (FP8)**: ~10 PFLOPS (10,000 TFLOPS)
- **Tensor (FP4)**: **~20 PFLOPS** (20,000 TFLOPS)

**Impact**: Targets trillion-parameter models (GPT-5 scale). 4× faster than H100 for LLM inference. GB200 NVL72 rack: 1.4 exaFLOPS FP4 compute.

---

### Complete Architecture Comparison Table

<table class="table-comparison">
<thead>
<tr>
<th>Specification</th>
<th>Pascal<br>GP100</th>
<th>Volta<br>GV100</th>
<th>Turing<br>TU102</th>
<th>Ampere<br>GA100</th>
<th>Hopper<br>GH100</th>
<th>Blackwell<br>GB100</th>
</tr>
</thead>
<tbody>
<tr class="accent">
<td><strong>Compute Capability</strong></td>
<td>6.0</td>
<td>7.0</td>
<td>7.5</td>
<td>8.0</td>
<td>9.0</td>
<td>10.0</td>
</tr>
<tr>
<td><strong>Manufacturing</strong></td>
<td>16nm</td>
<td>12nm</td>
<td>12nm</td>
<td>7nm</td>
<td>4nm</td>
<td>4nm</td>
</tr>
<tr class="green">
<td><strong>SM Count (max)</strong></td>
<td>60</td>
<td>84</td>
<td>72</td>
<td>128</td>
<td>132</td>
<td>208</td>
</tr>
<tr>
<td><strong>FP32 Cores/SM</strong></td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
<td><strong>128</strong></td>
<td><strong>128</strong></td>
</tr>
<tr>
<td><strong>FP64 Cores/SM</strong></td>
<td>32</td>
<td>32</td>
<td>2 (consumer)</td>
<td>32</td>
<td>64</td>
<td>64</td>
</tr>
<tr>
<td><strong>INT32 Units/SM</strong></td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
</tr>
<tr class="cyan">
<td><strong>Tensor Cores/SM</strong></td>
<td>—</td>
<td>8 (Gen 1)</td>
<td>8 (Gen 2)</td>
<td>4 (Gen 3)</td>
<td>4 (Gen 4)</td>
<td>4 (Gen 5)</td>
</tr>
<tr>
<td><strong>Tensor Precisions</strong></td>
<td>—</td>
<td>FP16</td>
<td>FP16, INT8/4</td>
<td>TF32, FP64, BF16, INT8</td>
<td>FP8, TF32, BF16, FP16</td>
<td><strong>FP4, FP8</strong>, TF32</td>
</tr>
<tr class="purple">
<td><strong>RT Cores/SM</strong></td>
<td>—</td>
<td>—</td>
<td>1</td>
<td>1 (GA102+)</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<td><strong>Warp Schedulers/SM</strong></td>
<td>2</td>
<td>4</td>
<td>4</td>
<td>4</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<td><strong>Max Warps/SM</strong></td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
<td>64</td>
</tr>
<tr>
<td><strong>Max Threads/SM</strong></td>
<td>2048</td>
<td>2048</td>
<td>2048</td>
<td>2048</td>
<td>2048</td>
<td>2048</td>
</tr>
<tr class="orange">
<td><strong>Shared Memory/SM</strong></td>
<td>64 KB</td>
<td>96 KB</td>
<td>64 KB</td>
<td>164 KB</td>
<td><strong>228 KB</strong></td>
<td>288 KB (est.)</td>
</tr>
<tr>
<td><strong>Register File/SM</strong></td>
<td>256 KB</td>
<td>256 KB</td>
<td>256 KB</td>
<td>256 KB</td>
<td>256 KB</td>
<td>256 KB</td>
</tr>
<tr>
<td><strong>L2 Cache</strong></td>
<td>4 MB</td>
<td>6 MB</td>
<td>6 MB</td>
<td>40 MB</td>
<td><strong>60 MB</strong></td>
<td>~80 MB (est.)</td>
</tr>
<tr class="pink">
<td><strong>Memory Type</strong></td>
<td>HBM2</td>
<td>HBM2</td>
<td>GDDR6</td>
<td>HBM2e</td>
<td>HBM3/HBM3e</td>
<td><strong>HBM3e</strong></td>
</tr>
<tr>
<td><strong>Memory Bandwidth</strong></td>
<td>732 GB/s</td>
<td>900 GB/s</td>
<td>616 GB/s</td>
<td>2.0 TB/s</td>
<td>3.35 TB/s<br>(4.8 TB/s H200)</td>
<td><strong>8 TB/s</strong></td>
</tr>
<tr>
<td><strong>Memory Capacity</strong></td>
<td>16 GB</td>
<td>16/32 GB</td>
<td>11 GB</td>
<td>40/80 GB</td>
<td>80 GB<br>(141 GB H200)</td>
<td>192 GB</td>
</tr>
<tr class="teal">
<td><strong>NVLink</strong></td>
<td>160 GB/s</td>
<td>300 GB/s</td>
<td>100 GB/s</td>
<td>600 GB/s</td>
<td>900 GB/s</td>
<td><strong>1.8 TB/s</strong></td>
</tr>
<tr>
<td><strong>FP64 Peak (TFLOPS)</strong></td>
<td>5.3</td>
<td>7.8</td>
<td>0.4</td>
<td>9.7 (19.5)</td>
<td>34 (67)</td>
<td>~60</td>
</tr>
<tr>
<td><strong>FP32 Peak (TFLOPS)</strong></td>
<td>10.6</td>
<td>15.7</td>
<td>13.4</td>
<td>19.5</td>
<td>67</td>
<td>~80</td>
</tr>
<tr class="yellow">
<td><strong>Tensor Peak (TFLOPS)</strong></td>
<td>—</td>
<td>125 (FP16)</td>
<td>107 (FP16)</td>
<td>312 (FP16)</td>
<td>989 (FP16)<br>1979 (FP8)</td>
<td>10,000 (FP8)<br><strong>20,000 (FP4)</strong></td>
</tr>
<tr>
<td><strong>TDP (Watts)</strong></td>
<td>300W</td>
<td>350W</td>
<td>260W</td>
<td>400W</td>
<td>700W</td>
<td>1000W (est.)</td>
</tr>
<tr>
<td><strong>Launch Year</strong></td>
<td>2016</td>
<td>2017</td>
<td>2018</td>
<td>2020</td>
<td>2022</td>
<td>2024</td>
</tr>
</tbody>
</table>

---

## 6. Thread Block Clusters (Hopper+)

### What Are Thread Block Clusters?

Introduced in **Hopper (Compute Capability 9.0)**, **Thread Block Clusters** group multiple thread blocks that execute concurrently on different SMs, enabling **distributed shared memory** — shared memory accessible across SMs within a cluster.

**Traditional Model**: Thread blocks are independent. Shared memory is per-block (per-SM).

**Cluster Model**: Up to 8 thread blocks form a cluster. Cluster-wide shared memory (distributed across SMs) enables fast inter-SM communication.

<div class="diagram">
<div class="diagram-title">Thread Block Cluster Architecture</div>
<div class="flow">
<div class="flow-node green wide">Thread Block Cluster (up to 8 blocks, 8 SMs)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node cyan">SM 0<br>Block 0<br>Shared Mem 0</div>
<div class="flow-node cyan">SM 1<br>Block 1<br>Shared Mem 1</div>
<div class="flow-node cyan">SM 2<br>Block 2<br>Shared Mem 2</div>
<div class="flow-node cyan">SM 3<br>Block 3<br>Shared Mem 3</div>
<div class="flow-arrow accent"></div>
<div class="flow-node purple wide">Distributed Shared Memory (accessible across all SMs in cluster)</div>
</div>
</div>

### Why Thread Block Clusters?

**Transformer Models**: Attention mechanisms require communication between different token embeddings. Clusters enable efficient cross-block data sharing.

**Matrix Tiling**: Large matrix multiplications can partition work across blocks, sharing partial results via distributed shared memory.

**Performance**: Distributed shared memory (~20 TB/s) is **10× faster** than L2 cache (2 TB/s) and **100× faster** than global memory (3.3 TB/s).

### Programming with Clusters

```cpp
#include <cuda/cluster>
#include <cooperative_groups.h>

__global__ void __cluster_dims__(2, 2, 1)  // 2×2 cluster (4 blocks)
cluster_kernel(float* data) {
    namespace cg = cooperative_groups;
    
    // Get cluster group
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    // Distributed shared memory (accessible across cluster)
    __shared__ float smem[256];
    
    // Each block initializes its portion
    smem[threadIdx.x] = data[blockIdx.x * 256 + threadIdx.x];
    cluster.sync();  // Synchronize all blocks in cluster
    
    // Access remote shared memory from another block in cluster
    int remote_block = (cluster.block_rank() + 1) % cluster.num_blocks();
    float* remote_smem = cluster.map_shared_rank(smem, remote_block);
    float remote_value = remote_smem[threadIdx.x];
    
    // Use remote data
    data[blockIdx.x * 256 + threadIdx.x] = smem[threadIdx.x] + remote_value;
}

// Launch with cluster
cudaLaunchConfig_t config = {};
config.gridDim = dim3(4, 1, 1);   // 4 blocks total
config.blockDim = dim3(256, 1, 1);
config.dynamicSmemBytes = 0;

cudaLaunchKernelEx(&config, cluster_kernel, d_data);
```

### Key APIs

- `__cluster_dims__(X, Y, Z)`: Kernel attribute specifying cluster dimensions
- `cg::this_cluster()`: Get cluster group handle
- `cluster.sync()`: Barrier synchronization across all blocks in cluster
- `cluster.map_shared_rank(ptr, rank)`: Get pointer to shared memory of block `rank`
- `cluster.block_rank()`: This block's rank within cluster (0 to `num_blocks-1`)

### Performance Impact — Transformer Example

**GPT-3 Attention Layer** (Hopper H100 vs Ampere A100):

Without clusters (A100):
- Inter-block communication via global memory
- Latency: ~400 cycles
- Bandwidth: 2 TB/s (limited by global memory)

With clusters (H100):
- Inter-block communication via distributed shared memory
- Latency: ~30 cycles
- Bandwidth: ~15 TB/s
- **Speedup**: 3.2× for attention computation

**When to Use Clusters**:
- Attention mechanisms (Q, K, V matrices shared across heads)
- Large matrix operations requiring inter-block communication
- Graph algorithms (neighbor aggregation across blocks)
- Distributed reductions

---

## 7. Occupancy & Performance Optimization

### Theoretical Occupancy Formula

$$
\text{Occupancy} = \min\left(
\frac{\text{Blocks}_\text{SM}}{\text{MaxBlocks}_\text{SM}},
\frac{\text{Warps}_\text{SM}}{\text{MaxWarps}_\text{SM}},
\frac{\text{Registers}_\text{available}}{\text{Registers}_\text{used}},
\frac{\text{SharedMem}_\text{available}}{\text{SharedMem}_\text{used}}
\right)
$$

Where:
- **MaxBlocks/SM**: 32 (all modern architectures)
- **MaxWarps/SM**: 64 (2048 threads)
- **Registers available**: 65,536 × 32-bit registers/SM
- **SharedMem available**: Architecture-dependent (96-228 KB)

### Calculating Occupancy by Hand

**Example**: Hopper H100, kernel with:
- Block size: 512 threads (16 warps)
- Registers/thread: 96
- Shared memory/block: 64 KB

**Step 1 — Block Limit**:
$$
\frac{32 \text{ max blocks}}{1 \text{ block needed}} = 32 \text{ blocks/SM possible}
$$

**Step 2 — Warp Limit**:
$$
\frac{64 \text{ max warps}}{16 \text{ warps/block}} = 4 \text{ blocks/SM}
$$

**Step 3 — Register Limit**:
$$
\text{Regs/block} = 512 \text{ threads} \times 96 \text{ regs/thread} = 49,152 \text{ regs}
$$
$$
\frac{65,536 \text{ regs available}}{49,152 \text{ regs/block}} = 1.33 \rightarrow \lfloor 1.33 \rfloor = 1 \text{ block/SM}
$$

**Step 4 — Shared Memory Limit**:
$$
\frac{228 \text{ KB available}}{64 \text{ KB/block}} = 3.56 \rightarrow \lfloor 3.56 \rfloor = 3 \text{ blocks/SM}
$$

**Limiting Factor**: Registers → **1 block/SM**

**Occupancy**:
$$
\text{Active warps} = 1 \text{ block} \times 16 \text{ warps/block} = 16 \text{ warps}
$$
$$
\text{Occupancy} = \frac{16}{64} = 25\%
$$

### CUDA Occupancy Calculator API

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void my_kernel(float* data, int N) {
    // Kernel implementation
    __shared__ float smem[1024];  // 4 KB
    // ... (uses ~64 registers/thread)
}

int main() {
    int blockSize = 256;
    int minGridSize, optimalBlockSize;
    
    // Calculate optimal block size for max occupancy
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optimalBlockSize,
        my_kernel,
        0,  // Dynamic shared memory per block (0 = static only)
        0   // Block size limit (0 = no limit)
    );
    
    printf("Optimal block size: %d\n", optimalBlockSize);
    
    // Calculate occupancy for specific configuration
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        my_kernel,
        blockSize,
        0  // Dynamic shared memory
    );
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;
    int activeWarps = numBlocks * (blockSize / 32);
    float occupancy = (float)activeWarps / maxWarpsPerSM;
    
    printf("Block size: %d\n", blockSize);
    printf("Blocks per SM: %d\n", numBlocks);
    printf("Active warps per SM: %d / %d\n", activeWarps, maxWarpsPerSM);
    printf("Occupancy: %.1f%%\n", occupancy * 100);
    
    return 0;
}
```

**Output** (example on H100):
```
Optimal block size: 1024
Block size: 256
Blocks per SM: 8
Active warps per SM: 64 / 64
Occupancy: 100.0%
```

### Optimizing for Occupancy vs Performance

#### Case Study 1: Reduce Register Usage

**Original Kernel** (72 registers/thread):
- Block size: 256 threads
- Occupancy: 50%
- Performance: 450 GFLOPS

**Optimized** (48 registers/thread via `-maxrregcount=48`):
- Block size: 256 threads
- Occupancy: 75%
- Performance: **420 GFLOPS** (worse!)

**Why?**: Forcing register limit caused **register spilling** to local memory (slow global memory). The reduced instruction-level parallelism (ILP) outweighed occupancy gains.

**Lesson**: Don't blindly optimize for occupancy. Profile actual performance.

#### Case Study 2: Increase Occupancy for Memory-Bound Kernel

**Original** (low occupancy):
- Block size: 128 threads, 96 registers/thread
- Occupancy: 25%
- Memory bandwidth utilization: 40%
- Performance: 1.2 TB/s

**Optimized** (higher occupancy):
- Block size: 256 threads, reduce to 64 registers/thread
- Occupancy: 75%
- Memory bandwidth utilization: 85%
- Performance: **2.6 TB/s** (2.2× speedup)

**Why?**: Memory-bound kernel benefits from more concurrent warps to hide memory latency. Higher occupancy → better latency hiding → higher throughput.

### Optimization Strategies Summary

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<div class="card-icon">🔧</div>
<div class="card-title">Reduce Register Pressure</div>
<div class="card-desc">
• Use `-maxrregcount=N` cautiously<br>
• Break large kernels into smaller ones<br>
• Recompute cheap values instead of storing<br>
• Use `__restrict__` for pointer aliasing
</div>
</div>

<div class="diagram-card green">
<div class="card-icon">💾</div>
<div class="card-title">Optimize Shared Memory</div>
<div class="card-desc">
• Adjust L1/shared split with `cudaFuncSetAttribute`<br>
• Use padding to avoid bank conflicts<br>
• Minimize per-block usage<br>
• Consider smaller block sizes
</div>
</div>

<div class="diagram-card cyan">
<div class="card-icon">⚡</div>
<div class="card-title">Tune Block Size</div>
<div class="card-desc">
• Use `cudaOccupancyMaxPotentialBlockSize`<br>
• Ensure multiple of 32 (warp size)<br>
• Test multiple sizes (128, 256, 512, 1024)<br>
• Profile with Nsight Compute
</div>
</div>
</div>

### When Occupancy Doesn't Matter

1. **Compute-Bound Kernels**: Already saturating ALUs at 30-50% occupancy.
2. **High ILP**: Kernels with long dependency chains benefit from registers more than threads.
3. **Cache-Sensitive**: Fewer concurrent blocks → better L1/L2 hit rates.

**Golden Rule**: **Optimize for performance, not occupancy metrics.** Use Nsight Compute to identify real bottlenecks.

---

## Summary

This chapter explored NVIDIA GPU microarchitecture from the **Streaming Multiprocessor** (SM) — with its warp schedulers, CUDA cores, Tensor Cores, and memory hierarchy — through the **SIMT execution model** that powers GPU parallelism, to the **architectural evolution** from Pascal's 16nm beginnings to Blackwell's 4nm, 20-petaFLOPS FP4 Tensor Cores.

**Key Takeaways**:

1. **SMs are throughput processors**: Hide latency via massive thread-level parallelism, not complex out-of-order execution.

2. **Tensor Cores revolutionized AI**: From Volta's FP16 to Blackwell's FP4, specialized matrix engines deliver 100-1000× speedups for deep learning.

3. **Memory hierarchy dominates performance**: Registers (1 cycle) → Shared Memory (30 cycles) → L2 (200 cycles) → HBM (600 cycles). Optimize data locality.

4. **Warp divergence is expensive**: SIMT efficiency requires convergent control flow. Refactor branches or accept serialization cost.

5. **Each architecture generation brings paradigm shifts**:
   - Volta: Tensor Cores + independent thread scheduling
   - Ampere: TF32 + structured sparsity + async copy
   - Hopper: FP8 + Thread Block Clusters + TMA
   - Blackwell: FP4 + 8 TB/s HBM3e

6. **Occupancy is a tuning knob, not a goal**: 30% occupancy can outperform 100% if ILP and cache locality are better.

7. **Thread Block Clusters enable new algorithms**: Distributed shared memory across SMs unlocks efficient inter-block communication for Transformers and graph processing.

Understanding these architectural details transforms you from a CUDA programmer who **uses** GPUs to one who **masters** them. In Chapter 3, we'll apply this knowledge to write high-performance CUDA kernels that extract maximum throughput from these incredible machines.

---

**Next: [Chapter 3 — CUDA Programming →](./03_cuda_programming.md)**

---

*Last updated: April 2026*
