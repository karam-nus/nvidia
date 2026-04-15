---
title: "Chapter 1 — GPU Fundamentals"
---

[← Back to Table of Contents](./README.md)

# Chapter 1: GPU Fundamentals — Architecture & Design Principles

Welcome to the foundational chapter of our comprehensive NVIDIA GPU guide. This chapter establishes the conceptual and architectural bedrock upon which all modern GPU computing rests. We'll explore the fundamental question: *what makes a GPU a GPU?* — and in doing so, we'll trace the evolution from specialized graphics accelerators to the massively parallel computing powerhouses that now drive artificial intelligence, scientific simulation, and computational research.

This chapter is dense by design. We embrace technical depth while maintaining pedagogical clarity. By the end, you'll understand not just *what* GPUs are, but *why* they exist, *how* they differ from CPUs at an architectural level, and *which* design principles make them uniquely suited for modern parallel workloads.

---

## Table of Contents

1. [What is a GPU?](#what-is-a-gpu)
2. [Why GPUs? The Case for Massive Parallelism](#why-gpus-the-case-for-massive-parallelism)
3. [CPU vs GPU Architecture](#cpu-vs-gpu-architecture)
4. [What Comprises a GPU](#what-comprises-a-gpu)
5. [GPU Features That Make It a GPU](#gpu-features-that-make-it-a-gpu)
6. [Data Types GPUs Prefer](#data-types-gpus-prefer)
7. [The Roofline Model](#the-roofline-model)
8. [GPU Memory Architecture](#gpu-memory-architecture)
9. [Brief History of GPUs](#brief-history-of-gpus)

---

## What is a GPU?

### Definition and Original Purpose

A **Graphics Processing Unit (GPU)** is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display device. This was the original mandate: render triangles, apply textures, compute lighting, and produce pixels at interactive frame rates.

The GPU's raison d'être was **rasterization** — the process of converting vector graphics (triangles, lines, points) into a raster image (a grid of pixels). This task is embarrassingly parallel: each pixel can be computed independently, each vertex can be transformed independently, each texture lookup can be performed independently. Early GPU architects recognized this inherent parallelism and designed hardware accordingly.

### The Graphics Pipeline

The traditional graphics pipeline that motivated GPU design consists of several stages:

1. **Vertex Processing**: Transform 3D coordinates to 2D screen space
2. **Primitive Assembly**: Group vertices into geometric primitives (triangles, lines)
3. **Rasterization**: Determine which pixels are covered by each primitive
4. **Fragment Processing**: Compute color and depth for each pixel fragment
5. **Output Merging**: Combine fragments with the frame buffer (blending, depth testing)

Each of these stages operates on millions of data elements per frame, and with target frame rates of 60+ fps, the throughput requirements are staggering. A GPU rendering a 1920×1080 scene at 60 fps processes over 124 million pixels per second — and that's before considering overdraw, multiple render passes, or complex shading.

### Evolution to General-Purpose Computing (GPGPU)

The pivotal realization came in the early 2000s: **if the GPU can do this for pixels, it can do this for anything**. Researchers began encoding general computational problems as rendering operations, using textures as data arrays and pixel shaders as compute kernels. This hacky approach proved the concept but was cumbersome.

NVIDIA's 2006 introduction of **CUDA (Compute Unified Device Architecture)** marked the inflection point. CUDA exposed the GPU's parallel execution resources through a C-like programming model, explicitly designed for general-purpose parallel computing (GPGPU). The GPU became a massively parallel coprocessor, not just a graphics accelerator.

### The Fundamental Paradigm Shift

<div class="diagram">
<div class="diagram-title">The CPU-GPU Paradigm Difference</div>
<div class="flow">
<div class="flow-node accent wide">CPU Philosophy: Few Complex Cores</div>
<div class="flow-arrow"></div>
<div class="flow-node purple wide">Minimize Latency<br/>Maximize Single-Thread Performance<br/>Complex Control Logic</div>
</div>
<div class="flow" style="margin-top: 2rem;">
<div class="flow-node green wide">GPU Philosophy: Thousands of Simple Cores</div>
<div class="flow-arrow"></div>
<div class="flow-node cyan wide">Maximize Throughput<br/>Maximize Parallel Performance<br/>Massive ALU Count</div>
</div>
</div>

The architectural philosophy diverges at the most fundamental level:

- **CPU**: A few (4-64) sophisticated cores, each capable of complex out-of-order execution, speculative execution, branch prediction, large caches. Designed to execute a single thread as fast as possible.
- **GPU**: Thousands (10,000+) of simple cores organized into **Streaming Multiprocessors (SMs)**. Each core is relatively simple, but the aggregate provides massive parallel throughput.

This is not a minor implementation detail — it's a foundational design choice that ripples through every aspect of the hardware and software stack.

### Jensen Huang's Vision

NVIDIA CEO Jensen Huang famously articulated the vision: *"The more you buy, the more you save"* — referring not to bulk discounts, but to the energy efficiency of GPU computing. A GPU can perform the same computational work as hundreds of CPUs while consuming a fraction of the total power. This efficiency stems from architectural specialization: by removing per-core overhead (complex branch predictors, out-of-order logic, large caches) and amortizing control logic across many ALUs, GPUs achieve superior performance-per-watt for parallel workloads.

Huang also predicted the "accelerated computing" revolution: that GPUs would become essential for domains far beyond graphics — AI, scientific computing, data analytics, even database queries. This prediction has proven prescient. As of 2024, GPUs power the majority of deep learning training and inference, exascale supercomputers rely on GPU acceleration, and even consumer laptops include discrete GPUs for computational tasks.

---

## Why GPUs? The Case for Massive Parallelism

### The End of Dennard Scaling and Frequency Scaling

For decades, the semiconductor industry rode two exponential curves:

1. **Moore's Law**: Transistor density doubles approximately every two years
2. **Dennard Scaling**: As transistors shrink, power density remains constant (voltage and current scale down proportionally)

Dennard scaling broke down around 2005-2006. Transistors continued to shrink (Moore's Law continued), but power density no longer scaled favorably. Voltage couldn't decrease further without causing unacceptable leakage current. The consequence: **we hit the power wall**. CPUs could no longer simply increase clock frequency to improve performance.

The CPU industry's response was multi-core processors: instead of one 4 GHz core, build four 2 GHz cores. But this only provides modest speedup for many workloads (see Amdahl's Law below). The GPU industry's response was more radical: build thousands of slower cores and embrace massive parallelism.

<div class="diagram">
<div class="diagram-title">The Power Wall and Architectural Responses</div>
<div class="flow-h">
<div class="flow-node accent">Dennard Scaling Ends<br/>(~2005)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node purple">Power Wall:<br/>Can't Increase Frequency</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange">CPU Response:<br/>Multi-Core (4-64 cores)</div>
</div>
<div class="flow-h" style="margin-top: 2rem;">
<div class="flow-node accent">Same Starting Point</div>
<div class="flow-arrow accent"></div>
<div class="flow-node purple">Same Constraint</div>
<div class="flow-arrow purple"></div>
<div class="flow-node green">GPU Response:<br/>Massively Parallel (10,000+ cores)</div>
</div>
</div>

### Why Massive Parallelism Matters

Modern computing workloads increasingly exhibit **data parallelism** — the same operation applied to many data elements:

- **Deep Learning**: Matrix multiplications, convolutions, element-wise operations across millions of parameters
- **Scientific Simulation**: Update millions of grid points, particles, or mesh elements
- **Image Processing**: Apply filters, transformations to millions of pixels
- **Data Analytics**: Aggregate, filter, join millions of records

These workloads are a perfect fit for GPUs. Rather than optimizing a single thread's execution (CPU approach), GPUs distribute the work across thousands of threads executing simultaneously.

### Amdahl's Law vs Gustafson's Law

Two fundamental laws govern parallel speedup, and understanding both is crucial for GPU programming.

#### Amdahl's Law: The Pessimistic View

Amdahl's Law describes the theoretical speedup achievable when parallelizing a program with a serial portion:

$$
S(N) = \frac{1}{(1-P) + \frac{P}{N}}
$$

Where:
- $S(N)$ is the speedup with $N$ processors
- $P$ is the proportion of the program that can be parallelized (0 ≤ P ≤ 1)
- $(1-P)$ is the serial portion

**Key Insight**: The serial portion dominates. Even with infinite processors, speedup is bounded by $\frac{1}{1-P}$. If 5% of your code is serial (P=0.95), maximum speedup is 20×, regardless of how many GPUs you throw at it.

**Example**: Consider a program that is 90% parallelizable:
- With 10 processors: $S(10) = \frac{1}{0.1 + 0.9/10} = \frac{1}{0.19} \approx 5.26×$
- With 100 processors: $S(100) = \frac{1}{0.1 + 0.9/100} = \frac{1}{0.109} \approx 9.17×$
- With ∞ processors: $S(\infty) = \frac{1}{0.1} = 10×$

This seems discouraging. But Amdahl's Law assumes fixed problem size, which is often unrealistic.

#### Gustafson's Law: The Optimistic View

Gustafson's Law recognizes that as computational power increases, we tend to solve larger problems (weak scaling), not just solve the same problem faster (strong scaling):

$$
S(N) = N - (1-P)(N-1) = (1-P) + N \cdot P
$$

Where variables are defined as before, but now P is the parallel portion of the scaled problem.

**Key Insight**: With scaled problems, speedup grows almost linearly with processor count. The serial portion doesn't dominate because it doesn't grow with problem size.

**Example**: Same 90% parallel program (P=0.9):
- With 10 processors: $S(10) = 0.1 + 10 \times 0.9 = 9.1×$
- With 100 processors: $S(100) = 0.1 + 100 \times 0.9 = 90.1×$
- With 10,000 processors: $S(10000) = 0.1 + 10000 \times 0.9 = 9000.1×$

**This is why GPUs matter**: They enable solving problems that were previously computationally infeasible. Training GPT-3 (175B parameters) would take centuries on CPUs; on GPUs, it took weeks. This isn't just a speedup — it's a qualitative transformation of what's possible.

### Real-World Speedup Examples

Let's ground this in concrete examples:

| Application | CPU Baseline | GPU Accelerated | Speedup | Notes |
|-------------|--------------|-----------------|---------|-------|
| Deep Learning Training (ResNet-50) | ~14 days (64-core CPU) | ~8 hours (8×A100) | ~42× | Weak scaling: larger batch size on GPU |
| Molecular Dynamics (NAMD) | 24 hours/ns (128 cores) | 1.5 hours/ns (4×V100) | ~16× | Strong scaling: same simulation |
| BLAST Sequence Alignment | 8 hours (single CPU) | 12 minutes (single GPU) | ~40× | Highly parallel search |
| JPEG Encoding | 45 fps (CPU) | 1200+ fps (GPU) | ~27× | Embarrassingly parallel |
| Monte Carlo Simulation | 1M samples/sec (CPU) | 500M samples/sec (GPU) | ~500× | Independent random trials |

These aren't cherry-picked benchmarks — they represent typical GPU-amenable workloads. The pattern is clear: **data-parallel operations with minimal inter-thread communication achieve massive speedups**.

---

## CPU vs GPU Architecture

### Architectural Philosophy: Latency vs Throughput

The CPU-GPU dichotomy is best understood through the lens of **design philosophy**:

**CPU (Latency-Oriented Design)**:
- **Goal**: Minimize time to complete a single task
- **Strategy**: Complex control logic, out-of-order execution, speculative execution, branch prediction, large caches
- **Metaphor**: A luxury sports car — fast acceleration, responsive, expensive per unit

**GPU (Throughput-Oriented Design)**:
- **Goal**: Maximize total work completed per unit time
- **Strategy**: Massive parallelism, simple in-order cores, hardware multithreading, high memory bandwidth
- **Metaphor**: A freight train — slow to start, but enormous total cargo capacity

### Control Logic vs ALU Ratio

This philosophical difference manifests in silicon allocation:

<div class="diagram">
<div class="diagram-title">Silicon Area Allocation: CPU vs GPU</div>
<div class="flow-h">
<div class="flow-node accent wide">CPU Die Area</div>
<div class="flow-arrow"></div>
<div class="flow-node purple">~50% Control<br/>(Branch Prediction, OoO, Scheduling)</div>
<div class="flow-arrow"></div>
<div class="flow-node orange">~30% Cache<br/>(L1/L2/L3)</div>
<div class="flow-arrow"></div>
<div class="flow-node cyan">~20% ALUs<br/>(Execution Units)</div>
</div>
<div class="flow-h" style="margin-top: 2rem;">
<div class="flow-node green wide">GPU Die Area</div>
<div class="flow-arrow"></div>
<div class="flow-node teal">~10% Control<br/>(Simple Schedulers)</div>
<div class="flow-arrow"></div>
<div class="flow-node yellow">~10% Cache<br/>(Small L1/L2)</div>
<div class="flow-arrow"></div>
<div class="flow-node pink">~80% ALUs<br/>(CUDA Cores, Tensor Cores)</div>
</div>
</div>

A modern CPU might devote:
- **50%** to control logic (branch predictors, reorder buffers, instruction decoders, schedulers)
- **30%** to cache hierarchies (L1/L2/L3)
- **20%** to ALUs (arithmetic logic units)

A modern GPU devotes:
- **80%** to ALUs (CUDA cores, Tensor cores, SFUs)
- **10%** to cache (smaller L1/L2, no L3)
- **10%** to control (simple warp schedulers)

**Why the difference?** GPUs amortize control logic across many ALUs. A single instruction decoder feeds 32 ALUs (a warp) in SIMT fashion. CPUs, optimizing for single-thread performance, must provide sophisticated control for each core.

### Cache Hierarchy Differences

Caches hide memory latency, but at the cost of silicon area and power. CPUs and GPUs make different trade-offs:

| Feature | CPU | GPU |
|---------|-----|-----|
| **L1 Cache per Core** | 32-64 KB (data), 32-64 KB (instruction) | 128 KB shared across SM (data+instruction) |
| **L2 Cache** | 256 KB - 2 MB per core | 40-60 MB shared across entire GPU |
| **L3 Cache** | 32-256 MB shared | None (replaced with more L2) |
| **Cache Line Size** | 64 bytes | 128 bytes |
| **Replacement Policy** | LRU or pseudo-LRU | Streaming or LRU |
| **Write Policy** | Write-back, write-allocate | Write-evict (L1), write-back (L2) |
| **Coherency** | Full hardware cache coherence (MESI/MOESI) | Relaxed (programmer-managed via atomics) |

**CPU Strategy**: Large caches compensate for limited thread count. If you have 64 threads across 8 cores, you can afford 256 MB of L3 cache.

**GPU Strategy**: Small caches supplemented by explicit programmer-managed shared memory and massive thread count. If memory access stalls, switch to another of the 2048 threads on this SM. The cache exists primarily for coalesced access and temporary reuse.

### Branch Prediction vs Warp Divergence

**CPU Branch Prediction**: Modern CPUs employ sophisticated branch predictors (tournament predictors, perceptrons, TAGE) that achieve 95%+ accuracy. They speculatively execute down predicted paths, rolling back if wrong. This allows hiding control flow costs in single-threaded code.

**GPU Warp Divergence**: GPUs use **SIMT (Single Instruction Multiple Thread)** execution. A warp of 32 threads executes the same instruction on different data. When a branch occurs:

```cuda
if (threadIdx.x < 16) {
    // Path A
} else {
    // Path B
}
```

The GPU must serialize execution: first execute Path A for threads 0-15 (while threads 16-31 are masked), then execute Path B for threads 16-31 (while threads 0-15 are masked). **Total time = Time(A) + Time(B)**, not max(Time(A), Time(B)).

**Why this difference?** GPUs trade single-thread performance for aggregate throughput. Branch prediction requires per-thread hardware (expensive at 10,000+ thread scale). SIMT divergence handling is simpler: just mask ALUs. For data-parallel workloads with uniform control flow, this is a winning trade-off.

### Comprehensive Comparison Table

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Design Philosophy** | Latency-oriented | Throughput-oriented |
| **Core Count** | 4-64 complex cores | 10,000+ simple cores (organized into 100+ SMs) |
| **Clock Frequency** | 3-5 GHz | 1-2 GHz |
| **Threads per Core** | 1-2 (SMT/Hyper-Threading) | 64-128 (hardware multithreading) |
| **Total Active Threads** | ~100-200 | 100,000+ |
| **Instruction Issue** | Out-of-order, speculative | In-order, grouped (warp-level) |
| **Branch Handling** | Sophisticated prediction | SIMT divergence (serialization) |
| **Control Logic** | ~50% of die | ~10% of die |
| **Cache Hierarchy** | Large (MB-scale L3) | Small (KB-scale L1, MB-scale shared L2) |
| **Memory Latency Hiding** | Caches + prefetching | Thread switching (zero-overhead context switching) |
| **Memory Bandwidth** | 50-100 GB/s (DDR4/DDR5) | 1000-3000 GB/s (HBM2/HBM3) |
| **Programming Model** | Sequential with explicit threads/processes | Massively parallel with kernel launches |
| **Best For** | Complex control flow, unpredictable branches | Data-parallel operations, uniform control flow |
| **Worst For** | Limited parallelism | Serial code, heavy branching |
| **Power Consumption** | 65-250W (desktop), 15-45W (laptop) | 250-700W (datacenter), 80-150W (consumer) |
| **Price/Performance** | High per-core cost | Low per-FLOP cost |

### When to Use Which?

The choice isn't binary — modern applications use both:

**Use CPU for**:
- Complex control flow (parsers, compilers, databases)
- Low-latency requirements (real-time systems)
- Small datasets (cache-resident)
- Sequential algorithms (linked list traversal, tree walks)
- Operating system tasks (scheduling, I/O)

**Use GPU for**:
- Large data-parallel operations (matrix math, convolutions)
- High throughput requirements (video encoding, molecular dynamics)
- Algorithms with high arithmetic intensity (dense linear algebra)
- Embarrassingly parallel tasks (Monte Carlo, ray tracing)

**Use Both for**:
- Most modern HPC applications (CPU handles I/O, setup, GPU handles compute)
- Deep learning pipelines (CPU handles data loading, GPU handles training)
- Scientific workflows (CPU handles analysis, GPU handles simulation)

---

## What Comprises a GPU

A modern NVIDIA GPU (e.g., H100, A100, RTX 4090) is a complex System-on-Chip (SoC) with heterogeneous components. Let's dissect the anatomy of a GPU, component by component.

### Streaming Multiprocessors (SMs)

The **Streaming Multiprocessor (SM)** is the fundamental building block — the "core" of a GPU, though each SM itself contains hundreds of cores. An NVIDIA H100 has 132 SMs; an RTX 4090 has 128 SMs.

Each SM contains:
- CUDA cores (FP32, INT32)
- Tensor Cores (mixed-precision matrix multiply-accumulate)
- Special Function Units (SFUs for transcendentals)
- Load/Store units (LD/ST)
- Texture units (TEX) for graphics
- Warp schedulers
- Register file
- Shared memory / L1 cache

**Analogy**: If the GPU is a factory, each SM is a production line with many workers (CUDA cores) and specialized machines (Tensor Cores, SFUs).

### CUDA Cores

**CUDA cores** are the scalar processing units — the individual ALUs that perform integer and floating-point arithmetic. The term "CUDA core" is somewhat marketing-speak (AMD calls them "stream processors"), but it's useful nomenclature.

A CUDA core can execute one floating-point or integer operation per clock cycle. Modern SMs contain:
- **64-128 FP32 cores** (single-precision)
- **32-64 FP64 cores** (double-precision, or absent in consumer GPUs)
- **64-128 INT32 cores** (integer operations)

**Specification Example (H100 SM)**:
- 128 FP32 CUDA cores
- 64 FP64 CUDA cores
- 128 INT32 cores
- 4 Tensor Cores (4th gen)

Total across 132 SMs: **16,896 FP32 cores**, **8,448 FP64 cores**.

### Tensor Cores

Introduced in Volta (2017), **Tensor Cores** are specialized units for matrix multiply-accumulate operations:

$$
D = A \times B + C
$$

Where A, B, C, D are matrices (typically 16×16 or smaller tiles). Tensor Cores perform this in a single instruction, vastly outperforming CUDA cores for this operation.

**Why specialized hardware?** Matrix multiplication is the cornerstone of deep learning (every fully connected layer, every convolution can be expressed as matrix multiplication). A Tensor Core can perform 256+ FLOPs per clock, vs 1 FLOP per CUDA core per clock.

**Evolution**:
- **1st Gen (Volta)**: FP16 input, FP32 accumulate
- **2nd Gen (Turing)**: Added INT8, INT4, binary operations
- **3rd Gen (Ampere)**: Added TF32, BF16, FP64 Tensor Cores
- **4th Gen (Hopper)**: FP8 (E4M3, E5M2), sparsity support

### RT Cores

**RT Cores** (Ray Tracing Cores) are specialized for bounding volume hierarchy (BVH) traversal and ray-triangle intersection — the fundamental operations in ray tracing. Introduced in Turing (2018).

While not typically used for general compute, RT Cores are an example of GPU heterogeneity: different workloads get different specialized hardware.

### Memory Controllers

**Memory controllers** interface between the GPU's internal fabric and external high-bandwidth memory (HBM). A modern GPU has 6-12 memory controllers, each connected to a stack of HBM.

**H100 Example**:
- 5 HBM3 stacks
- 80 GB total capacity
- 3.35 TB/s bandwidth
- ECC (error-correcting code) supported

### Cache Hierarchy

**L1 Cache**: Per-SM, typically 128 KB configurable as data cache or shared memory. Fast (400+ GB/s bandwidth per SM), low latency (~28 cycles).

**L2 Cache**: Shared across all SMs, 40-60 MB in modern GPUs. Slower than L1 (~300 cycles latency), but much larger. Serves as a backstop before going to HBM.

**No L3**: Unlike CPUs, GPUs typically don't have L3. The L2 is large enough for the common case, and HBM bandwidth is high enough that the latency hit is manageable (especially given thread switching).

### Register File

The **register file** is the fastest storage — per-thread registers. Modern SMs have **65,536 × 32-bit registers** (256 KB per SM). With 2048 resident threads, that's 32 registers per thread on average, though distribution is uneven (some threads use more, some less).

**Why so large?** Registers hide latency. When a memory load stalls, the SM switches to another warp whose registers are ready. This requires keeping many warps' register state simultaneously resident.

### Warp Schedulers

**Warp schedulers** select which warp to issue instructions from each cycle. Modern SMs have **4 warp schedulers**, each capable of issuing 1-2 instructions per cycle.

**Scheduling Policy**: Round-robin among ready warps (warps not stalled on memory or dependencies). This is hardware-managed — programmers don't control scheduling.

### Dispatch Units

**Dispatch units** send issued instructions to the appropriate execution units (CUDA cores, Tensor Cores, SFUs, LD/ST units). Multiple dispatch units allow instruction-level parallelism within a warp.

### Special Function Units (SFUs)

**SFUs** compute transcendental functions (sin, cos, exp, log, sqrt) and other special operations. These are implemented with polynomial approximations or table lookups, achieving one operation per cycle (vs many cycles if implemented with CUDA cores).

**Typical SM**: 16-32 SFUs (fewer than CUDA cores because transcendentals are less common than basic arithmetic).

### Load/Store Units (LD/ST)

**Load/Store units** handle memory transactions between registers and memory (shared memory, L1, L2, global memory). Modern SMs have 32-64 LD/ST units.

**Coalescing**: LD/ST units coalesce memory accesses from threads in a warp into fewer, wider transactions when threads access contiguous addresses. This is crucial for memory bandwidth efficiency.

### Block Diagram

<div class="diagram">
<div class="diagram-title">Streaming Multiprocessor (SM) Architecture</div>
<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<div class="card-icon">⚙️</div>
<div class="card-title">Warp Schedulers</div>
<div class="card-desc">4 schedulers issue instructions from ready warps to execution units</div>
</div>
<div class="diagram-card green">
<div class="card-icon">🔢</div>
<div class="card-title">CUDA Cores</div>
<div class="card-desc">64-128 FP32/INT32 ALUs for scalar arithmetic operations</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">🧮</div>
<div class="card-title">Tensor Cores</div>
<div class="card-desc">4-8 specialized matrix multiply-accumulate units</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">📐</div>
<div class="card-title">Special Function Units</div>
<div class="card-desc">16-32 SFUs for transcendentals (sin, cos, exp, sqrt)</div>
</div>
<div class="diagram-card cyan">
<div class="card-icon">💾</div>
<div class="card-title">Load/Store Units</div>
<div class="card-desc">32-64 LD/ST units for memory transactions with coalescing</div>
</div>
<div class="diagram-card pink">
<div class="card-icon">📦</div>
<div class="card-title">Register File</div>
<div class="card-desc">65,536×32-bit registers (256 KB) for thread-local storage</div>
</div>
<div class="diagram-card teal">
<div class="card-icon">⚡</div>
<div class="card-title">Shared Memory / L1</div>
<div class="card-desc">128 KB configurable fast memory shared among threads in a block</div>
</div>
<div class="diagram-card yellow">
<div class="card-icon">🎯</div>
<div class="card-title">Texture Units</div>
<div class="card-desc">Hardware for filtered texture sampling (graphics and interpolation)</div>
</div>
<div class="diagram-card accent">
<div class="card-icon">🔗</div>
<div class="card-title">Dispatch Units</div>
<div class="card-desc">Route instructions to appropriate execution units</div>
</div>
</div>
</div>

---

## GPU Features That Make It a GPU

What distinguishes a GPU from a generic multicore processor? Several architectural features define the "GPU-ness" of these devices.

### 1. SIMT Execution Model

**SIMT (Single Instruction Multiple Thread)** is NVIDIA's execution model. It's similar to SIMD (Single Instruction Multiple Data), but with key differences:

- **SIMD**: One instruction operates on a vector of data (e.g., AVX-512 on CPU)
- **SIMT**: One instruction is issued, and multiple threads execute it independently (but in lockstep)

**Warp**: A group of 32 threads that execute together in SIMT fashion. All threads in a warp execute the same instruction at the same time (modulo divergence).

**Example**:
```cuda
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // All threads in warp execute this simultaneously
    }
}
```

When 32 threads execute `c[idx] = a[idx] + b[idx]`, the hardware issues one add instruction, and 32 ALUs execute it simultaneously on different data.

### 2. Warp-Based Execution

The warp is the fundamental unit of execution. Key properties:

- **Size**: Always 32 threads (on NVIDIA; AMD uses 64)
- **Scheduling**: Warps, not individual threads, are scheduled
- **Divergence**: When threads in a warp take different branches, execution serializes
- **Convergence**: After divergent code, threads reconverge at immediate post-dominator

**Optimization Principle**: Minimize warp divergence. Organize data so threads in a warp follow the same control flow.

### 3. Massive Register Files

We mentioned earlier: **65,536 registers per SM**. This is 10-100× more than a CPU core's register file. Why?

**Latency Hiding Through Thread Switching**: When a warp stalls (memory load, long-latency operation), the scheduler immediately switches to another warp whose operands are ready. This is **zero-overhead context switching** — the new warp's registers are already loaded, because all resident warps' registers are simultaneously resident in the register file.

**Trade-off**: More registers per thread = fewer threads resident = less latency hiding. The occupancy calculation balances register usage against resident threads.

### 4. Hardware Thread Scheduling

**CPU**: Thread scheduling is OS-managed (millisecond timescale), with context switching overhead (saving/restoring registers, TLB flush).

**GPU**: Thread scheduling is hardware-managed (nanosecond timescale), with zero overhead. The warp scheduler simply selects a different ready warp each cycle.

This enables **latency hiding**: A memory load takes 400+ cycles? No problem — switch to another of the 64 resident warps. By the time we cycle through warps, the memory load is back.

**Consequence**: GPUs tolerate latency that would cripple CPU performance. A 400-cycle memory latency is acceptable if we have 64 warps to keep the ALUs busy.

### 5. Coalesced Memory Access Patterns

**Coalescing** is the GPU memory system's mechanism for achieving high bandwidth. When threads in a warp access contiguous memory addresses, the LD/ST units combine these into a single wide transaction.

**Example**:
```cuda
// Coalesced access (threads access contiguous elements)
__global__ void coalesced(float *data) {
    int idx = threadIdx.x;
    float val = data[idx];  // Thread 0→data[0], Thread 1→data[1], ...
}

// Uncoalesced access (threads access strided elements)
__global__ void strided(float *data, int stride) {
    int idx = threadIdx.x * stride;
    float val = data[idx];  // Thread 0→data[0], Thread 1→data[stride], ...
}
```

**Coalesced**: 32 threads access data[0:31], combined into 1-2 128-byte transactions (4-8 GB/s achieved).

**Strided with stride=32**: 32 threads access data[0, 32, 64, ...], resulting in 32 separate transactions (bandwidth reduced 32×).

**Optimization Principle**: Structure data and access patterns for coalescing. This is often the dominant factor in memory-bound kernel performance.

### 6. High Memory Bandwidth (HBM)

**HBM (High Bandwidth Memory)** is 3D-stacked DRAM placed on the same package as the GPU die, connected via wide (1024-bit) buses.

**Evolution**:
- **GDDR6** (consumer GPUs): 384-bit bus, ~900 GB/s
- **HBM2** (V100, A100): 4096-bit bus, ~1500 GB/s
- **HBM2e** (A100): 5120-bit bus, ~2000 GB/s
- **HBM3** (H100): 5120-bit bus, ~3350 GB/s

**Why so much bandwidth?** With 10,000+ threads, each potentially performing memory loads/stores, aggregate bandwidth demand is enormous. HBM ensures memory isn't the bottleneck.

**CPU Comparison**: DDR5 (high-end desktop) provides ~100 GB/s. Even with dual-channel, CPUs have 30× less memory bandwidth than GPUs.

<div class="diagram">
<div class="diagram-title">Key GPU Architectural Features</div>
<div class="diagram-grid cols-2">
<div class="diagram-card accent">
<div class="card-icon">🔄</div>
<div class="card-title">SIMT Execution</div>
<div class="card-desc">32-thread warps execute same instruction in lockstep, enabling massive parallelism with simple control hardware</div>
</div>
<div class="diagram-card green">
<div class="card-icon">🎯</div>
<div class="card-title">Zero-Overhead Switching</div>
<div class="card-desc">Hardware thread scheduler switches between warps instantly, hiding memory latency with computation</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">📦</div>
<div class="card-title">Massive Register Files</div>
<div class="card-desc">256 KB per SM enables 64+ warps resident simultaneously for effective latency hiding</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">⚡</div>
<div class="card-title">Coalesced Memory Access</div>
<div class="card-desc">Hardware combines contiguous thread accesses into wide transactions for maximum bandwidth utilization</div>
</div>
<div class="diagram-card cyan">
<div class="card-icon">💾</div>
<div class="card-title">High Bandwidth Memory</div>
<div class="card-desc">HBM3 delivers 3+ TB/s through 3D-stacked DRAM with 5120-bit interfaces</div>
</div>
<div class="diagram-card pink">
<div class="card-icon">🧮</div>
<div class="card-title">Specialized Compute Units</div>
<div class="card-desc">Tensor Cores, RT Cores provide 10-100× speedup for domain-specific operations</div>
</div>
</div>
</div>

---

## Data Types GPUs Prefer

Modern GPUs support a heterogeneous set of numeric data types, each with different precision, range, and performance characteristics. Understanding these types is crucial for performance optimization — using the wrong type can leave 10× performance on the table.

### Why Multiple Data Types?

The trade-off space is three-dimensional:

1. **Precision**: How many distinct values can be represented? How small is the rounding error?
2. **Range**: What's the largest/smallest representable value?
3. **Throughput**: How many operations per second can the hardware perform?

Deep learning, in particular, has driven innovation in reduced-precision formats. Neural networks are surprisingly robust to precision degradation — training with FP16 or even INT8 often matches FP32 accuracy while providing 2-16× speedup.

### Floating-Point Representation Primer

A floating-point number is represented as:

$$
\text{value} = (-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times (1 + \text{mantissa})
$$

Where:
- **Sign bit** (1 bit): 0 for positive, 1 for negative
- **Exponent** (E bits): Biased exponent (bias = $2^{E-1} - 1$)
- **Mantissa/Fraction** (M bits): Fractional part with implicit leading 1

**Example (FP32)**:
- 1 sign bit, 8 exponent bits (bias=127), 23 mantissa bits
- Value: $(-1)^s \times 2^{(e-127)} \times (1.m)$

### FP64 (Double Precision)

**Format**: 1 sign + 11 exponent + 52 mantissa = 64 bits

**Range**: ~$\pm 2.2 \times 10^{-308}$ to $\pm 1.8 \times 10^{308}$

**Precision**: ~15-17 decimal digits

**Use Cases**: Scientific computing requiring high accuracy (climate models, quantum chemistry, financial calculations)

**Hardware Support**:
- Consumer GPUs (RTX series): Heavily throttled (1/32 FP32 rate)
- Datacenter GPUs (A100, H100): Full support (1/2 FP32 rate)

**Performance (H100)**:
- 60 TFLOPS FP64
- 120 TFLOPS FP64 Tensor Core

### FP32 (Single Precision)

**Format**: 1 sign + 8 exponent + 23 mantissa = 32 bits

**Range**: ~$\pm 1.4 \times 10^{-45}$ to $\pm 3.4 \times 10^{38}$

**Precision**: ~6-9 decimal digits

**Use Cases**: Default for most GPU computing, graphics, neural network inference

**Hardware Support**: Native on all CUDA cores

**Performance (H100)**:
- 60 TFLOPS FP32 (CUDA cores)
- 989 TFLOPS FP32 (Tensor Cores with TF32)

### TF32 (TensorFloat-32)

**Format**: 1 sign + 8 exponent + 10 mantissa = 19 bits (internally), stored in 32-bit

**Range**: Same as FP32 (same exponent bits)

**Precision**: Reduced (~3 decimal digits) due to 10-bit mantissa

**Use Cases**: Tensor Core operations in deep learning, balancing FP32 range with FP16 performance

**Hardware Support**: Ampere (A100) and newer Tensor Cores

**Key Insight**: TF32 is NVIDIA's clever compromise — keep FP32's range (reducing overflow/underflow issues) but truncate precision to fit in Tensor Core datapaths designed for FP16. Enables using Tensor Cores with FP32 inputs without explicit conversion.

**Performance (H100)**: 989 TFLOPS (same as FP32 Tensor Core throughput)

### FP16 (Half Precision)

**Format**: 1 sign + 5 exponent + 10 mantissa = 16 bits

**Range**: ~$\pm 6.1 \times 10^{-5}$ to $\pm 6.5 \times 10^{4}$

**Precision**: ~3-4 decimal digits

**Use Cases**: Deep learning (training with mixed precision, inference), graphics

**Hardware Support**: Tensor Cores (1st gen onward), CUDA cores (FP16×2 instruction)

**Challenges**: Narrow range can cause overflow/underflow during training (requires loss scaling)

**Performance (H100)**:
- 120 TFLOPS FP16 (CUDA cores)
- 1979 TFLOPS FP16 (Tensor Cores)

### BF16 (Brain Float 16)

**Format**: 1 sign + 8 exponent + 7 mantissa = 16 bits

**Range**: Same as FP32 (same exponent bits)

**Precision**: ~2-3 decimal digits (reduced mantissa)

**Use Cases**: Deep learning training, balancing FP32 range with FP16 size

**Hardware Support**: Ampere (A100) and newer Tensor Cores, TPUs

**Key Insight**: BF16 is Google's innovation (from TPU design), adopted by NVIDIA. By keeping FP32's 8-bit exponent, it eliminates the overflow/underflow issues of FP16, at the cost of reduced precision. For neural networks (which tolerate precision loss better than range loss), this is a great trade-off.

**Performance (H100)**: 1979 TFLOPS (Tensor Cores)

### FP8 (8-bit Floating Point)

**Two Formats**:
1. **E4M3** (1 sign + 4 exponent + 3 mantissa): Higher precision, narrower range
   - Range: ~$\pm 1.9 \times 10^{-3}$ to $\pm 448$
2. **E5M2** (1 sign + 5 exponent + 2 mantissa): Lower precision, wider range
   - Range: ~$\pm 1.5 \times 10^{-5}$ to $\pm 5.7 \times 10^{4}$

**Use Cases**: Neural network inference, training with extreme quantization

**Hardware Support**: Hopper (H100) Tensor Cores (4th gen)

**Strategy**: Use E4M3 for forward pass (needs precision), E5M2 for backward pass (needs range for gradients)

**Performance (H100)**: 3958 TFLOPS (Tensor Cores) — 2× FP16 throughput!

### FP4 (4-bit Floating Point)

**Experimental**: Not yet standardized, but research explores 4-bit formats for inference.

**Potential Format**: 1 sign + 2 exponent + 1 mantissa = 4 bits

**Use Cases**: Ultra-low-power inference, edge devices

**Hardware Support**: Not in current GPUs, but future architectures may include it

### INT8 (8-bit Integer)

**Format**: Signed or unsigned 8-bit integer

**Range**: -128 to 127 (signed), 0 to 255 (unsigned)

**Use Cases**: Quantized neural network inference, image processing

**Hardware Support**: Tensor Cores (2nd gen onward), INT8 CUDA instructions

**Quantization**: Convert FP32 weights/activations to INT8 via:
$$
q = \text{round}\left( \frac{x}{\text{scale}} \right) + \text{zero\_point}
$$

**Performance (H100)**: 3958 TOPS (Tensor Cores, same as FP8)

### INT4 (4-bit Integer)

**Format**: Signed or unsigned 4-bit integer

**Range**: -8 to 7 (signed), 0 to 15 (unsigned)

**Use Cases**: Extreme quantization for inference (e.g., GPTQ, AWQ)

**Hardware Support**: Tensor Cores (2nd gen onward)

**Performance (H100)**: 7916 TOPS (Tensor Cores) — 2× INT8 throughput!

### Data Type Comparison Table

| Type | Bits | Sign | Exp | Mantissa | Range (approx) | Precision | Throughput (H100 Tensor Core) | Primary Use |
|------|------|------|-----|----------|----------------|-----------|-------------------------------|-------------|
| **FP64** | 64 | 1 | 11 | 52 | $\pm 10^{308}$ | 15-17 digits | 60 TFLOPS | Scientific computing |
| **FP32** | 32 | 1 | 8 | 23 | $\pm 10^{38}$ | 6-9 digits | 989 TFLOPS (TF32) | General compute, graphics |
| **TF32** | 19* | 1 | 8 | 10 | $\pm 10^{38}$ | 3 digits | 989 TFLOPS | DL training (Tensor Cores) |
| **BF16** | 16 | 1 | 8 | 7 | $\pm 10^{38}$ | 2-3 digits | 1979 TFLOPS | DL training |
| **FP16** | 16 | 1 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | 3-4 digits | 1979 TFLOPS | DL training/inference |
| **FP8 E4M3** | 8 | 1 | 4 | 3 | $\pm 448$ | ~2 digits | 3958 TFLOPS | DL inference, forward pass |
| **FP8 E5M2** | 8 | 1 | 5 | 2 | $\pm 5.7 \times 10^{4}$ | ~1 digit | 3958 TFLOPS | DL inference, backward pass |
| **INT8** | 8 | 1 | — | — | -128 to 127 | Integer | 3958 TOPS | Quantized inference |
| **INT4** | 4 | 1 | — | — | -8 to 7 | Integer | 7916 TOPS | Extreme quantization |

*TF32 is 19 bits logically but stored/transmitted as 32-bit.

### Bit Layout Diagrams

Here's how bits are allocated for key formats:

**FP32 (IEEE 754 Single Precision)**:
```
┌─┬────────┬───────────────────────┐
│S│EEEEEEEE│MMMMMMMMMMMMMMMMMMMMMMM│  S=Sign, E=Exponent, M=Mantissa
└─┴────────┴───────────────────────┘
 1    8              23             = 32 bits
```

**FP16 (IEEE 754 Half Precision)**:
```
┌─┬─────┬──────────┐
│S│EEEEE│MMMMMMMMMM│
└─┴─────┴──────────┘
 1   5       10      = 16 bits
```

**BF16 (Brain Float 16)**:
```
┌─┬────────┬───────┐
│S│EEEEEEEE│MMMMMMM│  Same exponent as FP32, truncated mantissa
└─┴────────┴───────┘
 1    8        7     = 16 bits
```

**TF32 (TensorFloat-32)**:
```
┌─┬────────┬──────────┐
│S│EEEEEEEE│MMMMMMMMMM│  Stored as FP32, but mantissa truncated in computation
└─┴────────┴──────────┘
 1    8         10      = 19 bits (padded to 32)
```

**FP8 E4M3**:
```
┌─┬────┬───┐
│S│EEEE│MMM│
└─┴────┴───┘
 1   4    3   = 8 bits
```

**FP8 E5M2**:
```
┌─┬─────┬──┐
│S│EEEEE│MM│
└─┴─────┴──┘
 1   5    2   = 8 bits
```

### GPU Generation Support Matrix

| Type | Volta (V100) | Turing (RTX 20) | Ampere (A100, RTX 30) | Hopper (H100) | Ada (RTX 40) |
|------|--------------|-----------------|----------------------|---------------|--------------|
| **FP64** | ✅ Full | ✅ Throttled | ✅ Full (A100) / Throttled (RTX) | ✅ Full | ✅ Throttled |
| **FP32** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **TF32** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **FP16** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **BF16** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **FP8** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **INT8** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **INT4** | ❌ | ✅ | ✅ | ✅ | ✅ |

### Practical Guidance

**When to use each type**:

- **FP64**: Only when absolutely necessary (iterative refinement, high-accuracy simulations). Check if mixed-precision (FP64/FP32) suffices.
- **FP32**: Default for prototyping, most scientific computing, graphics. Safe choice.
- **TF32**: Automatic for Tensor Core matmuls on Ampere+ when using FP32 inputs. Free speedup.
- **BF16**: DL training on modern GPUs. Preferred over FP16 due to better range.
- **FP16**: DL training on older GPUs (Volta, Turing), DL inference, mixed-precision scientific code.
- **FP8**: DL inference on Hopper, experimental training with extreme quantization.
- **INT8**: Quantized inference (post-training quantization), embedded deployment.
- **INT4**: Model compression (GPTQ, AWQ), ultra-low-latency inference.

**Code Example: Mixed Precision Training**:
```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # For loss scaling

for data, target in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16/BF16
    with autocast(dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

This automatically uses BF16 for matmuls (via Tensor Cores) while keeping master weights in FP32.

---

## The Roofline Model

The **Roofline Model** is a visual performance model that helps understand whether a kernel is compute-bound or memory-bound — and thus where optimization efforts should focus.

### Core Concepts

**Arithmetic Intensity**: The ratio of floating-point operations to memory traffic:

$$
I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}
$$

Units: FLOPs/Byte (or OPs/Byte for integer operations).

**Operational Intensity**: Often used interchangeably with arithmetic intensity, but sometimes refers to work per DRAM byte (excluding cache).

**Peak Performance**: Maximum computational throughput of the hardware (e.g., 989 TFLOPS for H100 TF32).

**Peak Bandwidth**: Maximum memory bandwidth (e.g., 3.35 TB/s for H100 HBM3).

### The Roofline Equation

The achievable performance is bounded by:

$$
\text{Performance} = \min\left( \text{Peak Compute}, \text{Peak Bandwidth} \times I \right)
$$

This creates two regimes:

1. **Memory-Bound**: $\text{Peak Bandwidth} \times I < \text{Peak Compute}$
   - Performance limited by memory bandwidth
   - Increasing compute doesn't help; optimize memory access

2. **Compute-Bound**: $\text{Peak Bandwidth} \times I \geq \text{Peak Compute}$
   - Performance limited by compute throughput
   - Increasing bandwidth doesn't help; optimize compute (use Tensor Cores, reduce instructions)

The boundary occurs at:

$$
I_{\text{ridge}} = \frac{\text{Peak Compute}}{\text{Peak Bandwidth}}
$$

This is the **ridge point** — kernels with $I < I_{\text{ridge}}$ are memory-bound; $I > I_{\text{ridge}}$ are compute-bound.

### H100 Example Calculation

**H100 Specs** (TF32):
- Peak Compute: 989 TFLOPS = $989 \times 10^{12}$ FLOPs/s
- Peak Bandwidth: 3.35 TB/s = $3.35 \times 10^{12}$ Bytes/s

**Ridge Point**:
$$
I_{\text{ridge}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOPs/Byte}
$$

**Interpretation**: Kernels with arithmetic intensity below 295 FLOPs/Byte are memory-bound on H100.

### Practical Examples

**Example 1: Element-wise Vector Addition**

```cuda
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**Analysis**:
- **FLOPs**: 1 add per element = $n$ FLOPs
- **Memory**: Read $a$ ($4n$ bytes), read $b$ ($4n$ bytes), write $c$ ($4n$ bytes) = $12n$ bytes
- **Arithmetic Intensity**: $I = \frac{n}{12n} = 0.083$ FLOPs/Byte

**Conclusion**: Extremely memory-bound ($0.083 \ll 295$). On H100, achievable performance:
$$
\text{Performance} = 3.35 \text{ TB/s} \times 0.083 \text{ FLOPs/Byte} = 278 \text{ GFLOPS}
$$

This is only **0.028%** of peak compute! Optimization should focus on memory coalescing, eliminating redundant loads, or fusing with other operations.

**Example 2: Dense Matrix Multiplication (GEMM)**

```cuda
// Simplified; real implementations use tiling and shared memory
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

**Analysis** (for $C = A \times B$, all $N \times N$):
- **FLOPs**: $2N^3$ (multiply-add for each of $N^2$ output elements, each summing $N$ products)
- **Memory**: Read $A$ ($4N^2$ bytes), read $B$ ($4N^2$ bytes), write $C$ ($4N^2$ bytes) = $12N^2$ bytes (assuming no reuse; real implementations reuse via cache/shared memory)
- **Arithmetic Intensity**: $I = \frac{2N^3}{12N^2} = \frac{N}{6}$ FLOPs/Byte

**For $N = 8192$**:
$$
I = \frac{8192}{6} \approx 1365 \text{ FLOPs/Byte}
$$

**Conclusion**: Compute-bound ($1365 > 295$). Performance approaches peak:
$$
\text{Performance} \approx 989 \text{ TFLOPS} \times \text{efficiency factor}
$$

With optimized implementations (cuBLAS, using Tensor Cores), efficiency factor can exceed 90%, achieving 800+ TFLOPS.

### Visualizing the Roofline

```
Performance
    │
    │                       ┌─────────────── Compute Bound (Flat Roof)
    │                      ╱
989 │                     ╱
TFLOPS                   ╱
    │                   ╱
    │                  ╱
    │                 ╱ Memory Bound (Diagonal, slope = bandwidth)
    │                ╱
    │               ╱
    │              ╱
    │             ╱
    │            ╱
    └───────────┴───────────────────────────────────────────────────► 
               295                                    Arithmetic Intensity
           (Ridge Point)                                 (FLOPs/Byte)
```

**How to use**:
1. Calculate your kernel's arithmetic intensity
2. Plot it on the x-axis
3. If left of ridge point: memory-bound, optimize data movement
4. If right of ridge point: compute-bound, optimize computation

### Optimization Strategies by Regime

**Memory-Bound Kernels**:
- Ensure coalesced memory access
- Use shared memory for data reuse
- Minimize global memory transactions
- Increase arithmetic intensity (fuse operations)
- Compress data (use FP16 instead of FP32)

**Compute-Bound Kernels**:
- Use specialized hardware (Tensor Cores for matmuls)
- Reduce instruction count (use intrinsics, SFUs)
- Improve instruction-level parallelism
- Ensure high occupancy (many active warps)

---

## GPU Memory Architecture

Memory hierarchy is perhaps the most critical aspect of GPU performance. Understanding the hierarchy — its capacity, bandwidth, latency, and semantics — is essential for writing efficient GPU code.

### The Memory Hierarchy

Modern GPUs have a multi-level memory hierarchy, each with different characteristics:

<div class="diagram">
<div class="diagram-title">GPU Memory Hierarchy (Fastest → Slowest)</div>
<div class="flow">
<div class="flow-node accent wide">Registers<br/>(Per-thread, 256 KB/SM)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node purple wide">Shared Memory / L1 Cache<br/>(Per-SM, 128 KB, Programmer-Managed)</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">L2 Cache<br/>(Device-wide, 40-60 MB, Hardware-Managed)</div>
<div class="flow-arrow orange"></div>
<div class="flow-node green wide">Global Memory (HBM)<br/>(80 GB, 3.35 TB/s, High Latency)</div>
</div>
</div>

### 1. Registers

**Characteristics**:
- **Scope**: Per-thread
- **Capacity**: 65,536 × 32-bit per SM (256 KB)
- **Latency**: ~1 cycle
- **Bandwidth**: Highest (multiple reads/writes per cycle)
- **Management**: Compiler-managed (automatic)

**Programming**:
```cuda
__global__ void kernel() {
    float x = 3.14f;  // Stored in register
    float y = x * 2.0f;  // Computation uses registers
}
```

**Trade-off**: More registers per thread → Fewer resident threads → Less latency hiding. The register allocation affects **occupancy**:

$$
\text{Max Threads/SM} = \min\left( 2048, \left\lfloor \frac{65536}{\text{registers per thread}} \right\rfloor \right)
$$

**Optimization**: Use `__launch_bounds__` to control register usage:
```cuda
__global__ void __launch_bounds__(256, 4) kernel() { ... }
// 256 threads/block, 4 blocks/SM target
```

### 2. Shared Memory

**Characteristics**:
- **Scope**: Per-block (shared among threads in a block)
- **Capacity**: 128 KB per SM (configurable split with L1)
- **Latency**: ~28 cycles (for a shared memory transaction)
- **Bandwidth**: ~10 TB/s (per SM)
- **Management**: Programmer-managed (explicit allocation)

**Programming**:
```cuda
__global__ void kernel() {
    __shared__ float tile[32][32];  // Allocated in shared memory
    
    // Cooperative loading
    tile[threadIdx.y][threadIdx.x] = input[...];
    __syncthreads();  // Synchronize before using shared data
    
    // Compute using shared data
    float result = tile[threadIdx.y][threadIdx.x] * 2.0f;
}
```

**Use Cases**:
- Tiling for matrix multiplication
- Reusing data across threads (reduction, convolution)
- Inter-thread communication within a block

**Bank Conflicts**: Shared memory is organized into 32 banks. Simultaneous accesses to the same bank (by different threads) serialize. Access `[threadIdx.x]` to avoid conflicts (each thread accesses a different bank).

### 3. L1 Cache

**Characteristics**:
- **Scope**: Per-SM
- **Capacity**: 128 KB (shared with shared memory in configurable ratio)
- **Latency**: ~28 cycles
- **Bandwidth**: Shared with shared memory (~10 TB/s per SM)
- **Management**: Hardware-managed (automatic)

**Programming**: Implicit (hardware manages). Can be configured:
```cuda
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
// Dedicate 100% to shared memory, 0% to L1 cache
```

**Caching Policy**:
- Caches global memory loads
- Write-evict for stores (no write-allocate)
- Can disable L1 caching per load: `__ldg()` intrinsic

### 4. L2 Cache

**Characteristics**:
- **Scope**: Device-wide (shared across all SMs)
- **Capacity**: 40-60 MB (H100 has 50 MB)
- **Latency**: ~200 cycles
- **Bandwidth**: ~10 TB/s (aggregate across all SMs)
- **Management**: Hardware-managed

**Programming**: Implicit. Can configure residency:
```cuda
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = ptr;
attr.accessPolicyWindow.num_bytes = size;
attr.accessPolicyWindow.hitRatio = 1.0f;  // High priority
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

**Persistence**: L2 can be configured to persist data (useful for reused data across kernel launches).

### 5. Global Memory (HBM)

**Characteristics**:
- **Scope**: Device-wide (accessible by all threads)
- **Capacity**: 24-80 GB (consumer to datacenter)
- **Latency**: 400-800 cycles
- **Bandwidth**: 1-3.35 TB/s (depending on generation)
- **Management**: Programmer-allocated (cudaMalloc)

**Programming**:
```cuda
float *d_data;
cudaMalloc(&d_data, N * sizeof(float));
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

kernel<<<blocks, threads>>>(d_data);

cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

**Optimization**:
- **Coalescing**: Ensure contiguous access patterns
- **Alignment**: Align allocations to 128-byte boundaries
- **Prefetching**: Use `__prefetch()` for known access patterns

### Memory Bandwidth and Latency Table

| Memory Type | Scope | Capacity | Latency (cycles) | Bandwidth | Management |
|-------------|-------|----------|------------------|-----------|------------|
| **Registers** | Per-thread | 256 KB/SM | ~1 | ~100 TB/s (per SM) | Compiler |
| **Shared Memory** | Per-block | 128 KB/SM | ~28 | ~10 TB/s (per SM) | Programmer |
| **L1 Cache** | Per-SM | 128 KB/SM | ~28 | ~10 TB/s (per SM) | Hardware |
| **L2 Cache** | Device | 40-60 MB | ~200 | ~10 TB/s (aggregate) | Hardware |
| **Global (HBM)** | Device | 24-80 GB | 400-800 | 1-3.35 TB/s | Programmer |
| **Host Memory** | System | System RAM | 100,000+ | ~25 GB/s (PCIe 4.0 x16) | Programmer |

### Memory Bandwidth Calculation

For HBM, theoretical bandwidth is:

$$
\text{Bandwidth} = \frac{\text{Bus Width (bits)} \times \text{Memory Clock (Hz)} \times 2}{8}
$$

The factor of 2 accounts for double data rate (DDR).

**H100 Example**:
- 5120-bit bus (5 HBM3 stacks × 1024-bit/stack)
- 2.619 GHz effective clock
$$
\text{Bandwidth} = \frac{5120 \times 2.619 \times 10^9 \times 2}{8} = 3.35 \times 10^{12} \text{ Bytes/s} = 3.35 \text{ TB/s}
$$

### Unified Memory

**Unified Memory** (introduced with CUDA 6) allows a single pointer to be accessible from both CPU and GPU:

```cuda
float *data;
cudaMallocManaged(&data, N * sizeof(float));

// Accessible on CPU
for (int i = 0; i < N; i++) data[i] = i;

// Accessible on GPU
kernel<<<blocks, threads>>>(data);

// Accessible on CPU again
printf("%f\n", data[0]);

cudaFree(data);
```

**Implementation**: Page migration + demand paging. When GPU accesses a page resident on CPU (or vice versa), a page fault triggers migration.

**Pros**: Simplified programming, no explicit transfers
**Cons**: Implicit transfers can be slower, page fault overhead

**Use Cases**: Prototyping, sparse access patterns, oversubscription

### Pinned Memory

**Pinned (Page-Locked) Memory** is host memory that cannot be paged out:

```cuda
float *h_data;
cudaMallocHost(&h_data, N * sizeof(float));  // Pinned memory

// Faster transfers to/from GPU
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

cudaFreeHost(h_data);
```

**Advantages**:
- Faster transfers (DMA without intermediate copy)
- Enables asynchronous transfers

**Disadvantages**:
- Reduces available memory for OS paging
- Should be used judiciously

---

## Brief History of GPUs

The GPU's evolution from fixed-function graphics accelerator to general-purpose computing powerhouse is one of modern computing's most remarkable stories.

<div class="diagram">
<div class="diagram-title">NVIDIA GPU Architecture Timeline</div>
<div class="timeline">
<div class="timeline-item">
<div class="timeline-year">1993</div>
<div class="timeline-title">NVIDIA Founded</div>
<div class="timeline-desc">Jensen Huang, Chris Malachowsky, and Curtis Priem found NVIDIA with the vision of accelerating 3D graphics for gaming and professional visualization.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">1999</div>
<div class="timeline-title">GeForce 256 - "The World's First GPU"</div>
<div class="timeline-desc">First chip marketed as a "GPU". Integrated transform and lighting (T&L) on-chip, offloading CPU work. 17M transistors, 120 MHz, 15M triangles/sec, 480M pixels/sec.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2001</div>
<div class="timeline-title">GeForce 3 - Programmable Shaders</div>
<div class="timeline-desc">First consumer GPU with programmable vertex shaders, enabling custom vertex transformations. Beginning of GPU programmability.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2006</div>
<div class="timeline-title">CUDA Launch - The GPGPU Revolution</div>
<div class="timeline-desc">CUDA (Compute Unified Device Architecture) released with GeForce 8800. Exposes GPU as general-purpose parallel processor. C-like programming model, thousands of threads, hierarchical execution.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2008</div>
<div class="timeline-title">Tesla Architecture (GT200)</div>
<div class="timeline-desc">First "compute-optimized" architecture. 240 cores, FP64 support, 1.4 TFLOPS FP32. Targeted at HPC and scientific computing.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2010</div>
<div class="timeline-title">Fermi Architecture (GF100)</div>
<div class="timeline-desc">512 cores, unified cache hierarchy (L1/L2), ECC memory, up to 6 GB GDDR5. First truly viable architecture for datacenter computing. IEEE 754-2008 FP64 compliance.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2012</div>
<div class="timeline-title">Kepler Architecture (GK110)</div>
<div class="timeline-desc">2880 cores, dynamic parallelism (kernels launch kernels), Hyper-Q (32 concurrent streams), 1.3 TFLOPS FP64. Powered early deep learning breakthroughs (AlexNet on ImageNet).</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2014</div>
<div class="timeline-title">Maxwell Architecture (GM200)</div>
<div class="timeline-desc">Focus on energy efficiency. 3072 cores, redesigned SM, 2× performance per watt vs Kepler. Introduced in consumer GTX 900 series.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2016</div>
<div class="timeline-title">Pascal Architecture (GP100)</div>
<div class="timeline-desc">HBM2 memory (732 GB/s), NVLink interconnect (160 GB/s), unified memory improvements, 16nm process. Tesla P100: 5.3 TFLOPS FP64, 10.6 TFLOPS FP32, 21.2 TFLOPS FP16.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2017</div>
<div class="timeline-title">Volta Architecture (GV100) - Tensor Cores Debut</div>
<div class="timeline-desc">First-generation Tensor Cores for mixed-precision matrix multiply-accumulate. 125 TFLOPS with FP16 Tensor Cores. New SM design with independent thread scheduling. Transformer training acceleration.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2018</div>
<div class="timeline-title">Turing Architecture (TU102) - Real-Time Ray Tracing</div>
<div class="timeline-desc">RT Cores for ray tracing, 2nd-gen Tensor Cores with INT8/INT4 support. Mesh shading, variable rate shading. GeForce RTX 2080 Ti: 14.2 TFLOPS, real-time ray tracing in games.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2020</div>
<div class="timeline-title">Ampere Architecture (GA100) - 3rd-Gen Tensor Cores</div>
<div class="timeline-desc">TF32 precision (FP32-like range, FP16-like speed), BF16 support, structural sparsity acceleration (2:4 sparsity). A100: 19.5 TFLOPS FP64, 312 TFLOPS TF32, 624 TFLOPS FP16. Multi-Instance GPU (MIG) partitioning.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2022</div>
<div class="timeline-title">Hopper Architecture (GH100) - 4th-Gen Tensor Cores</div>
<div class="timeline-desc">FP8 Tensor Cores (2× FP16 throughput), Transformer Engine (auto FP8/FP16 casting), 80 GB HBM3 (3.35 TB/s), Thread Block Clusters, asynchronous execution. H100: 60 TFLOPS FP64, 989 TFLOPS TF32, 3958 TFLOPS FP8. Powers ChatGPT, GPT-4 training.</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2024</div>
<div class="timeline-title">Blackwell Architecture (B100/B200) - Next Frontier</div>
<div class="timeline-desc">Multi-chip module design (two dies connected via 10 TB/s chip-to-chip interconnect), 208 billion transistors, 20 petaFLOPS FP4 (192 GB HBM3e), 2nd-gen Transformer Engine. Designed for trillion-parameter models.</div>
</div>
</div>
</div>

### Key Milestones Explained

**CUDA (2006)**: The inflection point. Before CUDA, GPGPU required encoding problems as graphics operations (vertices, textures, fragment shaders). CUDA provided a straightforward parallel programming model with threads, blocks, grids, and shared memory. This unlocked GPU computing for scientific researchers and enabled the deep learning revolution a decade later.

**Fermi (2010)**: Made GPUs viable for HPC. ECC memory, IEEE-compliant FP64, cache hierarchy, and better debugging support meant scientists could trust GPUs for production workloads. Early adopters: molecular dynamics (AMBER, GROMACS), climate modeling, astrophysics.

**Kepler (2012)**: Powered the AlexNet moment (2012 ImageNet competition), where GPUs trained a deep CNN faster than CPU-based approaches. This catalyzed deep learning research. Kepler's dynamic parallelism also enabled recursive algorithms on GPU.

**Volta (2017)**: Tensor Cores were a paradigm shift — specialized hardware for the matrix multiplications dominating deep learning. A100 delivered 125 TFLOPS for mixed-precision training, 10× faster than Pascal for deep learning. This timing aligned with the Transformer revolution (2017's "Attention Is All You Need"), making training BERT, GPT feasible.

**Ampere (2020)**: TF32 eliminated the need for manual mixed-precision code — automatic speedup for FP32 code via Tensor Cores. Structural sparsity doubled effective throughput for sparse networks. MIG allowed partitioning a single GPU into 7 instances, improving datacenter utilization.

**Hopper (2022)**: FP8 doubled Tensor Core throughput again, enabling training models 2× larger or 2× faster. Transformer Engine dynamically manages FP8/FP16 precision per layer, crucial for maintaining accuracy with extreme quantization. H100 powers the current generative AI boom (GPT-4, Claude, Gemini).

**Blackwell (2024)**: Multi-chip scaling breaks reticle limits. Two dies act as one GPU with coherent memory, effectively doubling compute density. 20 petaFLOPS FP4 targets inference of trillion-parameter models. Represents NVIDIA's bet on continued scaling through chiplet architectures.

### Performance Scaling Over Time

| Year | Architecture | Process | Transistors | FP32 Peak | FP64 Peak | Memory BW | Landmark Use Case |
|------|--------------|---------|-------------|-----------|-----------|-----------|-------------------|
| 2008 | Tesla (GT200) | 65nm | 1.4B | 1.4 TFLOPS | 0.08 TFLOPS | 141 GB/s | Early CUDA adoption |
| 2010 | Fermi (GF100) | 40nm | 3.0B | 1.3 TFLOPS | 0.5 TFLOPS | 144 GB/s | HPC viability |
| 2012 | Kepler (GK110) | 28nm | 7.1B | 4.3 TFLOPS | 1.3 TFLOPS | 288 GB/s | AlexNet training |
| 2016 | Pascal (GP100) | 16nm | 15.3B | 10.6 TFLOPS | 5.3 TFLOPS | 732 GB/s | VR, autonomous driving |
| 2017 | Volta (GV100) | 12nm | 21.1B | 15.7 TFLOPS | 7.8 TFLOPS | 900 GB/s | Transformer training |
| 2020 | Ampere (GA100) | 7nm | 54.2B | 19.5 TFLOPS | 9.7 TFLOPS | 1555 GB/s | GPT-3 training |
| 2022 | Hopper (GH100) | 4nm | 80B | 60 TFLOPS | 30 TFLOPS | 3350 GB/s | GPT-4, LLaMA-2 |
| 2024 | Blackwell (B200) | 4nm | 208B | ~80 TFLOPS | ~40 TFLOPS | ~8000 GB/s* | Trillion-param models |

*Blackwell bandwidth is aggregate across dual-chip module.

**Observation**: FP32 performance grew ~60× from Tesla to Hopper (15 years). But specialized compute (Tensor Cores) grew 1000×+ for matrix operations, reflecting hardware-software co-design for deep learning.

---

## Summary and Key Takeaways

This chapter established the foundational knowledge of GPU architecture and design philosophy:

1. **GPUs trade single-thread latency for aggregate throughput**, dedicating 80% of die area to ALUs vs CPU's 20%, enabled by amortizing control logic across many cores via SIMT execution.

2. **Massive parallelism is the solution to the power wall**. With frequency scaling dead, only parallel architectures scale performance. Gustafson's Law shows this enables solving qualitatively larger problems.

3. **The memory hierarchy is critical**. From registers (1-cycle, 256 KB) to HBM (400-cycle, 80 GB), understanding which memory to use and how to access it efficiently dominates performance.

4. **Data type choice matters**. FP8 provides 2× throughput vs FP16, which provides 2× vs FP32. Choosing the right precision can mean 4× performance gains with negligible accuracy loss.

5. **The Roofline Model guides optimization**. Calculate arithmetic intensity; if below ridge point, optimize memory; if above, optimize compute. Don't prematurely optimize the wrong thing.

6. **GPU evolution is hardware-software co-design**. Tensor Cores emerged to accelerate deep learning; FP8 emerged for LLM inference. Architecture responds to workload demands.

As we proceed through this guide, these foundational concepts will reappear in deeper, more specific contexts. Mastering them now provides the lens through which all subsequent GPU programming and optimization decisions become clear.

**Next: [Chapter 2 — GPU Microarchitecture →](./02_gpu_microarchitecture.md)**

---

*Last updated: April 2026*
