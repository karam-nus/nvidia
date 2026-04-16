---
title: "Chapter 8 — GPU Generations"
---

[← Back to Table of Contents](./README.md)

# Chapter 8: GPU Generations — The Evolution of Parallel Computing Power

*"Moore's Law is dead. Long live Huang's Law."* — The GPU industry, apparently

If CPU evolution is like watching a caterpillar slowly transform into a slightly faster caterpillar, GPU evolution is like watching a rocket ship upgrade to a warp drive while simultaneously learning to paint and compose poetry. Over the past two decades, NVIDIA has released a succession of GPU architectures that have fundamentally transformed computing, each generation bringing revolutionary capabilities that seemed impossible just years before.

This chapter traces the architectural lineage from the humble Tesla architecture (2006) through the mind-bending complexity of Blackwell (2024), examining not just *what* changed, but *why* it mattered, *which* applications benefited, *when* innovations appeared, *who* drove the designs, and *where* we're headed.

## The Architecture Evolution Timeline

Let's start with the big picture: NVIDIA's GPU architecture evolution over nearly two decades.

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2006</div>
    <div class="timeline-title">Tesla Architecture</div>
    <div class="timeline-desc">First unified shader architecture. CUDA born. G80 GPU with 128 CUDA cores. The beginning of GPGPU computing.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2010</div>
    <div class="timeline-title">Fermi Architecture</div>
    <div class="timeline-desc">True IEEE 754 compliance, ECC memory, unified cache hierarchy. First "serious" compute GPU (GF100).</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2012</div>
    <div class="timeline-title">Kepler Architecture</div>
    <div class="timeline-desc">SMX units, dynamic parallelism, Hyper-Q. 3× power efficiency vs Fermi. K20/K80 dominated HPC.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2014</div>
    <div class="timeline-title">Maxwell Architecture</div>
    <div class="timeline-desc">SM redesign for efficiency. 2× perf/watt over Kepler. Focused on gaming (GTX 900 series).</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2016</div>
    <div class="timeline-title">Pascal Architecture</div>
    <div class="timeline-desc">16nm FinFET, HBM2, NVLink, unified memory. P100 was the first deep learning powerhouse.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2017</div>
    <div class="timeline-title">Volta Architecture</div>
    <div class="timeline-desc">🎯 TENSOR CORES debut! Specialized mixed-precision units for AI. V100 changed everything.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2018</div>
    <div class="timeline-title">Turing Architecture</div>
    <div class="timeline-desc">RT Cores for ray tracing. 2nd-gen Tensor Cores. RTX brand born. Real-time ray tracing arrives.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2020</div>
    <div class="timeline-title">Ampere Architecture</div>
    <div class="timeline-desc">3rd-gen Tensor Cores, 2nd-gen RT Cores, sparse tensor ops. A100 with 6912 CUDA cores, RTX 30 series.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2022</div>
    <div class="timeline-title">Ada Lovelace Architecture</div>
    <div class="timeline-desc">3rd-gen RT Cores (opacity micromaps), 4th-gen Tensor Cores, DLSS 3 with frame generation. RTX 40 series.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2022</div>
    <div class="timeline-title">Hopper Architecture</div>
    <div class="timeline-desc">4th-gen Tensor Cores, FP8 precision, Transformer Engine, NVLink 4.0. H100 with 16,896 CUDA cores.</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2024</div>
    <div class="timeline-title">Blackwell Architecture</div>
    <div class="timeline-desc">2nd-gen Transformer Engine, 5th-gen Tensor Cores, double FP4 precision, 208B transistors. GB200 superchip.</div>
  </div>
</div>

Notice the acceleration in innovation post-2017 when AI workloads became dominant. The cadence went from 2-4 years between architectures to annual releases in some segments.

## The Great Divide: Consumer vs Data Center

Modern NVIDIA operates essentially two parallel product lines, each with distinct priorities:

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Consumer GPUs (GeForce)</div>
    <ul>
      <li><strong>Primary goal:</strong> Gaming performance & visual fidelity</li>
      <li><strong>Key features:</strong> RT Cores, DLSS, high boost clocks</li>
      <li><strong>Memory:</strong> GDDR6/6X/7 (bandwidth over capacity)</li>
      <li><strong>TDP:</strong> 200-450W (thermal constrained)</li>
      <li><strong>Software:</strong> DirectX, Vulkan, game-ready drivers</li>
      <li><strong>Price:</strong> $500-$2000 (consumer market)</li>
      <li><strong>Target:</strong> Gamers, creators, enthusiasts</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">Data Center GPUs (Tesla/A/H-series)</div>
    <ul>
      <li><strong>Primary goal:</strong> Compute throughput & AI performance</li>
      <li><strong>Key features:</strong> Tensor Cores, FP64, multi-GPU scaling</li>
      <li><strong>Memory:</strong> HBM2/3/3e (capacity over cost)</li>
      <li><strong>TDP:</strong> 300-700W (performance at any cost)</li>
      <li><strong>Software:</strong> CUDA, cuDNN, TensorRT, enterprise drivers</li>
      <li><strong>Price:</strong> $10,000-$40,000+ (enterprise pricing)</li>
      <li><strong>Target:</strong> Cloud providers, AI labs, HPC centers</li>
    </ul>
  </div>
</div>

The architectures share DNA but diverge significantly in implementation. A GeForce RTX 4090 and an H100 both use advanced manufacturing and tensor operations, but optimize for completely different workloads.

## Tesla Architecture (2006): The CUDA Revolution

**What:** The first unified shader architecture and birth of CUDA.

**Why it mattered:** Before Tesla, GPUs had separate vertex and pixel shaders. Tesla unified them into a single programmable core, making general-purpose GPU computing (GPGPU) practical.

**Key innovations:**
- Unified shader architecture
- CUDA programming model
- 128 "stream processors" (CUDA cores) in G80
- 384 GFLOPS single precision

The G80 (GeForce 8800 GTX) seems quaint now with its 128 cores, but it represented a paradigm shift. For the first time, developers could write C code that ran on the GPU.

```cuda
// The "Hello World" that started it all (2007)
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

This simple kernel, running on 128 cores in parallel, was more revolutionary than it looked.

## Fermi Architecture (2010): Computing Gets Serious

**What:** The first GPU designed explicitly for both gaming and scientific computing.

**Why it mattered:** Fermi (GF100) brought credibility to GPU computing in HPC with true IEEE 754 compliance, ECC memory, and a cache hierarchy.

<div class="diagram">
<div class="diagram-title">Fermi SM Architecture</div>
<div class="layer-stack">
  <div class="layer accent">L2 Cache (768 KB shared)</div>
  <div class="layer green">16 SM Units × 32 CUDA Cores = 512 cores total</div>
  <div class="layer blue">Each SM: 64KB configurable L1/shared memory</div>
  <div class="layer purple">Dual warp scheduler per SM</div>
  <div class="layer orange">16 Load/Store Units per SM</div>
  <div class="layer yellow">4 Special Function Units per SM</div>
</div>
</div>

**Key innovations:**
- **True IEEE 754-2008 compliance:** Correct rounding, subnormal numbers, exceptions
- **ECC memory:** Error correction for reliability
- **L1/L2 cache hierarchy:** Programmable 16KB/48KB or 48KB/16KB L1/shared split
- **Concurrent kernel execution:** Multiple kernels running simultaneously
- **64 KB register file per SM:** Massive register capacity

**Specifications (GF100):**
- 512 CUDA cores (16 SMs × 32 cores)
- 1.5 TFLOPS FP32, 750 GFLOPS FP64 (1:2 ratio)
- 384-bit GDDR5, 177 GB/s bandwidth
- 40nm process, 3 billion transistors
- TDP: 250W

The downside? Fermi ran **hot**. The GTX 480 became legendary for its thermal output, earning nicknames like "thermonuclear furnace." But it proved GPUs could handle serious scientific workloads.

## Kepler Architecture (2012): Efficiency and Scale

**What:** A major efficiency redesign with new SMX units and dynamic parallelism.

**Why it mattered:** 3× performance per watt over Fermi, enabling GPU computing at scale.

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">⚡</div>
    <div class="card-title">SMX Units</div>
    <div class="card-desc">192 CUDA cores per SM (up from 32), quad warp schedulers</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔄</div>
    <div class="card-title">Dynamic Parallelism</div>
    <div class="card-desc">GPU kernels can launch other kernels without CPU intervention</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🚀</div>
    <div class="card-title">Hyper-Q</div>
    <div class="card-desc">32 hardware work queues (up from 1), better concurrency</div>
  </div>
</div>

**Key innovations:**
- **SMX (Streaming Multiprocessor eXtreme):** 192 cores per SM
- **Dynamic Parallelism:** Recursive kernel launches
- **Hyper-Q:** 32 simultaneous hardware queues
- **GPU Boost:** Dynamic clock adjustments based on power/thermal
- **Grid Management Unit:** Hardware-managed work scheduling

**Specifications (K20X):**
- 2688 CUDA cores (14 SMX × 192 cores)
- 3.95 TFLOPS FP32, 1.31 TFLOPS FP64 (1:3 ratio)
- 384-bit GDDR5, 250 GB/s bandwidth
- 28nm process, 7.1 billion transistors
- TDP: 235W

The K20 and K80 dominated HPC for years. Many of the early deep learning breakthroughs (AlexNet, etc.) ran on Kepler GPUs.

## Maxwell Architecture (2014): The Efficiency King

**What:** Complete SM redesign focused on power efficiency for gaming.

**Why it mattered:** 2× performance per watt over Kepler, making high-end gaming accessible.

Maxwell represented a philosophical shift: instead of adding more cores, redesign the cores to do more with less power.

<div class="flow">
  <div class="flow-node accent wide">Kepler SMX: 192 cores, shared control logic</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Maxwell SMM: 4× 32-core partitions, dedicated control per partition</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">Result: Better utilization, lower power, higher clocks</div>
</div>

**Key innovations:**
- **SMM (Streaming Multiprocessor Maxwell):** Partitioned design
- **Improved scheduling:** Less wasted work
- **Memory compression:** Delta color compression (25% effective bandwidth increase)
- **HDMI 2.0, DisplayPort 1.2:** Modern display outputs

**Specifications (GTX 980):**
- 2048 CUDA cores (16 SMM × 128 cores)
- 5.0 TFLOPS FP32
- 256-bit GDDR5, 224 GB/s bandwidth
- 28nm process, 5.2 billion transistors
- TDP: 165W (remarkable for the performance)

Maxwell proved you didn't need brute force. The GTX 750 Ti, with just 640 cores, delivered GTX 660 performance at 60W TDP.

## Pascal Architecture (2016): The Deep Learning Catalyst

**What:** 16nm FinFET, HBM2, NVLink, unified memory — a complete revolution.

**Why it mattered:** Pascal was the first architecture designed with deep learning as a primary workload. The P100 became the engine of the AI boom.

<div class="diagram">
<div class="diagram-title">Pascal Innovation Stack</div>
<div class="layer-stack">
  <div class="layer accent">16nm FinFET (first in GPU, 2× density)</div>
  <div class="layer green">HBM2 Memory (720 GB/s, 3× bandwidth)</div>
  <div class="layer blue">NVLink 1.0 (160 GB/s GPU-GPU, 5× vs PCIe 3.0)</div>
  <div class="layer purple">Unified Memory with Page Migration</div>
  <div class="layer orange">SM Architecture (64 cores per SM)</div>
  <div class="layer yellow">FP16 × 2 (packed half-precision, 2× throughput)</div>
</div>
</div>

**Key innovations:**
- **16nm FinFET:** First major process shrink in years
- **HBM2 memory:** 4096-bit bus, 720 GB/s bandwidth
- **NVLink:** High-speed GPU interconnect
- **FP16 acceleration:** 2:1 ratio vs FP32 (21.2 TFLOPS FP16!)
- **Unified Memory:** Automatic page migration
- **Preemption:** Fine-grained context switching

**Specifications (P100):**
- 3584 CUDA cores (56 SMs × 64 cores)
- 10.6 TFLOPS FP32, 5.3 TFLOPS FP64, **21.2 TFLOPS FP16**
- 4096-bit HBM2, 720 GB/s bandwidth
- 16nm process, 15.3 billion transistors
- TDP: 300W

The FP16 performance was game-changing. Deep learning frameworks could train models 2× faster by using half precision. The P100 powered the first wave of large-scale deep learning deployments.

## Volta Architecture (2017): Tensor Cores Change Everything

**What:** The introduction of Tensor Cores — specialized matrix multiplication units.

**Why it mattered:** This is where NVIDIA went from "also good at AI" to "completely dominant in AI." Tensor Cores provided 12× FP16 performance leap for deep learning.

<div class="diagram">
<div class="diagram-title">Volta SM with Tensor Cores</div>
<div class="flow">
  <div class="flow-node accent wide">64 FP32 CUDA Cores (traditional)</div>
  <div class="flow-node green wide">64 INT32 Cores (separate from FP32!)</div>
  <div class="flow-node blue wide">8 Tensor Cores (4×4×4 matrix-multiply-accumulate)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Combined: Massive mixed-precision throughput</div>
</div>
</div>

Each Tensor Core performs:

$$
D = A \times B + C
$$

where $A$, $B$, $C$, $D$ are $4 \times 4$ matrices, completing 64 fused multiply-add operations in a single clock cycle.

**Tensor Core operation (1st generation):**
- Input: FP16 matrices $A$ (4×4), $B$ (4×4)
- Accumulator: FP32 matrix $C$ (4×4)
- Output: FP32 matrix $D$ (4×4)
- **Throughput:** 64 FP16 FMAs per clock per Tensor Core

**Key innovations:**
- **Tensor Cores:** 8 per SM, 640 total in V100
- **Independent thread scheduling:** Fine-grained synchronization
- **New SM design:** Separate FP32 and INT32 datapaths
- **HBM2 at 900 GB/s:** Faster memory subsystem
- **NVLink 2.0:** 300 GB/s bidirectional

**Specifications (V100):**
- 5120 CUDA cores (80 SMs × 64 cores)
- 15.7 TFLOPS FP32, 7.8 TFLOPS FP64
- **125 TFLOPS FP16 Tensor Core performance** 🚀
- 4096-bit HBM2, 900 GB/s bandwidth
- 12nm FFN process, 21.1 billion transistors
- TDP: 300W

The V100's 125 TFLOPS for matrix operations was **8× faster** than P100 for deep learning. This wasn't incremental improvement; it was a paradigm shift.

```python
# The operation that Tensor Cores accelerate
# Matrix multiplication in PyTorch
import torch

# FP16 inputs, FP32 accumulation (Tensor Core optimal)
A = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')
B = torch.randn(8192, 8192, dtype=torch.float16, device='cuda')

# This will automatically use Tensor Cores on V100+
C = torch.matmul(A, B)  # 125 TFLOPS vs 15 TFLOPS on CUDA cores

# The speedup is dramatic for transformer attention:
# Q, K, V matrices multiplication happens at Tensor Core speeds
```

## Turing Architecture (2018): Real-Time Ray Tracing Arrives

**What:** RT Cores for ray tracing, 2nd-gen Tensor Cores, RTX brand launch.

**Why it mattered:** Real-time ray tracing transitioned from "research curiosity" to "shippable feature." DLSS (Deep Learning Super Sampling) used Tensor Cores for upscaling.

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🎨</div>
    <div class="card-title">RT Cores (1st Gen)</div>
    <div class="card-desc">
      <strong>BVH Traversal & Ray-Triangle Intersection</strong><br>
      • 10 Giga Rays/sec (RTX 2080 Ti)<br>
      • Hardware-accelerated BVH traversal<br>
      • Programmable hit/miss shaders
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🧠</div>
    <div class="card-title">Tensor Cores (2nd Gen)</div>
    <div class="card-desc">
      <strong>AI Inference for DLSS</strong><br>
      • INT8 & INT4 precision support<br>
      • 114 TFLOPS FP16 (RTX 2080 Ti)<br>
      • DLSS neural network upscaling
    </div>
  </div>
</div>

**RT Core functionality:**

An RT Core is a fixed-function unit that accelerates:

1. **BVH (Bounding Volume Hierarchy) traversal:** Walking the spatial acceleration structure
2. **Ray-triangle intersection:** Testing if a ray hits a triangle

Traditional rasterization: $O(n)$ per pixel (check each triangle)  
RT Core acceleration: $O(\log n)$ per ray (hierarchical traversal)

**Key innovations:**
- **RT Cores:** Dedicated ray tracing acceleration
- **2nd-gen Tensor Cores:** INT8, INT4 support for inference
- **GDDR6 memory:** Next-gen VRAM
- **Turing SM:** Concurrent FP and INT execution
- **Mesh shaders:** Advanced geometry processing

**Specifications (RTX 2080 Ti):**
- 4352 CUDA cores (68 SMs × 64 cores)
- 68 RT Cores (1 per SM)
- 544 Tensor Cores (8 per SM)
- 13.4 TFLOPS FP32, 114 TFLOPS FP16 Tensor
- 10 Giga Rays/sec
- 352-bit GDDR6, 616 GB/s bandwidth
- 12nm FFN process, 18.6 billion transistors
- TDP: 260W

Turing was primarily a gaming architecture, but the 2nd-gen Tensor Cores made it excellent for AI inference workloads.

## Ampere Architecture (2020): The AI Boom Accelerator

**What:** 3rd-gen Tensor Cores with sparse matrix support, 2nd-gen RT Cores, huge scale-up.

**Why it mattered:** Ampere powered the ChatGPT era. The A100 became the gold standard for training large language models. RTX 30 series brought 4K gaming to the masses.

Let's cover both the data center (A100) and consumer (RTX 30) variants:

### Ampere Data Center: A100

<div class="diagram">
<div class="diagram-title">A100 Tensor Core Memory Unit (TCMU)</div>
<div class="layer-stack">
  <div class="layer accent">108 SMs × 64 CUDA Cores = 6,912 cores</div>
  <div class="layer green">108 SMs × 4 Tensor Cores (3rd gen) = 432 Tensor Cores</div>
  <div class="layer blue">40/80 GB HBM2e @ 1.6/2.0 TB/s bandwidth</div>
  <div class="layer purple">Multi-Instance GPU (MIG): 7 isolated partitions</div>
  <div class="layer orange">Structural Sparsity: 2:4 pattern, 2× throughput</div>
  <div class="layer yellow">TF32: 19.5 TFLOPS, FP16: 312 TFLOPS, INT8: 624 TOPS</div>
</div>
</div>

**3rd-gen Tensor Core precision support:**

| Precision | Use Case | A100 Performance |
|-----------|----------|------------------|
| FP64 | Scientific computing | 9.7 TFLOPS (19.5 with sparsity) |
| TF32 | DL training (default in Ampere) | 156 TFLOPS (312 with sparsity) |
| BFLOAT16 | DL training (wider range than FP16) | 312 TFLOPS (624 with sparsity) |
| FP16 | DL training/inference | 312 TFLOPS (624 with sparsity) |
| INT8 | DL inference | 624 TOPS (1248 with sparsity) |
| INT4 | Extreme inference | 1248 TOPS (2496 with sparsity) |

**Structural Sparsity:** Ampere Tensor Cores can skip zero values in a 2:4 pattern (2 non-zero values in every 4 elements), doubling throughput for sparse networks.

**TF32 (TensorFloat-32):** NVIDIA's secret weapon — FP32 dynamic range with FP16-like performance:

$$
\text{TF32} = \underbrace{\text{1 sign bit}}_{\text{FP32}} + \underbrace{\text{8 exponent bits}}_{\text{FP32}} + \underbrace{\text{10 mantissa bits}}_{\text{FP16-like}}
$$

**A100 Specifications:**
- 6,912 CUDA cores (108 SMs × 64 cores)
- 432 Tensor Cores (3rd gen, 108 SMs × 4)
- 19.5 TFLOPS FP32, 9.7 TFLOPS FP64
- **312 TFLOPS TF32, 624 TFLOPS FP16** (with sparsity)
- 40 GB or 80 GB HBM2e, 1.6 TB/s or 2 TB/s
- 7nm TSMC, 54.2 billion transistors
- TDP: 400W
- **Multi-Instance GPU (MIG):** Partition into 7 independent GPUs
- **NVLink 3.0:** 600 GB/s (12 links)

The A100 80GB with 2 TB/s memory bandwidth became the workhorse of the LLM training era.

### Ampere Consumer: RTX 30 Series

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🎮</div>
    <div class="card-title">RTX 3090 Ti</div>
    <div class="card-desc">
      <strong>Flagship Gaming</strong><br>
      • 10,752 CUDA cores<br>
      • 84 RT Cores (2nd gen)<br>
      • 336 Tensor Cores (3rd gen)<br>
      • 24 GB GDDR6X<br>
      • 40 TFLOPS FP32<br>
      • $1,999 MSRP
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">RTX 3080</div>
    <div class="card-desc">
      <strong>Sweet Spot</strong><br>
      • 8,704 CUDA cores<br>
      • 68 RT Cores<br>
      • 272 Tensor Cores<br>
      • 10/12 GB GDDR6X<br>
      • 29.8 TFLOPS FP32<br>
      • $699 MSRP
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">💚</div>
    <div class="card-title">RTX 3060 Ti</div>
    <div class="card-desc">
      <strong>Value King</strong><br>
      • 4,864 CUDA cores<br>
      • 38 RT Cores<br>
      • 152 Tensor Cores<br>
      • 8 GB GDDR6<br>
      • 16.2 TFLOPS FP32<br>
      • $399 MSRP
    </div>
  </div>
</div>

**Consumer Ampere innovations:**
- **2nd-gen RT Cores:** 2× throughput vs Turing, concurrent RT + shading
- **3rd-gen Tensor Cores:** Same tech as A100, enabling DLSS 2.0
- **GDDR6X memory:** PAM4 signaling, up to 19.5 Gbps (RTX 3090 Ti)
- **RTX IO:** DirectStorage support for fast asset loading
- **AV1 decode:** Hardware-accelerated next-gen video codec

The RTX 30 series became legendary during the cryptocurrency mining boom (and subsequent drought) of 2021-2022. Despite supply issues, it represented a massive generational leap in gaming performance.

## Ada Lovelace Architecture (2022): DLSS 3 and Ray Tracing Maturity

**What:** 3rd-gen RT Cores with opacity micromaps and micro-mesh engines, 4th-gen Tensor Cores, DLSS 3 with frame generation.

**Why it mattered:** Ada brought ray tracing performance to the point where it's truly viable at high resolutions. DLSS 3's frame generation was controversial but impressive.

<div class="diagram">
<div class="diagram-title">Ada Lovelace SM Architecture (RTX 40 Series)</div>
<div class="flow">
  <div class="flow-node accent wide">128 CUDA Cores per SM (FP32)</div>
  <div class="flow-node green wide">4 Tensor Cores (4th gen, FP8 support)</div>
  <div class="flow-node blue wide">1 RT Core (3rd gen, OMM + DMM engines)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">Combined: 2× RT perf, 2× AI perf vs Ampere</div>
</div>
</div>

**3rd-gen RT Core innovations:**

| Feature | Description | Speedup |
|---------|-------------|---------|
| **Opacity Micromap (OMM)** | Encode alpha-tested geometry (foliage) in acceleration structure | 2× for alpha-heavy scenes |
| **Displaced Micro-Mesh (DMM)** | Hardware-accelerated micro-tessellation | 10× geometry detail, no BVH overhead |
| **Shader Execution Reordering (SER)** | Reorganize incoherent ray workloads | 2-3× in complex scenes |

**4th-gen Tensor Cores:**
- **FP8 Transformer Engine:** 2× throughput vs FP16 for transformers
- **Sparsity support:** Maintains 2:4 sparse acceleration
- **DLSS 3 Frame Generation:** AI-generated intermediate frames

**DLSS 3 is wild:** It uses optical flow acceleration hardware and Tensor Cores to generate entirely new frames by interpolating motion between rendered frames. It's not just upscaling anymore; it's frame synthesis.

<div class="flow-h">
  <div class="flow-node accent">Game renders frame N at 60 FPS</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Optical Flow calculates motion vectors</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue">Tensor Cores generate synthetic frames</div>
  <div class="flow-arrow blue"></div>
  <div class="flow-node purple">Display shows 120+ FPS</div>
</div>

### RTX 40 Series Lineup

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">RTX 4090</div>
    <div class="timeline-title">The Absolute Unit</div>
    <div class="timeline-desc">16,384 CUDA cores • 128 RT Cores • 512 Tensor Cores • 24 GB GDDR6X @ 1008 GB/s • 82.6 TFLOPS FP32 • 450W TDP • $1,599</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">RTX 4080</div>
    <div class="timeline-title">High-End Standard</div>
    <div class="timeline-desc">9,728 CUDA cores • 76 RT Cores • 304 Tensor Cores • 16 GB GDDR6X @ 716 GB/s • 48.7 TFLOPS FP32 • 320W TDP • $1,199</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">RTX 4070 Ti</div>
    <div class="timeline-title">Enthusiast Choice</div>
    <div class="timeline-desc">7,680 CUDA cores • 60 RT Cores • 240 Tensor Cores • 12 GB GDDR6X @ 504 GB/s • 40.1 TFLOPS FP32 • 285W TDP • $799</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">RTX 4060 Ti</div>
    <div class="timeline-title">Mainstream Offering</div>
    <div class="timeline-desc">4,352 CUDA cores • 34 RT Cores • 136 Tensor Cores • 8/16 GB GDDR6 @ 288 GB/s • 22.1 TFLOPS FP32 • 160W TDP • $399/$499</div>
  </div>
</div>

**Ada Specifications (AD102 - RTX 4090):**
- 16,384 CUDA cores (128 SMs × 128 cores)
- 128 RT Cores (3rd gen)
- 512 Tensor Cores (4th gen)
- 82.6 TFLOPS FP32, **1321 TFLOPS FP8 Tensor**
- 384-bit GDDR6X @ 21 Gbps, 1008 GB/s bandwidth
- 4nm TSMC, 76.3 billion transistors
- TDP: 450W
- PCIe Gen 4 ×16, HDMI 2.1a, DisplayPort 1.4a

The RTX 4090 is comically overpowered, often GPU-limited by the monitor rather than the game. It's also become popular for local AI inference workloads (24 GB VRAM helps).

## Hopper Architecture (2022): The AI Training Juggernaut

**What:** Purpose-built for transformer models with 4th-gen Tensor Cores, Transformer Engine, FP8 precision, and massive scale.

**Why it mattered:** Hopper is the architecture training GPT-4, Llama, Claude, and nearly every major LLM. The H100 is the most sought-after piece of silicon on Earth (as of 2024).

<div class="diagram">
<div class="diagram-title">Hopper H100 Architecture</div>
<div class="layer-stack">
  <div class="layer accent">132 SMs × 128 CUDA Cores = 16,896 cores (full chip)</div>
  <div class="layer green">132 SMs × 4 Tensor Cores (4th gen) = 528 Tensor Cores</div>
  <div class="layer blue">80 GB HBM3 @ 3.35 TB/s bandwidth (H100 HBM3)</div>
  <div class="layer purple">Transformer Engine: Automatic FP8/FP16 switching</div>
  <div class="layer orange">NVLink 4.0: 900 GB/s (18 links), NVSwitch-based fabrics</div>
  <div class="layer yellow">Thread Block Clusters: New hierarchy level</div>
  <div class="layer pink">DPX Instructions: Dynamic programming acceleration</div>
</div>
</div>

**The Transformer Engine — Hopper's secret weapon:**

Transformers use attention mechanisms with matrix multiplies that have varying numerical precision requirements:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The Transformer Engine analyzes each layer's gradient statistics in real-time and automatically switches between FP8 and FP16:

- **FP8 when safe:** 2× throughput, lower memory usage
- **FP16 when needed:** Prevents numerical issues
- **Automatic scaling:** Per-tensor scaling factors maintained in FP32

**FP8 formats in Hopper:**

| Format | Sign | Exponent | Mantissa | Max Value | Use Case |
|--------|------|----------|----------|-----------|----------|
| **E4M3** | 1 | 4 bits | 3 bits | 448 | Forward pass (higher precision) |
| **E5M2** | 1 | 5 bits | 2 bits | 57,344 | Gradients (higher range) |

**4th-gen Tensor Core capabilities:**

| Precision | H100 Performance (Dense) | H100 Performance (Sparse) |
|-----------|--------------------------|---------------------------|
| FP64 | 34 TFLOPS | 67 TFLOPS |
| TF32 | 378 TFLOPS | 756 TFLOPS |
| BFLOAT16 | 756 TFLOPS | 1,513 TFLOPS |
| FP16 | 756 TFLOPS | 1,513 TFLOPS |
| FP8 | **1,513 TFLOPS** | **3,026 TFLOPS** |
| INT8 | 1,513 TOPS | 3,026 TOPS |

3 **PETAFLOPS** of sparse FP8. Let that sink in.

**H100 Variants:**

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🔥</div>
    <div class="card-title">H100 SXM5 (HBM3)</div>
    <div class="card-desc">
      <strong>Maximum Performance</strong><br>
      • 80 GB HBM3 @ 3.35 TB/s<br>
      • 700W TDP<br>
      • NVLink 4.0 (900 GB/s)<br>
      • For DGX systems, cloud<br>
      • $30,000-40,000
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">💼</div>
    <div class="card-title">H100 PCIe (HBM2e)</div>
    <div class="card-desc">
      <strong>Server Standard</strong><br>
      • 80 GB HBM2e @ 2.0 TB/s<br>
      • 350W TDP<br>
      • PCIe Gen 5 ×16<br>
      • For standard servers<br>
      • $25,000-30,000
    </div>
  </div>
</div>

**H100 Full Specifications (SXM5):**
- 16,896 CUDA cores (132 SMs × 128 cores)
- 528 Tensor Cores (4th gen)
- 67 TFLOPS FP64, 756 TFLOPS TF32, **3,026 TFLOPS FP8 sparse**
- 80 GB HBM3, 3.35 TB/s bandwidth
- 4nm TSMC, 80 billion transistors
- TDP: 700W
- NVLink 4.0: 900 GB/s bidirectional
- Multi-Instance GPU (MIG): Up to 7 partitions

**Additional Hopper innovations:**

1. **Thread Block Clusters:** A new CUDA hierarchy level above thread blocks, enabling efficient cooperation between blocks
2. **DPX instructions:** Accelerate dynamic programming algorithms (Smith-Waterman, etc.) by 7×
3. **Confidential Computing:** Hardware-based TEE (Trusted Execution Environment) for secure multi-tenant AI
4. **Asynchronous Transaction Barrier:** New synchronization primitive

```cuda
// Thread Block Cluster example (Hopper+)
#include <cuda/barrier>

__global__ void cluster_kernel() {
    // Threads can now synchronize across blocks in a cluster
    __cluster_dims__(2, 2, 1);  // 2×2 cluster of thread blocks
    
    // Shared memory visible across cluster
    __shared__ int data[256];
    
    // Cluster-wide barrier
    cluster.sync();
}
```

### H200: Hopper with HBM3e

In late 2023, NVIDIA announced the H200 — essentially H100 with upgraded memory:

- **141 GB HBM3e** (up from 80 GB, 76% increase)
- **4.8 TB/s bandwidth** (up from 3.35 TB/s, 43% increase)
- Same compute capabilities as H100

This matters enormously for LLMs, where memory capacity directly determines the maximum model size you can serve.

**Model capacity comparison:**

| Model | Precision | H100 (80GB) | H200 (141GB) |
|-------|-----------|-------------|--------------|
| GPT-3 175B | FP16 | ✅ Fits | ✅ Fits |
| Llama 2 70B | FP16 | ✅ Fits | ✅ Fits |
| Llama 3 405B | FP16 | ❌ Requires multi-GPU | ✅ Fits on single GPU |
| GPT-4 scale (~1.7T) | FP8 | ❌ Many GPUs | ❌ Still multi-GPU, but fewer |

The H200 enables serving larger models on fewer GPUs, reducing latency and cost.

## Blackwell Architecture (2024): The Latest Frontier

**What:** 2nd-generation Transformer Engine, 5th-gen Tensor Cores, double FP4 precision, and the GB200 Grace Blackwell Superchip combining CPU+GPU.

**Why it matters:** Blackwell is designed for the post-ChatGPT era of trillion-parameter models and real-time AI inference at scale.

<div class="diagram">
<div class="diagram-title">Blackwell B200 Architecture</div>
<div class="layer-stack">
  <div class="layer accent">208 Billion Transistors (2× H100)</div>
  <div class="layer green">5th-Gen Tensor Cores: FP4, FP6, FP8 support</div>
  <div class="layer blue">192 GB HBM3e @ 8 TB/s bandwidth</div>
  <div class="layer purple">2nd-Gen Transformer Engine: Advanced dynamic scaling</div>
  <div class="layer orange">NVLink 5.0: 1.8 TB/s (double H100)</div>
  <div class="layer yellow">RAS Engine: Advanced reliability features</div>
  <div class="layer pink">Secure AI: Confidential computing for AI workloads</div>
</div>
</div>

**5th-gen Tensor Core innovations:**

| Precision | B200 Performance | vs H100 |
|-----------|------------------|---------|
| FP64 | 45 TFLOPS | 1.3× |
| TF32 | 4.5 PETAFLOPS | 6× |
| FP16 | 9 PETAFLOPS | 6× |
| FP8 | 18 PETAFLOPS | 6× |
| **FP4** | **36 PETAFLOPS** | **New!** |
| INT8 | 18 PETAOPS | 6× |

**36 PETAFLOPS** of FP4. We've gone from GFLOPS to TFLOPS to PETAFLOPS in 18 years.

**FP4 format:** 
- 1 sign bit, 2 exponent bits, 1 mantissa bit
- Extreme quantization for inference
- Range: ±0.5 to ±6.0
- Used with careful calibration for LLM inference

**GB200 Grace Blackwell Superchip:**

The GB200 combines two B200 GPUs with an NVIDIA Grace CPU in a single package:

<div class="flow">
  <div class="flow-node accent wide">Grace CPU (ARM Neoverse, 72 cores)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">900 GB/s coherent CPU-GPU link (NVLink-C2C)</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">2× B200 GPUs connected via NVLink 5.0</div>
  <div class="flow-arrow blue"></div>
  <div class="flow-node purple wide">Combined: 1.4 exaFLOPS FP4 per rack (72× GB200)</div>
</div>

**GB200 System Specs:**
- **CPU:** 72-core Grace (ARM Neoverse V2)
- **GPU:** 2× B200 (208B transistors each)
- **CPU-GPU Link:** 900 GB/s NVLink-C2C
- **GPU-GPU Link:** 1.8 TB/s NVLink 5.0
- **Memory:** 480 GB Grace LPDDR5X + 384 GB HBM3e
- **Total System Bandwidth:** 16 TB/s
- **Power:** ~1200W (system)

**Blackwell Rack (GB200 NVL72):**
- 36× GB200 Superchips (72 GPUs + 36 CPUs)
- 720 petaFLOPS FP8, **1.4 exaFLOPS FP4**
- 13.8 TB total HBM3e
- 130 TB/s aggregate bandwidth
- NVLink Switch System: All-to-all GPU connectivity
- 120 kW power consumption

**2nd-Gen Transformer Engine improvements:**

1. **Fine-grained dynamic range management:** Per-block scaling instead of per-tensor
2. **FP8 GEMM with FP16 accumulation:** Better accuracy without sacrificing speed
3. **Automatic mixed precision:** Extends to FP4/FP6 for inference
4. **Distributed training optimizations:** Better overlap of compute/communication

**RAS (Reliability, Availability, Serviceability) Engine:**
- Hardware fault detection and isolation
- Predictive failure analysis
- Live GPU migration (move workloads off failing GPU)
- Critical for massive 10,000+ GPU clusters

**Blackwell Consumer GPUs (RTX 50 series):**

As of early 2025, Blackwell consumer GPUs have been announced:

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">👑</div>
    <div class="card-title">RTX 5090</div>
    <div class="card-desc">
      <strong>Flagship Beast</strong><br>
      • 21,760 CUDA cores (est.)<br>
      • 170 RT Cores (4th gen)<br>
      • 680 Tensor Cores (5th gen)<br>
      • 32 GB GDDR7<br>
      • ~120 TFLOPS FP32<br>
      • 600W TDP
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">RTX 5080</div>
    <div class="card-desc">
      <strong>High Performance</strong><br>
      • 15,360 CUDA cores (est.)<br>
      • 120 RT Cores<br>
      • 480 Tensor Cores<br>
      • 16 GB GDDR7<br>
      • ~80 TFLOPS FP32<br>
      • 400W TDP
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🎯</div>
    <div class="card-title">RTX 5070 Ti</div>
    <div class="card-desc">
      <strong>Enthusiast</strong><br>
      • 11,264 CUDA cores (est.)<br>
      • 88 RT Cores<br>
      • 352 Tensor Cores<br>
      • 16 GB GDDR7<br>
      • ~60 TFLOPS FP32<br>
      • 320W TDP
    </div>
  </div>
</div>

**4th-gen RT Cores (Blackwell RTX):**
- Neural Radiance Cache (AI-accelerated GI)
- Improved SER (Shader Execution Reordering)
- Path guiding using AI

**DLSS 4:** Continues frame generation with improved quality and lower latency.

## Memory Evolution Across Generations

Memory has evolved as dramatically as compute. Let's trace the journey:

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">GDDR5</div>
    <div class="timeline-title">2008-2016</div>
    <div class="timeline-desc">
      <strong>Bandwidth:</strong> 7-8 Gbps per pin<br>
      <strong>Example:</strong> GTX 980 Ti (384-bit, 336 GB/s)<br>
      <strong>Tech:</strong> Prefetch architecture, quad data rate
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">GDDR5X</div>
    <div class="timeline-title">2016-2018</div>
    <div class="timeline-desc">
      <strong>Bandwidth:</strong> 10-14 Gbps per pin<br>
      <strong>Example:</strong> GTX 1080 Ti (352-bit, 484 GB/s)<br>
      <strong>Tech:</strong> QDR to ODR transition (octal data rate)
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">GDDR6</div>
    <div class="timeline-title">2018-2022</div>
    <div class="timeline-desc">
      <strong>Bandwidth:</strong> 14-16 Gbps per pin<br>
      <strong>Example:</strong> RTX 3070 (256-bit, 448 GB/s)<br>
      <strong>Tech:</strong> Lower voltage (1.35V), higher efficiency
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">GDDR6X</div>
    <div class="timeline-title">2020-2024</div>
    <div class="timeline-desc">
      <strong>Bandwidth:</strong> 19-23 Gbps per pin<br>
      <strong>Example:</strong> RTX 4090 (384-bit, 1008 GB/s)<br>
      <strong>Tech:</strong> PAM4 signaling (4 levels vs 2)
    </div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">GDDR7</div>
    <div class="timeline-title">2024+</div>
    <div class="timeline-desc">
      <strong>Bandwidth:</strong> 28-32 Gbps per pin<br>
      <strong>Example:</strong> RTX 5090 (512-bit, 1536 GB/s)<br>
      <strong>Tech:</strong> PAM3 signaling, improved power efficiency
    </div>
  </div>
</div>

**HBM (High Bandwidth Memory) evolution:**

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">HBM2 (2016-2020)</div>
    <ul>
      <li><strong>Width:</strong> 4096-bit bus (stacked dies)</li>
      <li><strong>Bandwidth:</strong> 720-900 GB/s</li>
      <li><strong>Capacity:</strong> Up to 32 GB</li>
      <li><strong>Example:</strong> V100 (900 GB/s)</li>
      <li><strong>Power:</strong> ~20-25W per stack</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">HBM2e (2020-2022)</div>
    <ul>
      <li><strong>Width:</strong> 4096-bit bus</li>
      <li><strong>Bandwidth:</strong> 1.6-2.0 TB/s</li>
      <li><strong>Capacity:</strong> Up to 80 GB</li>
      <li><strong>Example:</strong> A100 80GB (2.0 TB/s)</li>
      <li><strong>Power:</strong> Similar to HBM2</li>
    </ul>
  </div>
</div>

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">HBM3 (2022-2024)</div>
    <ul>
      <li><strong>Width:</strong> 6144-bit bus (wider)</li>
      <li><strong>Bandwidth:</strong> 3.0-3.35 TB/s</li>
      <li><strong>Capacity:</strong> Up to 80 GB</li>
      <li><strong>Example:</strong> H100 (3.35 TB/s)</li>
      <li><strong>Power:</strong> Better efficiency than HBM2e</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">HBM3e (2024+)</div>
    <ul>
      <li><strong>Width:</strong> 6144-bit+ bus</li>
      <li><strong>Bandwidth:</strong> 4.8-8.0 TB/s</li>
      <li><strong>Capacity:</strong> Up to 192 GB</li>
      <li><strong>Example:</strong> B200 (8 TB/s), H200 (4.8 TB/s)</li>
      <li><strong>Power:</strong> Optimized for AI workloads</li>
    </ul>
  </div>
</div>

**Memory bandwidth scaling:**

$$
\text{Bandwidth} = \text{Bus Width (bits)} \times \text{Transfer Rate (GT/s)} \times \frac{1}{8 \text{ bits/byte}}
$$

Example (RTX 4090):
$$
384\text{-bit} \times 21 \text{ Gbps} \times \frac{1}{8} = 1008 \text{ GB/s}
$$

Example (H100 HBM3):
$$
6144\text{-bit} \times 4.4 \text{ Gbps} \times \frac{1}{8} = 3379 \text{ GB/s} \approx 3.35 \text{ TB/s}
$$

HBM achieves massive bandwidth through width (6144-bit vs 384-bit), not just speed.

## Comprehensive Comparison Tables

### Data Center GPU Evolution

| GPU | Arch | Year | SMs | CUDA Cores | Tensor Cores | FP32 (TFLOPS) | FP16 Tensor (TFLOPS) | Memory | Bandwidth | TDP | Process |
|-----|------|------|-----|------------|--------------|---------------|----------------------|--------|-----------|-----|---------|
| **P100** | Pascal | 2016 | 56 | 3,584 | - | 10.6 | 21.2* | 16 GB HBM2 | 720 GB/s | 300W | 16nm |
| **V100** | Volta | 2017 | 80 | 5,120 | 640 (1st) | 15.7 | 125 | 32 GB HBM2 | 900 GB/s | 300W | 12nm |
| **A100** | Ampere | 2020 | 108 | 6,912 | 432 (3rd) | 19.5 | 312/624** | 80 GB HBM2e | 2.0 TB/s | 400W | 7nm |
| **H100** | Hopper | 2022 | 132 | 16,896 | 528 (4th) | 67 | 1,513/3,026** | 80 GB HBM3 | 3.35 TB/s | 700W | 4nm |
| **H200** | Hopper | 2023 | 132 | 16,896 | 528 (4th) | 67 | 1,513/3,026** | 141 GB HBM3e | 4.8 TB/s | 700W | 4nm |
| **B200** | Blackwell | 2024 | TBD | TBD | TBD (5th) | 45 | 9,000/18,000** | 192 GB HBM3e | 8 TB/s | 1000W | 4nm |

*P100 FP16 is packed 2:1 throughput  
**Second number includes 2:4 sparsity

### Consumer GPU Flagship Evolution

| GPU | Arch | Year | CUDA Cores | RT Cores | Tensor Cores | FP32 (TFLOPS) | Memory | Bandwidth | TDP | MSRP |
|-----|------|------|------------|----------|--------------|---------------|--------|-----------|-----|------|
| **GTX 1080 Ti** | Pascal | 2017 | 3,584 | - | - | 11.3 | 11 GB GDDR5X | 484 GB/s | 250W | $699 |
| **RTX 2080 Ti** | Turing | 2018 | 4,352 | 68 (1st) | 544 (2nd) | 13.4 | 11 GB GDDR6 | 616 GB/s | 260W | $999 |
| **RTX 3090 Ti** | Ampere | 2022 | 10,752 | 84 (2nd) | 336 (3rd) | 40.0 | 24 GB GDDR6X | 1,008 GB/s | 450W | $1,999 |
| **RTX 4090** | Ada | 2022 | 16,384 | 128 (3rd) | 512 (4th) | 82.6 | 24 GB GDDR6X | 1,008 GB/s | 450W | $1,599 |
| **RTX 5090** | Blackwell | 2025 | ~21,760 | 170 (4th) | 680 (5th) | ~120 | 32 GB GDDR7 | ~1,536 GB/s | 600W | TBD |

### NVLink Evolution

| Generation | Architecture | Bandwidth (per link) | Links (typical) | Total Bandwidth | Year |
|------------|--------------|----------------------|-----------------|-----------------|------|
| **NVLink 1.0** | Pascal | 20 GB/s | 4 | 160 GB/s | 2016 |
| **NVLink 2.0** | Volta | 25 GB/s | 6 | 300 GB/s | 2017 |
| **NVLink 3.0** | Ampere | 25 GB/s | 12 | 600 GB/s | 2020 |
| **NVLink 4.0** | Hopper | 25 GB/s | 18 | 900 GB/s | 2022 |
| **NVLink 5.0** | Blackwell | 50 GB/s | 18 | 1,800 GB/s | 2024 |

NVLink bandwidth has increased 11× from Pascal to Blackwell, crucial for multi-GPU scaling.

## Performance Scaling Analysis

Let's quantify the generational improvements:

### Compute Performance (FP32 TFLOPS)

<div class="diagram">
<div class="diagram-title">Data Center FP32 Performance Evolution</div>
<div class="flow">
  <div class="flow-node accent wide">P100 (2016): 10.6 TFLOPS — Baseline</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">V100 (2017): 15.7 TFLOPS — 1.5× in 1 year</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">A100 (2020): 19.5 TFLOPS — 1.2× in 3 years</div>
  <div class="flow-arrow blue"></div>
  <div class="flow-node purple wide">H100 (2022): 67 TFLOPS — 3.4× in 2 years</div>
  <div class="flow-arrow purple"></div>
  <div class="flow-node orange wide">B200 (2024): 45 TFLOPS — 0.67× (specialized for AI)</div>
</div>
</div>

Wait, B200 has *lower* FP32? Yes! Because modern AI workloads don't use FP32. They use FP8/TF32/FP16.

### AI Performance (Tensor Core TFLOPS, FP16/FP8)

| GPU | FP16 Tensor (TFLOPS) | FP8 Tensor (TFLOPS) | Speedup vs P100 |
|-----|----------------------|---------------------|-----------------|
| P100 (2016) | 21.2 (packed FP16) | - | 1× baseline |
| V100 (2017) | 125 | - | **5.9×** |
| A100 (2020) | 312 | - | **14.7×** |
| H100 (2022) | 1,513 | 1,513 | **71.4× (FP16), 71.4× (FP8)** |
| B200 (2024) | 9,000 | 18,000 | **424× (FP16), 849× (FP8)** |

**849× improvement in 8 years.** This is not Moore's Law. This is specialization.

### Memory Bandwidth Scaling

| GPU | Bandwidth | vs P100 | Annual Growth |
|-----|-----------|---------|---------------|
| P100 (2016) | 720 GB/s | 1.0× | - |
| V100 (2017) | 900 GB/s | 1.25× | 25% YoY |
| A100 (2020) | 2,000 GB/s | 2.78× | 30% YoY |
| H100 (2022) | 3,350 GB/s | 4.65× | 29% YoY |
| B200 (2024) | 8,000 GB/s | 11.1× | 55% YoY |

Bandwidth has grown 11× in 8 years (37% CAGR), but AI compute has grown 849× (118% CAGR). This creates a memory bandwidth bottleneck.

### The Memory Wall

The gap between compute and bandwidth is widening:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

For a model to be compute-bound (good GPU utilization):

$$
\text{AI} > \frac{\text{Peak FLOPs}}{\text{Peak Bandwidth}}
$$

**Examples:**

**H100:**
$$
\text{AI threshold} = \frac{1513 \text{ TFLOPS}}{3.35 \text{ TB/s}} = 451.6 \text{ FLOPs/Byte}
$$

**B200:**
$$
\text{AI threshold} = \frac{18000 \text{ TFLOPS}}{8 \text{ TB/s}} = 2250 \text{ FLOPs/Byte}
$$

B200 requires **5× higher arithmetic intensity** to be compute-bound! This is why:
1. Flash Attention and other algorithms that reduce memory traffic are critical
2. Model architectures are evolving to increase compute intensity (e.g., MoE layers)
3. FP8 quantization helps (smaller transfers)

### Power Efficiency Evolution

| GPU | AI Perf (TFLOPS FP16) | TDP | Efficiency (TFLOPS/W) | vs P100 |
|-----|----------------------|-----|----------------------|---------|
| P100 | 21.2 | 300W | 0.071 | 1.0× |
| V100 | 125 | 300W | 0.417 | 5.9× |
| A100 | 312 | 400W | 0.780 | 11.0× |
| H100 | 1,513 | 700W | 2.161 | 30.5× |
| B200 | 9,000 | 1000W | 9.000 | 127× |

**127× better TFLOPS per watt** in 8 years. Energy efficiency has improved faster than raw performance.

## Architectural Innovations Summary

Let's summarize the *key innovation* each generation brought:

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔧</div>
    <div class="card-title">Tesla (2006)</div>
    <div class="card-desc">
      <strong>Unified shaders</strong><br>
      Birth of GPGPU. CUDA programming model. Changed computing forever.
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🎓</div>
    <div class="card-title">Fermi (2010)</div>
    <div class="card-desc">
      <strong>HPC credibility</strong><br>
      IEEE 754, ECC, cache hierarchy. GPUs get serious about science.
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">⚡</div>
    <div class="card-title">Kepler (2012)</div>
    <div class="card-desc">
      <strong>Efficiency & Scale</strong><br>
      Dynamic parallelism, Hyper-Q, 3× perf/watt. Massive core counts.
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔋</div>
    <div class="card-title">Maxwell (2014)</div>
    <div class="card-desc">
      <strong>Power efficiency</strong><br>
      Partitioned SM, 2× perf/watt. Gaming focused, architectural refinement.
    </div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">🚀</div>
    <div class="card-title">Pascal (2016)</div>
    <div class="card-desc">
      <strong>HBM2 & NVLink</strong><br>
      16nm FinFET, FP16×2, unified memory. Deep learning breakthrough.
    </div>
  </div>
  <div class="diagram-card yellow">
    <div class="card-icon">🎯</div>
    <div class="card-title">Volta (2017)</div>
    <div class="card-desc">
      <strong>TENSOR CORES</strong><br>
      Matrix multiply acceleration. 8× AI speedup. The game changer.
    </div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">🎨</div>
    <div class="card-title">Turing (2018)</div>
    <div class="card-desc">
      <strong>RT CORES</strong><br>
      Real-time ray tracing. DLSS. Gaming + AI convergence.
    </div>
  </div>
  <div class="diagram-card red">
    <div class="card-icon">📊</div>
    <div class="card-title">Ampere (2020)</div>
    <div class="card-desc">
      <strong>Sparsity & TF32</strong><br>
      Sparse tensors, TF32 precision. A100 dominates AI training.
    </div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">✨</div>
    <div class="card-title">Ada Lovelace (2022)</div>
    <div class="card-desc">
      <strong>Advanced RT</strong><br>
      OMM, DMM, SER. DLSS 3 frame generation. RT maturity.
    </div>
  </div>
  <div class="diagram-card teal">
    <div class="card-icon">🧠</div>
    <div class="card-title">Hopper (2022)</div>
    <div class="card-desc">
      <strong>Transformer Engine</strong><br>
      FP8, auto-precision switching. Built for LLMs. H100 era begins.
    </div>
  </div>
  <div class="diagram-card accent">
    <div class="card-icon">🌌</div>
    <div class="card-title">Blackwell (2024)</div>
    <div class="card-desc">
      <strong>Extreme Scale</strong><br>
      FP4, 2nd-gen Transformer Engine, 208B transistors. GB200 superchip.
    </div>
  </div>
</div>

## Where Are We Headed?

Looking at the trends:

1. **Precision diversification:** We've gone from FP32/FP64 to supporting FP64, TF32, BF16, FP16, FP8, FP6, FP4, INT8, INT4. Future: FP2? Binary networks?

2. **Memory becomes the bottleneck:** Compute grows faster than bandwidth. Solutions: HBM3e/HBM4, on-chip SRAM, algorithmic improvements.

3. **Specialization accelerates:** General-purpose CUDA cores matter less. Specialized engines (Tensor Cores, RT Cores, Transformer Engines) dominate.

4. **Scale up and out:** Single-GPU → multi-GPU → clusters → data center-scale fabrics. NVSwitch, NVLink 5.0 enable 10,000+ GPU supercomputers.

5. **Power wall approaching:** B200 at 1000W, GB200 systems at 120kW/rack. Physics limits loom. Efficiency becomes critical.

6. **Software co-design:** Hardware alone isn't enough. Transformer Engine, TensorRT-LLM, Flash Attention, etc. Software must evolve with hardware.

**Huang's Law:** Jensen Huang's observation that GPU performance doubles every year (vs Moore's Law's 18-24 months). The data supports this for AI workloads:

| Period | Metric | Annual Growth |
|--------|--------|---------------|
| 2016-2024 | FP16 Tensor Performance | **61% CAGR** (doubles every ~1.2 years) |
| 2016-2024 | Memory Bandwidth | 37% CAGR (doubles every ~2.2 years) |
| 2016-2024 | Transistor Count | 35% CAGR (doubles every ~2.3 years) |

So yes, Huang's Law holds — for specialized AI operations. General-purpose compute (FP32) has grown much slower (~25% CAGR).

## Code Example: Generation Detection

Let's write a CUDA program that detects which GPU generation you're running:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

const char* getArchitectureName(int major, int minor) {
    if (major == 3) return "Kepler";
    if (major == 5) return "Maxwell";
    if (major == 6) {
        if (minor == 0) return "Pascal (P100)";
        if (minor == 1) return "Pascal (Consumer)";
    }
    if (major == 7) {
        if (minor == 0) return "Volta";
        if (minor == 5) return "Turing";
    }
    if (major == 8) {
        if (minor == 0) return "Ampere (A100)";
        if (minor == 6) return "Ampere (RTX 30)";
        if (minor == 9) return "Ada Lovelace (RTX 40)";
    }
    if (major == 9) {
        if (minor == 0) return "Hopper";
    }
    if (major == 10) {
        if (minor == 0) return "Blackwell";
    }
    return "Unknown";
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("NVIDIA GPU Architecture Detection\n");
    printf("==================================\n\n");
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("GPU %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Architecture: %s\n", getArchitectureName(prop.major, prop.minor));
        printf("  CUDA Cores: %d SMs × %d cores/SM = ~%d cores\n",
               prop.multiProcessorCount,
               (prop.major >= 8) ? 128 : 64,  // Ampere+ has 128/SM
               prop.multiProcessorCount * ((prop.major >= 8) ? 128 : 64));
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Memory Bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
        printf("  L2 Cache: %.2f MB\n", prop.l2CacheSize / 1e6);
        
        // Check for special features
        if (prop.major >= 7) {
            printf("  ✓ Tensor Cores available\n");
        }
        if (prop.major >= 8) {
            printf("  ✓ Sparsity support (2:4)\n");
        }
        if (prop.major >= 9) {
            printf("  ✓ FP8 Tensor Cores (Hopper+)\n");
            printf("  ✓ Transformer Engine\n");
        }
        
        printf("\n");
    }
    
    return 0;
}
```

Compile and run:
```bash
nvcc -o gpu_detect gpu_detect.cu
./gpu_detect

# Example output on H100:
# GPU 0: NVIDIA H100 PCIe
#   Compute Capability: 9.0
#   Architecture: Hopper
#   CUDA Cores: 132 SMs × 128 cores/SM = ~16896 cores
#   Global Memory: 80.00 GB
#   Memory Bandwidth: 2039.1 GB/s
#   L2 Cache: 50.00 MB
#   ✓ Tensor Cores available
#   ✓ Sparsity support (2:4)
#   ✓ FP8 Tensor Cores (Hopper+)
#   ✓ Transformer Engine
```

## Final Thoughts: The Exponential Age

We've traced NVIDIA's GPU evolution from the 128-core G80 (2006) to the 208-billion-transistor Blackwell (2024). In that time:

- **CUDA cores:** 128 → 16,896 (132× increase)
- **Transistors:** 681 million → 208 billion (305× increase)
- **AI performance:** ~0.5 TFLOPS → 18,000 TFLOPS (36,000× increase)
- **Memory bandwidth:** ~86 GB/s → 8,000 GB/s (93× increase)

But the most important evolution isn't in the numbers — it's in the **architectural philosophy**:

<div class="flow">
  <div class="flow-node accent wide">2006-2012: General-purpose parallelism</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">2012-2017: Efficiency optimization</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">2017-2020: AI specialization (Tensor Cores)</div>
  <div class="flow-arrow blue"></div>
  <div class="flow-node purple wide">2020-2024: Extreme specialization (Transformer Engine, FP8, FP4)</div>
  <div class="flow-arrow purple"></div>
  <div class="flow-node orange wide">2024+: Full-stack co-design (hardware + software + algorithms)</div>
</div>

GPUs have evolved from "graphics cards that can also do math" to "AI supercomputers that can also render graphics." The tail is wagging the dog now.

Each generation doesn't just add more cores or memory — it fundamentally rethinks what computation should look like for the workloads that matter. Volta introduced matrix ops. Hopper added dynamic precision. Blackwell optimizes for trillion-parameter models.

And we're not done. The next decade will bring challenges that make today's 208-billion-transistor chips look quaint:
- 10-trillion parameter models
- Real-time photorealistic rendering
- Embodied AI and robotics
- Quantum-classical hybrid computing
- Brain-computer interfaces

NVIDIA's GPU generations have consistently delivered the impossible. The question isn't whether the next generation will be revolutionary — it's which impossibility they'll make routine.

**Next: [Chapter 9 — NVIDIA CPUs →](./09_nvidia_cpus.md)**

---

*Last updated: April 2026*

