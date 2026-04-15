---
title: "Chapter 3 — CUDA Programming"
---

[← Back to Table of Contents](./README.md)

# Chapter 3: CUDA Programming

## The Language of GPU Computation

CUDA (Compute Unified Device Architecture) represents one of the most significant paradigm shifts in parallel computing. Released by NVIDIA in 2006, CUDA transformed GPUs from fixed-function graphics processors into general-purpose parallel computing engines. This chapter provides a comprehensive exploration of CUDA programming—from fundamental concepts to advanced optimization techniques—equipping you with the knowledge to harness the full power of NVIDIA GPUs.

---

## 1. What is CUDA?

### The Revolutionary Programming Model

**CUDA (Compute Unified Device Architecture)** is NVIDIA's parallel computing platform and programming model that enables developers to use C, C++, and Fortran to write code that executes on NVIDIA GPUs. Before CUDA, GPU programming required expertise in graphics APIs like OpenGL or DirectX, forcing developers to express computations as graphics operations—a cumbersome and unintuitive process.

<div class="diagram">
<div class="diagram-title">CUDA's Historical Impact</div>
<div class="timeline">
<div class="timeline-item">
<div class="timeline-marker accent"></div>
<div class="timeline-content">
<h4>Pre-2006: Graphics-Only Era</h4>
<p>GPUs limited to graphics rendering. GPGPU (General-Purpose GPU) programming required expressing computations as texture operations—highly restrictive and complex.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker green"></div>
<div class="timeline-content">
<h4>2006: CUDA 1.0 Launch</h4>
<p>First accessible GPU programming model. Introduced with GeForce 8800 GTX (Tesla architecture). Enabled C/C++ developers to write GPU code without graphics knowledge.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker purple"></div>
<div class="timeline-content">
<h4>2008-2012: Ecosystem Growth</h4>
<p>CUDA 2.0 added shared memory, CUDA 3.0 introduced Fermi architecture support, CUDA 4.0 brought unified virtual addressing. Libraries like cuBLAS, cuFFT, cuDNN emerged.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker cyan"></div>
<div class="timeline-content">
<h4>2014-2018: Deep Learning Boom</h4>
<p>CUDA became the de facto standard for AI/ML. TensorFlow, PyTorch built on CUDA. Tensor Cores introduced in Volta (2017).</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker orange"></div>
<div class="timeline-content">
<h4>2020-Present: Modern CUDA</h4>
<p>CUDA 11+ brings enhanced cooperative groups, CUDA Graphs, multi-instance GPU (MIG), Hopper's Tensor Memory Accelerator. CUDA 12 introduces dynamic kernel loading.</p>
</div>
</div>
</div>
</div>

### Why CUDA Was Revolutionary

1. **Familiar Programming Model**: C/C++ extensions rather than graphics shaders
2. **Explicit Memory Control**: Direct management of GPU memory hierarchies
3. **Scalable Parallelism**: Code scales automatically across different GPU sizes
4. **Rich Ecosystem**: Extensive libraries, debugging tools (CUDA-GDB, cuda-memcheck), profilers (Nsight)
5. **Industry Adoption**: Became the standard for scientific computing, deep learning, and HPC

### CUDA vs. Alternative GPU Programming Models

<div class="diagram-grid cols-2">
<div class="diagram-card accent">
<h3>🚀 CUDA</h3>
<p><strong>Vendor:</strong> NVIDIA-exclusive</p>
<p><strong>Languages:</strong> C/C++, Python (PyCUDA, Numba), Fortran</p>
<p><strong>Pros:</strong> Mature ecosystem, excellent tooling, massive library support (cuDNN, cuBLAS, TensorRT), best performance on NVIDIA hardware</p>
<p><strong>Cons:</strong> Vendor lock-in, not portable to AMD/Intel GPUs</p>
<p><strong>Best For:</strong> Maximum performance on NVIDIA GPUs, production AI/ML workloads</p>
</div>

<div class="diagram-card green">
<h3>🌐 OpenCL</h3>
<p><strong>Vendor:</strong> Cross-platform (NVIDIA, AMD, Intel, ARM, FPGAs)</p>
<p><strong>Languages:</strong> C/C++</p>
<p><strong>Pros:</strong> True portability, runs on heterogeneous devices, open standard</p>
<p><strong>Cons:</strong> More verbose, less tooling, often slower than native implementations, fragmented support</p>
<p><strong>Best For:</strong> Cross-vendor portability requirements, embedded systems</p>
</div>

<div class="diagram-card purple">
<h3>⚡ ROCm/HIP</h3>
<p><strong>Vendor:</strong> AMD (with HIP providing CUDA compatibility layer)</p>
<p><strong>Languages:</strong> C/C++, HIP (CUDA-like syntax)</p>
<p><strong>Pros:</strong> Can convert CUDA code via hipify tools, good performance on AMD GPUs</p>
<p><strong>Cons:</strong> Limited to AMD hardware, smaller ecosystem than CUDA</p>
<p><strong>Best For:</strong> AMD GPU deployments, porting CUDA applications</p>
</div>

<div class="diagram-card cyan">
<h3>🔧 SYCL</h3>
<p><strong>Vendor:</strong> Cross-platform (Intel oneAPI, ComputeCpp)</p>
<p><strong>Languages:</strong> Modern C++ (C++17/20)</p>
<p><strong>Pros:</strong> Single-source C++, clean abstractions, growing adoption</p>
<p><strong>Cons:</strong> Younger ecosystem, less mature tooling</p>
<p><strong>Best For:</strong> Modern C++ codebases, Intel hardware</p>
</div>

<div class="diagram-card orange">
<h3>🎮 Vulkan Compute</h3>
<p><strong>Vendor:</strong> Cross-platform (graphics-focused)</p>
<p><strong>Languages:</strong> SPIR-V shaders</p>
<p><strong>Pros:</strong> Unified graphics + compute, mobile support, explicit control</p>
<p><strong>Cons:</strong> Verbose, complex, primarily for graphics pipelines</p>
<p><strong>Best For:</strong> Games/graphics with compute, mobile GPUs</p>
</div>

<div class="diagram-card pink">
<h3>🐍 High-Level Frameworks</h3>
<p><strong>Examples:</strong> Numba, JAX, Triton, Taichi</p>
<p><strong>Languages:</strong> Python-centric</p>
<p><strong>Pros:</strong> Rapid development, automatic optimization, Python integration</p>
<p><strong>Cons:</strong> Less control, may not achieve hand-tuned CUDA performance</p>
<p><strong>Best For:</strong> Prototyping, research, when developer time > compute time</p>
</div>
</div>

### CUDA Architecture Compatibility

| **CUDA Compute Capability** | **Architecture** | **Key Features** | **Example GPUs** |
|------------------------------|------------------|------------------|------------------|
| 3.5 | Kepler | Dynamic Parallelism, Hyper-Q | Tesla K40, GTX Titan |
| 5.0 - 5.3 | Maxwell | Improved power efficiency | GTX 980, Titan X |
| 6.0 - 6.2 | Pascal | NVLink, Unified Memory improvements | Tesla P100, GTX 1080 Ti |
| 7.0 - 7.5 | Volta/Turing | Tensor Cores, Independent Thread Scheduling | Tesla V100, RTX 2080 |
| 8.0 - 8.9 | Ampere | 3rd-gen Tensor Cores, Multi-Instance GPU | A100, RTX 3090 |
| 9.0 | Hopper | Transformer Engine, TMA, FP8 | H100, H200 |

---

## 2. The CUDA Programming Model

### Host and Device: A Heterogeneous Computing Partnership

CUDA programs execute on a **heterogeneous system**: a **host** (CPU with system memory) and one or more **devices** (GPUs with dedicated memory). The CPU orchestrates computation, while the GPU executes massively parallel workloads.

<div class="diagram">
<div class="diagram-title">Host-Device Execution Model</div>
<div class="flow">
<div class="flow-node accent wide">CPU (Host) - Serial/Complex Logic</div>
<div class="flow-arrow accent">⬇ Launch Kernel</div>
<div class="flow-node green wide">GPU (Device) - Massively Parallel Execution</div>
<div class="flow-arrow green">⬇ Return Results</div>
<div class="flow-node purple wide">CPU Processes Results</div>
</div>
</div>

### Kernel Functions: The Heart of CUDA

A **kernel** is a function that executes on the GPU. Unlike regular functions, a kernel runs on **thousands of threads in parallel**.

**Function Type Qualifiers:**

```cpp
// Executes on device (GPU), callable only from device
__global__ void myKernel(float* data) {
    // Kernel code - runs on GPU
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2.0f;
}

// Executes on device, callable only from device
__device__ float deviceFunction(float x) {
    return x * x;
}

// Executes on host (CPU), callable from host
__host__ float hostFunction(float x) {
    return x * x;
}

// Can be compiled for both host and device
__host__ __device__ float hybridFunction(float x) {
    return x * x + 1.0f;
}
```

**Key Distinctions:**
- `__global__`: Entry point for GPU execution (kernel), called from host
- `__device__`: Helper function on GPU, called from other device functions
- `__host__`: Explicit CPU function (default if no qualifier)
- Combined `__host__ __device__`: Compiled for both architectures

### Thread Hierarchy: Grid → Block → Thread

CUDA organizes threads in a three-level hierarchy, providing flexibility for mapping algorithms to hardware:

<div class="diagram">
<div class="diagram-title">CUDA Thread Hierarchy</div>
<div class="flow">
<div class="flow-node accent wide">Grid (1D, 2D, or 3D)</div>
<div class="flow-arrow accent">⬇ Contains</div>
<div class="flow-node green wide">Blocks (1D, 2D, or 3D)</div>
<div class="flow-arrow green">⬇ Contains</div>
<div class="flow-node purple wide">Threads (1D, 2D, or 3D)</div>
<div class="flow-arrow purple">⬇ Grouped into</div>
<div class="flow-node cyan wide">Warps (32 threads, execution unit)</div>
</div>
</div>

**Built-in Variables:**

```cpp
// Thread indices within a block
threadIdx.x, threadIdx.y, threadIdx.z  // 0 to blockDim-1

// Block indices within a grid
blockIdx.x, blockIdx.y, blockIdx.z     // 0 to gridDim-1

// Block dimensions (threads per block)
blockDim.x, blockDim.y, blockDim.z

// Grid dimensions (blocks per grid)
gridDim.x, gridDim.y, gridDim.z
```

### Computing Global Thread Index

For **1D layout** (most common):
```cpp
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

For **2D layout** (image processing):
```cpp
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int tid = y * width + x;
```

For **3D layout** (volumetric data):
```cpp
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int z = threadIdx.z + blockIdx.z * blockDim.z;
int tid = z * width * height + y * width + x;
```

### Kernel Launch Configuration

Kernels are launched using the triple-chevron syntax: `kernel<<<grid, block, sharedMem, stream>>>(args)`

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<h3>1D Configuration</h3>
<pre><code>int N = 1000000;
int threadsPerBlock = 256;
int blocksPerGrid = 
    (N + threadsPerBlock - 1) / threadsPerBlock;

kernel<<<blocksPerGrid, threadsPerBlock>>>(data, N);</code></pre>
<p><strong>Use Case:</strong> Vectors, 1D arrays, simple parallel loops</p>
</div>

<div class="diagram-card green">
<h3>2D Configuration</h3>
<pre><code>dim3 threadsPerBlock(16, 16);  // 256 threads
dim3 blocksPerGrid(
    (width + 15) / 16,
    (height + 15) / 16
);

kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);</code></pre>
<p><strong>Use Case:</strong> Images, matrices, 2D grids</p>
</div>

<div class="diagram-card purple">
<h3>3D Configuration</h3>
<pre><code>dim3 threadsPerBlock(8, 8, 8);  // 512 threads
dim3 blocksPerGrid(
    (width + 7) / 8,
    (height + 7) / 8,
    (depth + 7) / 8
);

kernel<<<blocksPerGrid, threadsPerBlock>>>(volume);</code></pre>
<p><strong>Use Case:</strong> 3D volumes, simulations, CFD</p>
</div>
</div>

**Optimal Block Sizes:**
- **64-256 threads per block** is typical (256 is a safe default)
- Must be a **multiple of warp size (32)** for efficiency
- Hardware limits: **max 1024 threads per block** on modern GPUs
- Consider occupancy: more blocks allow better hiding of memory latency

---

## 3. Memory Model

CUDA exposes a sophisticated memory hierarchy, each with different characteristics:

<div class="diagram">
<div class="diagram-title">CUDA Memory Hierarchy</div>
<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<h3>🚀 Registers</h3>
<p><strong>Scope:</strong> Per-thread</p>
<p><strong>Speed:</strong> Fastest (~1 cycle)</p>
<p><strong>Size:</strong> ~64 KB per SM</p>
<p><strong>Usage:</strong> Automatic local variables</p>
</div>

<div class="diagram-card green">
<h3>⚡ Shared Memory</h3>
<p><strong>Scope:</strong> Per-block</p>
<p><strong>Speed:</strong> Very fast (~4-8 cycles)</p>
<p><strong>Size:</strong> 48-164 KB per SM</p>
<p><strong>Usage:</strong> Explicitly managed (__shared__)</p>
</div>

<div class="diagram-card purple">
<h3>🌐 Global Memory</h3>
<p><strong>Scope:</strong> All threads</p>
<p><strong>Speed:</strong> Slow (200-400 cycles)</p>
<p><strong>Size:</strong> GB-scale (device DRAM)</p>
<p><strong>Usage:</strong> Main data storage</p>
</div>

<div class="diagram-card cyan">
<h3>📦 Local Memory</h3>
<p><strong>Scope:</strong> Per-thread</p>
<p><strong>Speed:</strong> Slow (global mem speed)</p>
<p><strong>Size:</strong> Spilled registers/arrays</p>
<p><strong>Usage:</strong> Automatic (register spillage)</p>
</div>

<div class="diagram-card orange">
<h3>📌 Constant Memory</h3>
<p><strong>Scope:</strong> Read-only, all threads</p>
<p><strong>Speed:</strong> Fast (cached)</p>
<p><strong>Size:</strong> 64 KB</p>
<p><strong>Usage:</strong> __constant__ declaration</p>
</div>

<div class="diagram-card pink">
<h3>🖼️ Texture Memory</h3>
<p><strong>Scope:</strong> Read-only, cached</p>
<p><strong>Speed:</strong> Fast (with spatial locality)</p>
<p><strong>Size:</strong> Global memory size</p>
<p><strong>Usage:</strong> 2D/3D data with filtering</p>
</div>
</div>
</div>

### Global Memory: The Primary Workspace

**Allocation and Transfer:**

```cpp
// Host (CPU) code
float *h_data;      // Host pointer
float *d_data;      // Device pointer
size_t size = N * sizeof(float);

// Allocate host memory
h_data = (float*)malloc(size);

// Allocate device memory
cudaMalloc((void**)&d_data, size);

// Copy host → device
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Launch kernel
kernel<<<grid, block>>>(d_data, N);

// Copy device → host
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Free memory
cudaFree(d_data);
free(h_data);
```

**Memory Transfer Directions:**
- `cudaMemcpyHostToDevice`: CPU → GPU
- `cudaMemcpyDeviceToHost`: GPU → CPU
- `cudaMemcpyDeviceToDevice`: GPU → GPU (within device)
- `cudaMemcpyHostToHost`: CPU → CPU (rarely used)

### Shared Memory: Fast On-Chip Collaboration

Shared memory enables threads within a block to cooperate efficiently:

```cpp
__global__ void sharedMemoryExample(float *input, float *output, int N) {
    // Allocate shared memory (visible to all threads in block)
    __shared__ float sharedData[256];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;
    
    // Load from global memory into shared memory
    if (tid < N) {
        sharedData[localIdx] = input[tid];
    }
    
    // Synchronize to ensure all threads have loaded data
    __syncthreads();
    
    // Now all threads can access sharedData quickly
    if (tid < N) {
        float sum = 0.0f;
        // Example: compute moving average using shared memory
        for (int i = -1; i <= 1; i++) {
            int idx = localIdx + i;
            if (idx >= 0 && idx < blockDim.x) {
                sum += sharedData[idx];
            }
        }
        output[tid] = sum / 3.0f;
    }
}
```

**Dynamic Shared Memory** (size determined at runtime):

```cpp
__global__ void dynamicShared(float *data, int N) {
    extern __shared__ float sharedData[];  // Size specified at launch
    // ... use sharedData ...
}

// Launch with dynamic shared memory
int sharedMemSize = 512 * sizeof(float);
kernel<<<grid, block, sharedMemSize>>>(data, N);
```

### Constant Memory: Read-Only Broadcast

Ideal for parameters accessed uniformly by all threads:

```cpp
// Declare in global scope
__constant__ float constParams[256];

// Host code: copy data to constant memory
float h_params[256];
// ... initialize h_params ...
cudaMemcpyToSymbol(constParams, h_params, 256 * sizeof(float));

// Device code: read constant memory
__global__ void useConstants(float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] *= constParams[0];  // Fast broadcast read
    }
}
```

### Unified Memory: Simplified Programming

Unified Memory creates a single memory space accessible from both CPU and GPU:

```cpp
float *data;
// Allocate unified memory (accessible from host and device)
cudaMallocManaged(&data, N * sizeof(float));

// Initialize on CPU
for (int i = 0; i < N; i++) {
    data[i] = i;
}

// Use on GPU (automatic migration)
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();  // Wait for kernel completion

// Access on CPU (automatic migration back)
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += data[i];
}

cudaFree(data);  // Works for both regular and managed memory
```

**Unified Memory Benefits:**
- Simplified programming (no explicit cudaMemcpy)
- Automatic page migration
- Oversubscription (use more memory than GPU DRAM)

**Limitations:**
- May be slower than explicit transfers (migration overhead)
- Best with demand paging (Pascal+) and access counters (Volta+)

### Memory Allocation Pattern Comparison

| **Pattern** | **Allocation** | **Transfer** | **Performance** | **Use Case** |
|-------------|----------------|--------------|-----------------|--------------|
| **Explicit** | `cudaMalloc` | Manual `cudaMemcpy` | Fastest (full control) | Production, performance-critical |
| **Unified Memory** | `cudaMallocManaged` | Automatic | Good (Pascal+) | Prototyping, complex data structures |
| **Zero-Copy** | `cudaHostAlloc` | Pinned host memory | Slow (PCIe bandwidth) | Small data, frequent access |
| **Mapped Memory** | `cudaHostAlloc` + `cudaHostGetDevicePointer` | Direct access | Slow | Streaming, overlapped compute |

---

## 4. Writing Your First Kernel

### Example 1: Vector Addition (The "Hello World" of CUDA)

Let's implement a complete CUDA program that adds two vectors:

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel: Add two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    // Vector size
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    // 1. Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }
    
    // 2. Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));
    
    // 3. Copy host → device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // 4. Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());  // Check kernel launch errors
    
    // 5. Copy device → host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // 6. Verify results
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f (expected %f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }
    
    // 7. Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Vector addition completed successfully!\n");
    return 0;
}
```

**Compilation:**
```bash
nvcc -o vector_add vector_add.cu
./vector_add
```

### Example 2: Matrix Multiplication (Naive Version)

Computing C = A × B where A is M×K, B is K×N, C is M×N:

```cpp
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    // Compute row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 1024, K = 1024, N = 1024;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate and initialize host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    
    // ... initialize h_A and h_B ...
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 threadsPerBlock(16, 16);  // 256 threads
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
```

**Problem with Naive Version:**
- Each thread reads from global memory K times
- A[row,:] is reused N times (once per column)
- B[:,col] is reused M times (once per row)
- Total global memory reads: **O(M × N × K) = 1 billion reads for 1024×1024!**

### Example 3: Tiled Matrix Multiplication (Optimized with Shared Memory)

By loading tiles into shared memory, we dramatically reduce global memory accesses:

```cpp
#define TILE_SIZE 16

__global__ void matrixMulTiled(const float *A, const float *B, float *C,
                                int M, int K, int N) {
    // Allocate shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Performance Improvement:**
- **Naive version**: ~50 GFLOPS on RTX 3090
- **Tiled version**: ~800 GFLOPS on RTX 3090
- **cuBLAS (vendor-optimized)**: ~10,000+ GFLOPS on RTX 3090

<div class="diagram">
<div class="diagram-title">Tiled Matrix Multiplication Strategy</div>
<div class="diagram-grid cols-2">
<div class="diagram-card accent">
<h3>Memory Access Pattern</h3>
<ul>
<li><strong>Global reads per element:</strong> Reduced from K to K/TILE_SIZE</li>
<li><strong>Shared memory reuse:</strong> Each tile element used TILE_SIZE times</li>
<li><strong>Bandwidth reduction:</strong> ~16× fewer global memory transactions</li>
</ul>
</div>
<div class="diagram-card green">
<h3>Synchronization Strategy</h3>
<ul>
<li><strong>Load phase:</strong> All threads load tile from global to shared</li>
<li><strong>__syncthreads():</strong> Ensure load completion</li>
<li><strong>Compute phase:</strong> All threads compute using shared data</li>
<li><strong>__syncthreads():</strong> Ensure compute before next tile</li>
</ul>
</div>
</div>
</div>

---

## 5. Synchronization Primitives

CUDA provides multiple synchronization mechanisms for different scopes:

### Block-Level Synchronization: __syncthreads()

Ensures all threads in a block reach the same point before proceeding:

```cpp
__global__ void reductionWithSync(float *input, float *output, int N) {
    __shared__ float sharedData[256];
    
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedData[tid] = (globalIdx < N) ? input[globalIdx] : 0.0f;
    __syncthreads();  // CRITICAL: Wait for all loads to complete
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();  // CRITICAL: Wait for all additions before next iteration
    }
    
    // First thread writes block result
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
```

**__syncthreads() Rules:**
- ✅ **Must be reached by all threads in the block** (no conditional execution)
- ✅ Synchronizes threads within a single block only
- ❌ Does NOT synchronize across different blocks
- ❌ Cannot be used in divergent code paths

**Bad Example (undefined behavior):**
```cpp
if (threadIdx.x < 100) {
    __syncthreads();  // ERROR: Only some threads reach this
}
```

### Warp-Level Synchronization: __syncwarp()

Introduced in Volta with independent thread scheduling:

```cpp
__global__ void warpSyncExample(int *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < N) {
        // Warp-level operation
        int value = data[tid];
        __syncwarp();  // Synchronize threads in the same warp
        
        // Shuffle: exchange data between threads in warp
        int neighbor = __shfl_down_sync(0xffffffff, value, 1);
    }
}
```

### Atomic Operations: Thread-Safe Updates

Atomics ensure thread-safe read-modify-write operations:

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<h3>atomicAdd</h3>
<pre><code>__global__ void atomicSum(float *result, float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        atomicAdd(result, data[tid]);
    }
}</code></pre>
<p>Thread-safe addition. Supported for int, float, double (compute 6.0+).</p>
</div>

<div class="diagram-card green">
<h3>atomicCAS</h3>
<pre><code>// Compare-And-Swap: atomic if-then-set
__device__ void atomicMax(float *addr, float value) {
    unsigned int *addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_uint, assumed,
            __float_as_uint(fmaxf(value, __uint_as_float(assumed))));
    } while (assumed != old);
}</code></pre>
<p>Building block for custom atomics.</p>
</div>

<div class="diagram-card purple">
<h3>atomicExch</h3>
<pre><code>__global__ void atomicSwap(int *locks, int *data) {
    int tid = threadIdx.x;
    
    // Spinlock acquisition
    while (atomicExch(&locks[tid], 1) == 1) {
        // Wait
    }
    
    // Critical section
    data[tid] *= 2;
    
    // Release lock
    atomicExch(&locks[tid], 0);
}</code></pre>
<p>Atomic exchange (swap).</p>
</div>
</div>

**Available Atomic Operations:**
- `atomicAdd`, `atomicSub`, `atomicExch`, `atomicMin`, `atomicMax`
- `atomicInc`, `atomicDec`, `atomicAnd`, `atomicOr`, `atomicXor`
- `atomicCAS` (compare-and-swap): foundation for custom atomics

**Performance Considerations:**
- Atomics serialize conflicting operations → can become bottleneck
- Use shared memory atomics when possible (faster than global)
- Consider reduction patterns instead of global atomicAdd when possible

### Cooperative Groups: Flexible Synchronization

Cooperative Groups (CUDA 9.0+) provide flexible thread coordination:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperativeKernel(float *data, int N) {
    // Get thread block group
    cg::thread_block block = cg::this_thread_block();
    
    int tid = block.thread_rank();
    int globalIdx = block.group_index().x * block.size() + tid;
    
    // Load data into shared memory
    __shared__ float sharedData[256];
    if (globalIdx < N) {
        sharedData[tid] = data[globalIdx];
    }
    
    block.sync();  // Equivalent to __syncthreads()
    
    // Work with tile (sub-group of threads)
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    
    // Warp-level reduction using tile
    float value = sharedData[tid];
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        value += tile32.shfl_down(value, offset);
    }
}
```

**Grid-Wide Synchronization** (Volta+, requires special launch):
```cpp
__global__ void gridSyncKernel(float *data) {
    cg::grid_group grid = cg::this_grid();
    
    // Phase 1: compute
    // ...
    
    grid.sync();  // Synchronize ALL blocks in grid!
    
    // Phase 2: use results from all blocks
    // ...
}

// Launch with cooperative groups API
void* kernelArgs[] = { &d_data };
cudaLaunchCooperativeKernel((void*)gridSyncKernel, 
    gridDim, blockDim, kernelArgs, 0, 0);
```

### Host Synchronization: cudaDeviceSynchronize()

Wait for all GPU operations to complete:

```cpp
kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();  // Block until kernel finishes
// Now safe to access results
```

**When to Synchronize:**
- Before reading kernel results on host
- Before timing measurements
- Before freeing memory used by async operations
- Between dependent kernel launches (usually automatic)

---

## 6. CUDA Streams & Concurrency

CUDA streams enable **overlapping computation with data transfer**, dramatically improving GPU utilization.

### Understanding Streams

A **stream** is a sequence of operations that execute in order. Operations in different streams can execute concurrently.

<div class="diagram">
<div class="diagram-title">Stream Execution Model</div>
<div class="diagram-grid cols-2">
<div class="diagram-card accent">
<h3>Default Stream (Stream 0)</h3>
<ul>
<li><strong>Behavior:</strong> Serializes all operations</li>
<li><strong>Synchronization:</strong> Blocks on all prior work</li>
<li><strong>Use:</strong> Simple programs, debugging</li>
</ul>
<pre><code>// Default stream (implicit)
kernel1<<<grid, block>>>(data);
cudaMemcpy(h_data, d_data, size, ...);
// All sequential</code></pre>
</div>

<div class="diagram-card green">
<h3>Non-Default Streams</h3>
<ul>
<li><strong>Behavior:</strong> Execute concurrently with other streams</li>
<li><strong>Synchronization:</strong> Independent unless explicitly synchronized</li>
<li><strong>Use:</strong> Overlapping compute and memory transfers</li>
</ul>
<pre><code>cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel<<<grid, block, 0, stream1>>>(data1);
kernel<<<grid, block, 0, stream2>>>(data2);
// Execute concurrently!</code></pre>
</div>
</div>
</div>

### Asynchronous Memory Transfers

**Synchronous (blocking) transfer:**
```cpp
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// CPU waits until transfer completes
```

**Asynchronous (non-blocking) transfer:**
```cpp
// Host memory must be pinned!
float *h_data_pinned;
cudaHostAlloc(&h_data_pinned, size, cudaHostAllocDefault);

cudaStream_t stream;
cudaStreamCreate(&stream);

// Initiate transfer (returns immediately)
cudaMemcpyAsync(d_data, h_data_pinned, size, cudaMemcpyHostToDevice, stream);

// CPU can do other work here while transfer happens
// ...

cudaStreamSynchronize(stream);  // Wait for stream operations
```

**Pinned Memory** (required for async transfers):
```cpp
float *h_data;
// Regular malloc: pageable memory (cannot overlap)
h_data = (float*)malloc(size);

float *h_data_pinned;
// Pinned memory: locked in RAM, DMA-capable
cudaHostAlloc(&h_data_pinned, size, cudaHostAllocDefault);
// or
cudaMallocHost(&h_data_pinned, size);

cudaFreeHost(h_data_pinned);
```

### Overlapping Compute and Transfer

The classic pattern: break data into chunks and pipeline them:

```cpp
#define NUM_STREAMS 4
#define CHUNK_SIZE (N / NUM_STREAMS)

cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// Pinned host memory
float *h_data, *d_data;
cudaMallocHost(&h_data, N * sizeof(float));
cudaMalloc(&d_data, N * sizeof(float));

// Pipeline: transfer and compute overlap
for (int i = 0; i < NUM_STREAMS; i++) {
    int offset = i * CHUNK_SIZE;
    
    // Async transfer chunk i
    cudaMemcpyAsync(&d_data[offset], &h_data[offset], 
                    CHUNK_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    
    // Launch kernel on chunk i (overlaps with next transfer)
    int blocks = (CHUNK_SIZE + 255) / 256;
    kernel<<<blocks, 256, 0, streams[i]>>>(&d_data[offset], CHUNK_SIZE);
    
    // Async transfer results back
    cudaMemcpyAsync(&h_data[offset], &d_data[offset],
                    CHUNK_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
}

// Wait for all streams
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

<div class="diagram">
<div class="diagram-title">Stream Pipelining Timeline</div>
<div class="flow">
<div class="flow-node accent wide">Stream 0: Transfer H→D | Kernel | Transfer D→H</div>
<div class="flow-node green wide">Stream 1:     Transfer H→D | Kernel | Transfer D→H</div>
<div class="flow-node purple wide">Stream 2:         Transfer H→D | Kernel | Transfer D→H</div>
<div class="flow-node cyan wide">Stream 3:             Transfer H→D | Kernel | Transfer D→H</div>
</div>
<p style="text-align: center; margin-top: 1rem;"><strong>Timeline →</strong></p>
<p style="text-align: center;">Operations overlap significantly, reducing total execution time by ~2-3×</p>
</div>

### CUDA Events: Precise Timing and Synchronization

Events mark points in streams and measure elapsed time:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Record start event
cudaEventRecord(start, stream);

kernel<<<grid, block, 0, stream>>>(data, N);

// Record stop event
cudaEventRecord(stop, stream);

// Wait for event completion
cudaEventSynchronize(stop);

// Calculate elapsed time
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**Stream Synchronization with Events:**
```cpp
cudaEvent_t event;
cudaEventCreate(&event);

kernel1<<<grid, block, 0, stream1>>>(data);
cudaEventRecord(event, stream1);

// Make stream2 wait for stream1's event
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<grid, block, 0, stream2>>>(data);  // Waits for kernel1
```

---

## 7. Error Handling

Robust CUDA programs always check for errors. Many CUDA functions return `cudaError_t`:

### Error Checking Patterns

**Macro for Runtime API calls:**
```cpp
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            fprintf(stderr, "Error code: %d (%s)\n", \
                    err, cudaGetErrorName(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage:
CHECK_CUDA(cudaMalloc(&d_data, size));
CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

**Checking Kernel Launch Errors:**
```cpp
kernel<<<grid, block>>>(data, N);

// Check for launch errors
cudaError_t launchErr = cudaGetLastError();
if (launchErr != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", 
            cudaGetErrorString(launchErr));
}

// Check for execution errors
cudaError_t syncErr = cudaDeviceSynchronize();
if (syncErr != cudaSuccess) {
    fprintf(stderr, "Kernel execution failed: %s\n",
            cudaGetErrorString(syncErr));
}
```

**Production-Ready Error Handling:**
```cpp
class CudaException : public std::exception {
private:
    std::string message;
    cudaError_t error;
public:
    CudaException(cudaError_t err, const char* file, int line) 
        : error(err) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << ": "
            << cudaGetErrorString(err) << " (" << cudaGetErrorName(err) << ")";
        message = oss.str();
    }
    
    const char* what() const noexcept override {
        return message.c_str();
    }
    
    cudaError_t getError() const { return error; }
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CudaException(err, __FILE__, __LINE__); \
        } \
    } while(0)

// Usage with C++ exception handling:
try {
    CUDA_CHECK(cudaMalloc(&d_data, size));
    kernel<<<grid, block>>>(d_data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
} catch (const CudaException& e) {
    std::cerr << e.what() << std::endl;
    // Cleanup and recovery
}
```

### Common Error Codes

| **Error Code** | **Meaning** | **Common Causes** |
|----------------|-------------|-------------------|
| `cudaSuccess` | No error | - |
| `cudaErrorMemoryAllocation` | Out of memory | Allocating more than GPU DRAM |
| `cudaErrorInvalidValue` | Invalid argument | Null pointers, invalid sizes |
| `cudaErrorLaunchOutOfResources` | Too many resources | Too much shared memory, registers |
| `cudaErrorInvalidConfiguration` | Invalid kernel config | Block size > 1024, invalid grid dimensions |
| `cudaErrorIllegalAddress` | Invalid memory access | Out-of-bounds access, unaligned access |

---

## 8. Performance Optimization

Writing correct CUDA code is just the beginning. Achieving peak performance requires understanding hardware characteristics and optimization techniques.

### Memory Coalescing: The #1 Optimization

**Coalesced Access** (fast): Threads in a warp access consecutive memory addresses.

```cpp
// GOOD: Coalesced access (sequential, aligned)
__global__ void coalescedAccess(float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = tid;  // Each thread accesses consecutive address
    }
    // Threads 0-31 (warp 0) access data[0:31] → single 128-byte transaction
}
```

**Uncoalesced Access** (slow): Threads access scattered or misaligned addresses.

```cpp
// BAD: Uncoalesced access (strided)
__global__ void stridedAccess(float *data, int N, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid * stride] = tid;  // Threads access strided addresses
    }
    // Threads 0-31 access data[0, stride, 2*stride, ...] → 32 transactions!
}
```

**Performance Impact:**
- Coalesced: **1 memory transaction per warp** (128 bytes)
- Uncoalesced (stride=32): **32 memory transactions per warp** → 32× slower!

**Matrix Access Pattern:**
```cpp
// Column-major access (bad for row-major storage)
__global__ void columnMajor(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        float value = matrix[col * height + row];  // Strided access!
    }
}

// Row-major access (good for row-major storage)
__global__ void rowMajor(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        float value = matrix[row * width + col];  // Coalesced!
    }
}
```

### Shared Memory Bank Conflicts

Shared memory is divided into **32 banks** (4-byte words). Multiple threads accessing the same bank cause serialization.

```cpp
// BAD: Bank conflicts (all threads access stride=32)
__shared__ float sharedData[1024];
int tid = threadIdx.x;
float value = sharedData[tid * 32];  // 32-way bank conflict!

// GOOD: No bank conflicts (sequential access)
float value = sharedData[tid];  // Each thread accesses different bank
```

**Avoiding Bank Conflicts:**
- Ensure threads in a warp access different banks
- Padding arrays to avoid pathological access patterns:
  ```cpp
  __shared__ float sharedData[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
  ```

### Loop Unrolling

Reduce loop overhead and increase instruction-level parallelism:

```cpp
// Manual unrolling
__global__ void unrolledLoop(float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Process 4 elements per thread
    int base = tid * 4;
    if (base + 3 < N) {
        data[base] *= 2.0f;
        data[base + 1] *= 2.0f;
        data[base + 2] *= 2.0f;
        data[base + 3] *= 2.0f;
    }
}

// Compiler-directed unrolling
__global__ void pragmaUnroll(float *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    #pragma unroll 8  // Unroll next loop 8 times
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        if (idx < N) {
            data[idx] *= 2.0f;
        }
    }
}
```

### Occupancy Optimization

**Occupancy** = (Active warps per SM) / (Maximum warps per SM)

Higher occupancy → better latency hiding → better performance (usually).

**Factors affecting occupancy:**
1. **Threads per block**: 128-512 is typical sweet spot
2. **Registers per thread**: Fewer registers → more blocks resident
3. **Shared memory per block**: Less shared memory → more blocks resident

**CUDA Occupancy Calculator:**
```cpp
// Query occupancy
int blockSize;      // Input
int minGridSize;    // Returned minimum grid size for max occupancy
int maxActiveBlocks; // Returned occupancy

cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                    myKernel, dynamicSharedMemSize, 0);

printf("Suggested block size: %d\n", blockSize);
printf("Minimum grid size for max occupancy: %d\n", minGridSize);
```

### Register Usage Optimization

```cpp
// Compile with register usage info
// nvcc -Xptxas -v kernel.cu
// Output: "Used 32 registers, 512 bytes shared memory"

// Limit registers per thread to increase occupancy
// nvcc -maxrregcount=32 kernel.cu
```

### Instruction-Level Parallelism

Modern GPUs benefit from multiple independent operations:

```cpp
// Poor ILP: dependencies
__global__ void poorILP(float *data) {
    int tid = threadIdx.x;
    float x = data[tid];
    x = x + 1.0f;    // Depends on previous line
    x = x * 2.0f;    // Depends on previous line
    x = x - 0.5f;    // Depends on previous line
    data[tid] = x;
}

// Good ILP: independent operations
__global__ void goodILP(float *data, int N) {
    int tid = threadIdx.x;
    int base = tid * 4;
    
    // Load 4 independent values
    float x0 = data[base];
    float x1 = data[base + 1];
    float x2 = data[base + 2];
    float x3 = data[base + 3];
    
    // Compute independently (can execute in parallel)
    x0 = x0 * 2.0f + 1.0f;
    x1 = x1 * 2.0f + 1.0f;
    x2 = x2 * 2.0f + 1.0f;
    x3 = x3 * 2.0f + 1.0f;
    
    // Store
    data[base] = x0;
    data[base + 1] = x1;
    data[base + 2] = x2;
    data[base + 3] = x3;
}
```

### Profiling with Nsight Compute

```bash
# Profile kernel with Nsight Compute
ncu --set full -o profile_output ./my_cuda_app

# Key metrics to examine:
# - Memory throughput (% of peak bandwidth)
# - SM efficiency (% of time SMs are active)
# - Occupancy (achieved vs. theoretical)
# - Warp execution efficiency (divergence)
# - Memory coalescing efficiency
```

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<h3>Memory Optimizations</h3>
<ul>
<li>✅ Coalesced access patterns</li>
<li>✅ Use shared memory for reuse</li>
<li>✅ Avoid bank conflicts</li>
<li>✅ Prefer L1/texture cache for reads</li>
<li>✅ Use streaming loads for non-cached data</li>
</ul>
</div>

<div class="diagram-card green">
<h3>Compute Optimizations</h3>
<ul>
<li>✅ High occupancy (not always max)</li>
<li>✅ Minimize divergence</li>
<li>✅ Loop unrolling</li>
<li>✅ Instruction-level parallelism</li>
<li>✅ Use intrinsics (__fmaf, __expf)</li>
</ul>
</div>

<div class="diagram-card purple">
<h3>Launch Configuration</h3>
<ul>
<li>✅ Block size: 128-512 threads</li>
<li>✅ Multiple of warp size (32)</li>
<li>✅ Enough blocks to saturate GPU</li>
<li>✅ Balance occupancy vs. resources</li>
<li>✅ Consider persistent kernels for small tasks</li>
</ul>
</div>
</div>

---

## 9. CUDA from Python

Python has become the lingua franca of scientific computing and AI. Multiple frameworks enable CUDA programming from Python:

### Option 1: PyCUDA — Direct CUDA from Python

Write CUDA kernels as strings in Python:

```python
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

# Define CUDA kernel as string
kernel_code = """
__global__ void vector_add(float *a, float *b, float *c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
"""

# Compile kernel
mod = SourceModule(kernel_code)
vector_add = mod.get_function("vector_add")

# Prepare data
N = 1000000
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)

# Launch kernel
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

vector_add(
    drv.In(a), drv.In(b), drv.Out(c), np.int32(N),
    block=(threads_per_block, 1, 1),
    grid=(blocks_per_grid, 1)
)

print(f"Result: {c[:10]}")
print(f"Expected: {(a + b)[:10]}")
```

**PyCUDA Features:**
- Direct kernel compilation at runtime
- Automatic memory management with `gpuarray`
- Integration with NumPy
- Suitable for research and prototyping

### Option 2: Numba — JIT CUDA Compilation

Numba compiles Python functions to CUDA kernels via decorators:

```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def vector_add_numba(a, b, c):
    tid = cuda.grid(1)  # Equivalent to threadIdx.x + blockIdx.x * blockDim.x
    if tid < c.size:
        c[tid] = a[tid] + b[tid]

# Prepare data
N = 1000000
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)

# Copy to device
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.device_array_like(c)

# Launch kernel
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
vector_add_numba[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy back
c = d_c.copy_to_host()
print(f"Result: {c[:10]}")
```

**Numba Advanced Example with Shared Memory:**

```python
@cuda.jit
def matmul_shared(A, B, C):
    # Define shared memory
    sA = cuda.shared.array(shape=(16, 16), dtype=float32)
    sB = cuda.shared.array(shape=(16, 16), dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx
    
    tmp = 0.0
    for tile in range((A.shape[1] + 15) // 16):
        # Load tile into shared memory
        if row < A.shape[0] and tile * 16 + tx < A.shape[1]:
            sA[ty, tx] = A[row, tile * 16 + tx]
        else:
            sA[ty, tx] = 0.0
            
        if col < B.shape[1] and tile * 16 + ty < B.shape[0]:
            sB[ty, tx] = B[tile * 16 + ty, col]
        else:
            sB[ty, tx] = 0.0
            
        cuda.syncthreads()
        
        # Compute partial product
        for k in range(16):
            tmp += sA[ty, k] * sB[k, tx]
            
        cuda.syncthreads()
    
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp
```

### Option 3: CuPy — NumPy on CUDA

CuPy is a drop-in replacement for NumPy with GPU acceleration:

```python
import cupy as cp
import numpy as np

# Create arrays (automatically on GPU)
a = cp.random.randn(1000000, dtype=cp.float32)
b = cp.random.randn(1000000, dtype=cp.float32)

# NumPy-like operations (executed on GPU)
c = a + b
d = cp.sin(a) * cp.cos(b)
e = cp.sum(c)

# Convert back to NumPy
c_cpu = cp.asnumpy(c)

# Matrix multiplication
A = cp.random.randn(1000, 1000, dtype=cp.float32)
B = cp.random.randn(1000, 1000, dtype=cp.float32)
C = cp.dot(A, B)  # GPU-accelerated

# Custom kernel in CuPy
multiply_kernel = cp.ElementwiseKernel(
    'float32 x, float32 y',  # Input types
    'float32 z',             # Output type
    'z = x * y * 2',         # Operation
    'multiply_by_two'        # Kernel name
)

result = multiply_kernel(a, b)
```

**CuPy Custom RawKernel:**

```python
kernel_code = '''
extern "C" __global__
void my_kernel(const float* a, const float* b, float* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
'''

my_kernel = cp.RawKernel(kernel_code, 'my_kernel')

N = 1000000
a = cp.random.randn(N, dtype=cp.float32)
b = cp.random.randn(N, dtype=cp.float32)
c = cp.zeros(N, dtype=cp.float32)

threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

my_kernel((blocks_per_grid,), (threads_per_block,), (a, b, c, N))
```

### Option 4: PyTorch Custom CUDA Extensions

Extend PyTorch with custom CUDA kernels:

```python
from torch.utils.cpp_extension import load_inline

cuda_source = '''
#include <torch/extension.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    int N = a.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );
    
    return c;
}
'''

cpp_source = '''
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
'''

# Compile and load
module = load_inline(
    name='custom_add',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['add_cuda']
)

# Use in PyTorch
import torch
a = torch.randn(1000000, device='cuda')
b = torch.randn(1000000, device='cuda')
c = module.add_cuda(a, b)

print(c[:10])
```

### Option 5: Triton — Python-like GPU Programming

OpenAI's Triton offers Python-like syntax with automatic optimization:

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary conditions
    mask = offsets < N
    
    # Load data
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute
    c = a + b
    
    # Store result
    tl.store(c_ptr + offsets, c, mask=mask)

def add_triton(a: torch.Tensor, b: torch.Tensor):
    c = torch.empty_like(a)
    N = a.numel()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    return c

# Use it
a = torch.randn(1000000, device='cuda')
b = torch.randn(1000000, device='cuda')
c = add_triton(a, b)
```

**Triton Advanced: Fused Softmax**

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Load row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store
    tl.store(output_ptr + row_start + cols, softmax_output, mask=mask)
```

### Python CUDA Framework Comparison

| **Framework** | **Ease of Use** | **Performance** | **Flexibility** | **Best For** |
|---------------|-----------------|-----------------|-----------------|--------------|
| **PyCUDA** | Medium | High | Very High | Custom kernels, research |
| **Numba** | High | High | High | Rapid prototyping, NumPy users |
| **CuPy** | Very High | High | Medium | Drop-in NumPy replacement |
| **PyTorch Extensions** | Medium | Very High | High | ML model optimization |
| **Triton** | High | Very High | Medium | Fused kernels, modern syntax |

---

## 10. Advanced CUDA Techniques

### Dynamic Parallelism: Kernels Launching Kernels

Introduced in Kepler (compute capability 3.5), kernels can launch child kernels:

```cpp
__global__ void childKernel(float *data, int depth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = depth;
}

__global__ void parentKernel(float *data, int depth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (depth < 5) {  // Recursion limit
        // Parent does work
        data[tid] = depth * 10;
        
        // Launch child kernel from GPU!
        if (threadIdx.x == 0) {
            childKernel<<<16, 256>>>(data, depth + 1);
        }
    }
}

// Host launches parent
parentKernel<<<32, 256>>>(d_data, 0);
cudaDeviceSynchronize();  // Wait for all kernels (parent + children)
```

**Use Cases:**
- Recursive algorithms (quicksort, tree traversal)
- Adaptive mesh refinement
- Dynamic load balancing
- Graph algorithms with varying workloads

**Considerations:**
- Child kernel launches have overhead (~5-10 µs)
- Limited nesting depth (implementation dependent)
- Careful synchronization required

### CUDA Graphs: Capture and Replay

CUDA Graphs (CUDA 10+) reduce launch overhead by capturing a sequence of operations:

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

// Begin capture on stream
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Record operations (kernels, memcpy, etc.)
kernel1<<<grid, block, 0, stream>>>(data1);
kernel2<<<grid, block, 0, stream>>>(data2);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);

// End capture
cudaStreamEndCapture(stream, &graph);

// Instantiate executable graph
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// Launch graph (replays all captured operations)
cudaGraphLaunch(instance, stream);
cudaStreamSynchronize(stream);

// Can relaunch many times with minimal overhead
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);
}

cudaGraphExecDestroy(instance);
cudaGraphDestroy(graph);
```

**Performance Benefits:**
- Reduces CPU overhead by **~40-60%**
- Single submission for entire workflow
- Better optimization opportunities for driver

**Manual Graph Construction:**

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

cudaGraphNode_t kernelNode, memcpyNode;
cudaKernelNodeParams kernelParams = {0};
cudaMemcpy3DParms memcpyParams = {0};

// Add kernel node
kernelParams.func = (void*)myKernel;
kernelParams.gridDim = dim3(blocks, 1, 1);
kernelParams.blockDim = dim3(threads, 1, 1);
kernelParams.kernelParams = args;

cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams);

// Add memcpy node (depends on kernel)
memcpyParams.srcPtr = make_cudaPitchedPtr(d_data, size, width, height);
memcpyParams.dstPtr = make_cudaPitchedPtr(h_data, size, width, height);
memcpyParams.kind = cudaMemcpyDeviceToHost;

cudaGraphNode_t deps[] = {kernelNode};
cudaGraphAddMemcpyNode(&memcpyNode, graph, deps, 1, &memcpyParams);
```

### Tensor Memory Accelerator (TMA) in Hopper

Hopper architecture (H100) introduces TMA for efficient data movement:

```cpp
// TMA enables asynchronous bulk data transfers between global and shared memory
__global__ void tmaExample(float *global_data) {
    __shared__ float shared_tile[128][128];
    
    // TMA load: hardware-managed transfer
    // Syntax and exact API subject to CUDA version
    if (threadIdx.x == 0) {
        // Asynchronous bulk copy from global to shared using TMA
        // (Requires CUDA 12+ and Hopper GPU)
        __pipeline_memcpy_async(shared_tile, global_data, 
                                 sizeof(float) * 128 * 128);
    }
    
    __syncthreads();
    
    // Use data from shared memory
    int tid = threadIdx.x;
    float value = shared_tile[tid / 128][tid % 128];
}
```

**TMA Benefits:**
- Hardware-accelerated bulk transfers
- Reduced register pressure
- Better latency hiding
- Optimized for Transformer workloads

### Warp-Level Primitives

Modern CUDA exposes warp-level operations for fine-grained control:

```cpp
__global__ void warpPrimitives(int *data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int value = (tid < N) ? data[tid] : 0;
    
    // Warp shuffle: exchange data between threads in warp
    // Get value from thread (tid + 1) in warp
    int neighbor = __shfl_down_sync(0xffffffff, value, 1);
    
    // Warp vote: query condition across warp
    int all_positive = __all_sync(0xffffffff, value > 0);
    int any_negative = __any_sync(0xffffffff, value < 0);
    
    // Ballot: get bitmask of condition
    unsigned int mask = __ballot_sync(0xffffffff, value % 2 == 0);
    
    // Match: find threads with same value
    unsigned int match_mask = __match_any_sync(0xffffffff, value);
    
    // Warp reduction (without shared memory!)
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    // Thread 0 in each warp now has warp sum
}
```

**Warp Primitives Use Cases:**
- Fast reductions (sum, max, min)
- Broadcasting within warp
- Warp-level voting/consensus
- Prefix sums (scan operations)
- Replacing shared memory for small aggregations

### Cooperative Groups: Advanced Thread Collaboration

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void advancedCooperativeGroups(float *data, int N) {
    // Grid group (requires special launch)
    cg::grid_group grid = cg::this_grid();
    
    // Thread block
    cg::thread_block block = cg::this_thread_block();
    
    // Tile (subgroup of block)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    cg::thread_block_tile<4> quad = cg::tiled_partition<4>(warp);
    
    int tid = grid.thread_rank();
    
    // Phase 1: All threads work
    float value = (tid < N) ? data[tid] : 0.0f;
    
    // Warp reduction using tile
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        value += warp.shfl_down(value, offset);
    }
    
    // Synchronize entire grid (all blocks!)
    grid.sync();
    
    // Phase 2: Use globally reduced results
    if (warp.thread_rank() == 0) {
        // Write warp sum
        data[tid / 32] = value;
    }
}

// Special launch for grid synchronization
void launchCooperativeKernel(float *data, int N) {
    int device;
    cudaGetDevice(&device);
    
    // Check if device supports cooperative launch
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, 
                          cudaDevAttrCooperativeLaunch, device);
    
    if (supportsCoopLaunch) {
        dim3 grid(256);
        dim3 block(256);
        void* args[] = { &data, &N };
        
        cudaLaunchCooperativeKernel((void*)advancedCooperativeGroups,
                                   grid, block, args);
    }
}
```

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<h3>Dynamic Parallelism</h3>
<p><strong>When:</strong> Recursive algorithms, irregular workloads</p>
<p><strong>Performance:</strong> Overhead per child launch</p>
<p><strong>Complexity:</strong> High (nested synchronization)</p>
</div>

<div class="diagram-card green">
<h3>CUDA Graphs</h3>
<p><strong>When:</strong> Repeated workflow, low-latency</p>
<p><strong>Performance:</strong> 40-60% overhead reduction</p>
<p><strong>Complexity:</strong> Medium (capture/replay model)</p>
</div>

<div class="diagram-card purple">
<h3>Cooperative Groups</h3>
<p><strong>When:</strong> Grid-wide sync, flexible grouping</p>
<p><strong>Performance:</strong> Hardware-accelerated sync</p>
<p><strong>Complexity:</strong> Medium (new programming model)</p>
</div>
</div>

---

## Summary: The CUDA Programming Journey

You've now explored the complete landscape of CUDA programming—from fundamental concepts to advanced optimization techniques. Let's recap the key insights:

<div class="diagram">
<div class="diagram-title">CUDA Mastery Path</div>
<div class="timeline">
<div class="timeline-item">
<div class="timeline-marker accent"></div>
<div class="timeline-content">
<h4>Level 1: Foundations</h4>
<p>Understanding host-device model, grid-block-thread hierarchy, basic kernel launches. Writing correct CUDA programs with proper memory management.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker green"></div>
<div class="timeline-content">
<h4>Level 2: Memory Optimization</h4>
<p>Mastering memory hierarchy, coalesced access, shared memory usage, avoiding bank conflicts. Achieving 10× performance gains.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker purple"></div>
<div class="timeline-content">
<h4>Level 3: Concurrency & Synchronization</h4>
<p>Leveraging streams for overlapped execution, precise timing with events, proper use of synchronization primitives. Maximizing GPU utilization.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker cyan"></div>
<div class="timeline-content">
<h4>Level 4: Advanced Techniques</h4>
<p>Dynamic parallelism, CUDA Graphs, warp-level primitives, cooperative groups. Achieving near-peak hardware performance.</p>
</div>
</div>
<div class="timeline-item">
<div class="timeline-marker orange"></div>
<div class="timeline-content">
<h4>Level 5: Production Excellence</h4>
<p>Profiling with Nsight, robust error handling, integration with high-level frameworks (Python/PyTorch), architectural awareness (Ampere/Hopper). Building production-grade GPU applications.</p>
</div>
</div>
</div>
</div>

### Key Takeaways

**✅ CUDA revolutionized GPU programming** by making parallel computing accessible to C/C++ developers without graphics expertise.

**✅ The thread hierarchy** (grid → block → warp → thread) provides flexibility while mapping efficiently to GPU hardware.

**✅ Memory is king**: Coalesced access patterns and effective use of the memory hierarchy (registers → shared → global) determine performance.

**✅ Synchronization tools** (__syncthreads, atomics, cooperative groups) enable thread collaboration while avoiding race conditions.

**✅ Streams and concurrency** allow overlapping compute with data transfer, dramatically improving throughput.

**✅ Python integration** (PyCUDA, Numba, CuPy, Triton) brings CUDA to the AI/ML ecosystem, enabling rapid development with high performance.

**✅ Advanced techniques** (dynamic parallelism, CUDA Graphs, warp primitives) unlock the full potential of modern GPU architectures.

### Performance Optimization Hierarchy

```
1. Algorithm Selection (1000× impact)
   ↓
2. Memory Access Patterns (10-100× impact)
   ↓
3. Occupancy & Launch Configuration (2-5× impact)
   ↓
4. Instruction-Level Optimizations (1.2-2× impact)
   ↓
5. Architecture-Specific Features (<1.5× impact)
```

**Always optimize from top to bottom!** A better algorithm beats micro-optimizations every time.

### When to Use CUDA

| **Use CUDA When...** | **Consider Alternatives When...** |
|----------------------|-----------------------------------|
| ✅ Maximum performance on NVIDIA GPUs required | ❌ Cross-platform portability is critical (→ OpenCL, SYCL) |
| ✅ Access to full GPU hardware features needed | ❌ Rapid prototyping without optimization (→ Numba, CuPy) |
| ✅ Production AI/ML workloads (TensorRT, cuDNN) | ❌ CPU-bound workloads (GPU overhead not justified) |
| ✅ Scientific computing, simulations, HPC | ❌ Small datasets (transfer overhead dominates) |
| ✅ Custom kernels for unique algorithms | ❌ Standard operations covered by libraries (→ use cuBLAS, Thrust) |

### Resources for Continued Learning

- **NVIDIA CUDA Programming Guide**: [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/)
- **CUDA by Example** (Sanders & Kandrot): Classic introductory text
- **Programming Massively Parallel Processors** (Hwu, Kirk, Hajj): Deep dive into GPU architecture
- **Nsight Tools**: cuda-gdb, cuda-memcheck, Nsight Compute, Nsight Systems
- **CUDA Samples**: Extensive examples in CUDA Toolkit installation
- **GTC On-Demand**: Free technical talks from NVIDIA's GPU Technology Conference

---

**Next: [Chapter 4 — NVIDIA Libraries →](./04_nvidia_libraries.md)**

---

*Last updated: April 2026*
