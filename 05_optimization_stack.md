---
title: "Chapter 5 — NVIDIA Optimization Stack"
---

[← Back to Table of Contents](./README.md)

# Chapter 5: NVIDIA Optimization Stack

## TensorRT, TensorRT-LLM, ModelOpt & AITune

Once you've trained a deep learning model — investing days or weeks in GPU compute, hyperparameter tuning, and architecture search — the journey is only half complete. Deploying that model to production introduces an entirely new set of challenges: **inference latency**, **throughput requirements**, **memory constraints**, and **cost optimization**. A PyTorch model that achieves 95% accuracy during training might deliver only 20 queries per second in production — far from the 1000+ QPS required for real-time applications.

This is where NVIDIA's optimization stack enters. **TensorRT**, **TensorRT-LLM**, **ModelOpt (NVIDIA Model Optimizer)**, **AITune**, and **Triton Inference Server** form an integrated pipeline that transforms trained models into highly optimized inference engines. These tools don't just "speed things up" — they fundamentally restructure computation graphs, fuse operations, quantize weights, and exploit hardware-specific features (Tensor Cores, structured sparsity, FP8) to extract maximum performance from NVIDIA GPUs.

Understanding this optimization stack is critical for anyone deploying production AI systems. A well-optimized model can deliver 3-10× better throughput than naive PyTorch inference, translating directly to reduced infrastructure costs, lower latency, and better user experience. For large language models (LLMs), where a single forward pass might cost thousands of dollars at scale, optimization isn't optional — it's existential.

<div class="diagram">
<div class="diagram-title">The NVIDIA Inference Optimization Pipeline</div>
<div class="flow">
<div class="flow-node accent wide">Trained Model (PyTorch/TensorFlow/JAX)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">ModelOpt: Quantization + Pruning + Distillation</div>
<div class="flow-arrow green"></div>
<div class="flow-node purple wide">TensorRT/TensorRT-LLM: Kernel Fusion + Auto-Tuning</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">AITune: Batch Size + Memory Optimization</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan wide">Triton Inference Server: Dynamic Batching + Model Ensemble</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node teal wide">Production Deployment (3-10× Faster, 4× Lower Memory)</div>
</div>
</div>

---

## Table of Contents

1. [TensorRT — High-Performance Inference Engine](#1-tensorrt--high-performance-inference-engine)
2. [TensorRT-LLM — Purpose-Built LLM Optimization](#2-tensorrt-llm--purpose-built-llm-optimization)
3. [ModelOpt — Quantization, Pruning & Distillation](#3-modelopt--quantization-pruning--distillation)
4. [AITune — Automated Configuration Optimization](#4-aitune--automated-configuration-optimization)
5. [Triton Inference Server — Production Model Serving](#5-triton-inference-server--production-model-serving)
6. [The Complete Optimization Pipeline](#6-the-complete-optimization-pipeline)
7. [Performance Benchmarks](#7-performance-benchmarks)

---

## 1. TensorRT — High-Performance Inference Engine

### What is TensorRT?

**TensorRT** is NVIDIA's high-performance deep learning inference optimizer and runtime. Released in 2016, TensorRT takes trained neural networks (from PyTorch, TensorFlow, ONNX, or other frameworks) and applies a series of graph optimizations, precision calibration, and kernel auto-tuning to produce a highly optimized **inference engine** that executes 3-5× faster than native framework inference.

TensorRT is *not* a training framework — it's exclusively focused on deployment and inference. Think of it as a compiler that transforms a generic neural network into a GPU-specific execution plan optimized for minimal latency and maximum throughput.

### Why TensorRT?

Training frameworks prioritize flexibility: dynamic graphs, eager execution, automatic differentiation, research-friendly APIs. These features enable rapid experimentation but introduce overhead during inference:

- **Framework overhead**: Python interpreter, dynamic dispatch, operator fusion opportunities missed
- **Generic kernels**: One-size-fits-all CUDA kernels that don't exploit model-specific patterns
- **Unnecessary precision**: FP32 weights when FP16 or INT8 would suffice
- **Suboptimal memory usage**: Redundant allocations, fragmented memory, cache inefficiency

TensorRT addresses all of these:

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<div class="card-icon">🔗</div>
<div class="card-title">Layer Fusion</div>
<div class="card-desc">Combine Conv + BatchNorm + ReLU into single kernel, reducing memory bandwidth by 3×</div>
</div>
<div class="diagram-card green">
<div class="card-icon">⚡</div>
<div class="card-title">Precision Calibration</div>
<div class="card-desc">FP16/INT8 quantization with minimal accuracy loss, 2-4× throughput gain</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">🎯</div>
<div class="card-title">Kernel Auto-Tuning</div>
<div class="card-desc">Profile hundreds of kernel configurations, select optimal for target GPU</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">🧠</div>
<div class="card-title">Dynamic Tensors</div>
<div class="card-desc">Support variable batch sizes/sequence lengths without rebuilding engines</div>
</div>
<div class="diagram-card cyan">
<div class="card-icon">🔌</div>
<div class="card-title">Plugin System</div>
<div class="card-desc">Custom layers via C++ plugins for operations TensorRT doesn't natively support</div>
</div>
<div class="diagram-card teal">
<div class="card-icon">📊</div>
<div class="card-title">Multi-Stream Execution</div>
<div class="card-desc">Concurrent inference streams to maximize GPU utilization</div>
</div>
</div>

### How TensorRT Works — The Optimization Pipeline

TensorRT's optimization happens in two phases: **build time** (creating the optimized engine) and **runtime** (executing inference).

<div class="diagram">
<div class="diagram-title">TensorRT Build & Runtime Architecture</div>
<div class="flow">
<div class="flow-node accent">Input Model (ONNX / PyTorch / TF SavedModel)</div>
<div class="flow-arrow"></div>
<div class="flow-node green">Parser: Convert to TensorRT Network</div>
<div class="flow-arrow"></div>
<div class="flow-node purple wide">
<strong>Builder Optimizations</strong><br/>
├─ Layer & Tensor Fusion (Conv+BN+ReLU)<br/>
├─ Kernel Auto-Tuning (profile 100s of configs)<br/>
├─ Precision Calibration (FP32 → FP16 → INT8)<br/>
├─ Dynamic Range Analysis (for INT8)<br/>
├─ Memory Optimization (buffer reuse)<br/>
└─ Horizontal Fusion (parallel ops → single kernel)
</div>
<div class="flow-arrow"></div>
<div class="flow-node orange">Serialized Engine (.plan file)</div>
<div class="flow-arrow"></div>
<div class="flow-node cyan">Runtime: Deserialize Engine</div>
<div class="flow-arrow"></div>
<div class="flow-node pink">Execute Inference (enqueue → GPU kernels → outputs)</div>
</div>
</div>

#### Phase 1: Building the Engine (Offline Optimization)

**Step 1: Network Construction**

Import model from framework:
```python
import tensorrt as trt
from tensorrt import Builder, Logger, Runtime

# Create builder and network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Parse ONNX model
parser = trt.OnnxParser(network, logger)
with open("resnet50.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("ONNX parse failed")
```

**Step 2: Optimization Configuration**

Configure precision, calibration, and optimization profiles:
```python
config = builder.create_builder_config()

# Enable FP16 precision (2× faster on Tensor Cores)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# Enable INT8 quantization (4× faster, requires calibration)
if builder.platform_has_fast_int8:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = MyCalibrator()  # Custom calibration dataset

# Set workspace for kernel tuning (larger = more options)
config.max_workspace_size = 4 * (1 << 30)  # 4 GB

# Define optimization profile for dynamic shapes
profile = builder.create_optimization_profile()
profile.set_shape("input", 
                  min=(1, 3, 224, 224),    # Min batch size
                  opt=(8, 3, 224, 224),    # Optimal batch size
                  max=(32, 3, 224, 224))   # Max batch size
config.add_optimization_profile(profile)
```

**Step 3: Layer Fusion & Graph Optimization**

TensorRT automatically applies these transformations:

| Optimization | Description | Example | Speedup |
|--------------|-------------|---------|---------|
| **Vertical Fusion** | Merge sequential layers into single kernel | Conv + BN + ReLU → CBR kernel | 3× (eliminates 2 memory roundtrips) |
| **Horizontal Fusion** | Execute parallel branches simultaneously | Two independent Conv layers → fused kernel | 1.5× (better SM utilization) |
| **Eliminate Layer Noop** | Remove identity operations | BatchNorm with scale=1, bias=0 | 1.2× |
| **Concat Elimination** | Fuse concat into preceding/following layers | Conv → Concat → Conv becomes Conv | 1.4× |
| **Reduced Precision** | Use FP16/INT8 Tensor Cores | FP32 matmul → FP16 Tensor Core matmul | 2-4× |

**Step 4: Kernel Auto-Tuning**

For each layer, TensorRT profiles hundreds of kernel implementations (different tile sizes, thread block configurations, memory access patterns) and selects the fastest:

```python
# Auto-tuning happens during build (may take minutes)
# Example: For a 512×512 matrix multiply, TensorRT tests:
#   - cuBLAS implementations (various algorithms)
#   - cuDNN implementations (if convolution-like)
#   - Custom fused kernels
#   - Different Tensor Core configurations (TF32, FP16, INT8)
#   - Various tile sizes (32×32, 64×64, 128×128)

engine = builder.build_engine(network, config)  # Runs auto-tuning
```

**Step 5: Engine Serialization**

Save optimized engine to disk:
```python
with open("resnet50_fp16.plan", "wb") as f:
    f.write(engine.serialize())
```

#### Phase 2: Runtime Inference (Online Execution)

**Step 1: Deserialize Engine**

```python
runtime = trt.Runtime(logger)
with open("resnet50_fp16.plan", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
```

**Step 2: Create Execution Context**

```python
context = engine.create_execution_context()

# For dynamic shapes, set input dimensions
context.set_binding_shape(0, (8, 3, 224, 224))  # batch=8
```

**Step 3: Allocate GPU Buffers**

```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Allocate input/output buffers
input_shape = (8, 3, 224, 224)
output_shape = (8, 1000)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.float16().itemsize)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float16().itemsize)

bindings = [int(d_input), int(d_output)]
```

**Step 4: Execute Inference**

```python
# Copy input to GPU
input_data = preprocess_image(image)  # NumPy array
cuda.memcpy_htod(d_input, input_data)

# Execute inference
stream = cuda.Stream()
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
stream.synchronize()

# Copy output from GPU
output_data = np.empty(output_shape, dtype=np.float16)
cuda.memcpy_dtoh(output_data, d_output)
```

### Dynamic Shapes — Supporting Variable Inputs

Real-world applications often require variable batch sizes (user request patterns vary) or sequence lengths (text varies in length). TensorRT 8+ supports **dynamic shapes** via optimization profiles:

```python
# Define multiple optimization points
profile = builder.create_optimization_profile()

# Input: [batch, channels, height, width]
profile.set_shape("input",
    min=(1, 3, 224, 224),    # Single image
    opt=(16, 3, 224, 224),   # Sweet spot for throughput
    max=(64, 3, 224, 224))   # Max batch for memory constraints

# For transformers: [batch, sequence_length]
profile.set_shape("input_ids",
    min=(1, 16),     # Short sequence
    opt=(8, 128),    # Typical sentence
    max=(32, 512))   # Long document

config.add_optimization_profile(profile)
```

At runtime, set actual dimensions:
```python
context.set_binding_shape(0, (batch_size, 3, 224, 224))
if not context.all_binding_shapes_specified:
    raise RuntimeError("Not all shapes specified")
context.execute_v2(bindings)
```

### INT8 Quantization — 4× Throughput with Calibration

**Why INT8?**
- **4× memory bandwidth reduction** (8 bits vs 32 bits)
- **4× higher compute throughput** (Turing/Ampere INT8 Tensor Cores: 4× FP16 performance)
- **~1-2% accuracy loss** for most computer vision models with proper calibration

**Calibration Process:**

INT8 requires computing dynamic ranges for each activation tensor (weights can be quantized without calibration):

```python
import tensorrt as trt

class ImageNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.batch_size = data_loader.batch_size
        self.current_index = 0
        
        # Allocate GPU buffer for calibration batch
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 224 * 224 * 4)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index < len(self.data_loader):
            batch = next(iter(self.data_loader))
            cuda.memcpy_htod(self.device_input, batch)
            self.current_index += 1
            return [int(self.device_input)]
        else:
            return None
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Use calibrator during build
calibrator = ImageNetCalibrator(calibration_loader, "resnet50_int8.cache")
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = calibrator
```

**Quantization Formula:**

For each activation tensor, TensorRT computes:
- **Dynamic range**: `[min_activation, max_activation]` across calibration dataset
- **Scale factor**: `scale = 127.0 / max(abs(min_activation), abs(max_activation))`
- **Quantized value**: `int8_value = clip(round(fp32_value * scale), -128, 127)`

### Plugin API — Custom Layers

When TensorRT doesn't natively support an operation (e.g., custom attention mechanism, specialized pooling), implement a **plugin**:

```cpp
// Custom plugin example: LeakyReLU with configurable slope
class LeakyReLUPlugin : public IPluginV2DynamicExt {
public:
    LeakyReLUPlugin(float negativeSlope) : mNegativeSlope(negativeSlope) {}
    
    int getNbOutputs() const noexcept override { return 1; }
    
    DimsExprs getOutputDimensions(int outputIndex, 
                                  const DimsExprs* inputs,
                                  int nbInputs,
                                  IExprBuilder& exprBuilder) noexcept override {
        return inputs[0];  // Output shape = input shape
    }
    
    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override {
        
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);
        int count = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] * 
                    inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
        
        // Launch custom CUDA kernel
        leakyReLUKernel<<<(count+255)/256, 256, 0, stream>>>(
            input, output, count, mNegativeSlope);
        
        return 0;
    }

private:
    float mNegativeSlope;
};
```

### Performance Benchmarks — TensorRT vs PyTorch

**ResNet-50 Inference (ImageNet, NVIDIA A100 GPU, Batch=32, FP16):**

| Framework | Latency (ms) | Throughput (imgs/sec) | Speedup |
|-----------|--------------|----------------------|---------|
| PyTorch Eager | 28.4 | 1,127 | 1.0× (baseline) |
| PyTorch TorchScript | 19.2 | 1,667 | 1.5× |
| PyTorch torch.compile | 14.1 | 2,270 | 2.0× |
| TensorRT FP16 | 6.8 | 4,706 | 4.2× |
| TensorRT INT8 | 3.2 | 10,000 | 8.9× |

**BERT-Base Inference (SQuAD, A100, Batch=8, Seq=128):**

| Framework | Latency (ms) | Throughput (seq/sec) | Memory (GB) |
|-----------|--------------|---------------------|-------------|
| PyTorch FP32 | 12.4 | 645 | 2.8 |
| PyTorch FP16 | 8.9 | 899 | 1.6 |
| TensorRT FP16 | 3.2 | 2,500 | 1.2 |
| TensorRT INT8 | 1.9 | 4,211 | 0.8 |

**Key Takeaways:**
- TensorRT delivers **3-5× speedup** for typical CNN models
- **INT8 quantization** provides **8-10× total speedup** with <1% accuracy loss
- **Memory usage drops 50-70%** through layer fusion and precision reduction
- **Larger models benefit more** (transformers see 5-8× improvements)

---

## 2. TensorRT-LLM — Purpose-Built LLM Optimization

### What is TensorRT-LLM?

**TensorRT-LLM** (released 2023) is a purpose-built library for optimizing Large Language Model (LLM) inference. While vanilla TensorRT excels at computer vision and BERT-sized models, LLMs (GPT, LLaMA, Mistral) present unique challenges:

- **Autoregressive generation**: Sequential token generation, not parallel
- **Massive KV caches**: Cache key/value tensors from all previous tokens (memory bottleneck)
- **Variable sequence lengths**: Requests vary from 10 tokens to 10,000+ tokens
- **Multi-GPU inference**: Models too large for single GPU (70B+ parameters)

TensorRT-LLM addresses these with specialized optimizations:

<div class="diagram">
<div class="diagram-title">TensorRT-LLM Architecture & Optimizations</div>
<div class="flow">
<div class="flow-node accent wide">Input: Pre-trained LLM (LLaMA, GPT, Mistral, Falcon)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">
<strong>Attention Optimizations</strong><br/>
├─ FlashAttention-2 (fused attention kernel)<br/>
├─ FlashDecoding (optimized for decode phase)<br/>
├─ Multi-Query Attention (MQA) support<br/>
└─ Grouped-Query Attention (GQA) support
</div>
<div class="flow-arrow green"></div>
<div class="flow-node purple wide">
<strong>KV Cache Management</strong><br/>
├─ Paged Attention (PagedKV, inspired by vLLM)<br/>
├─ Dynamic memory allocation (reuse freed blocks)<br/>
└─ 80% KV cache memory reduction
</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">
<strong>Batching Strategies</strong><br/>
├─ In-Flight Batching (Continuous Batching)<br/>
├─ Variable sequence length support<br/>
└─ Dynamic request scheduling
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan wide">
<strong>Quantization</strong><br/>
├─ SmoothQuant (INT8 weight + activation)<br/>
├─ AWQ (Activation-Aware Weight Quantization)<br/>
├─ GPTQ (post-training quantization)<br/>
└─ FP8 (Hopper H100 native)
</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node pink wide">
<strong>Multi-GPU Parallelism</strong><br/>
├─ Tensor Parallelism (split weight matrices)<br/>
├─ Pipeline Parallelism (split layers across GPUs)<br/>
└─ NVLink/NVSwitch high-bandwidth interconnect
</div>
</div>
</div>

### Why TensorRT-LLM? The LLM Inference Challenge

**Autoregressive Generation Problem:**

Unlike computer vision (single forward pass), LLM text generation is iterative:

```
Input: "The quick brown"
Step 1: Forward pass → predict "fox" → new sequence: "The quick brown fox"
Step 2: Forward pass → predict "jumps" → new sequence: "The quick brown fox jumps"
Step 3: Forward pass → predict "over" → ...
... (repeat until EOS token or max length)
```

Each step requires:
1. **Recomputing attention** over all previous tokens (quadratic complexity)
2. **Storing KV cache** for all previous tokens (memory grows linearly)
3. **Sequential execution** (can't parallelize token generation)

**The Memory Wall:**

For LLaMA-70B generating 2048 tokens with batch size 32:

- **Model weights**: 70B params × 2 bytes (FP16) = **140 GB**
- **KV cache**: 32 batch × 2048 seq × 80 layers × 8192 hidden × 2 (K+V) × 2 bytes = **160 GB**
- **Total**: **300 GB** — exceeds even H100's 80 GB HBM!

TensorRT-LLM's optimizations directly address this.

### FlashAttention & FlashDecoding — Fused Attention Kernels

**Standard Attention Bottleneck:**

Naive attention implementation:
```python
# Pseudocode for standard attention (inefficient!)
Q, K, V = linear_qkv(x)  # [batch, seq, hidden]
scores = Q @ K.T / sqrt(d)  # [batch, seq, seq] — MATERIALIZED IN HBM!
attn_weights = softmax(scores)  # Another HBM write
output = attn_weights @ V  # Read from HBM again
```

Problem: Intermediate `scores` matrix is `O(batch × seq^2)` — **huge memory footprint**, and each HBM read/write is slow (1000× slower than SRAM).

**FlashAttention-2 Solution:**

Fused kernel that:
1. **Tiles the computation** to fit in SRAM (shared memory)
2. **Computes attention incrementally** without materializing full scores matrix
3. **Reduces HBM accesses** by 10-20× (only read Q, K, V once; write output once)

```python
# FlashAttention pseudocode (conceptual)
# Tile Q, K, V into blocks that fit in SRAM
for q_block in tile(Q):
    for k_block, v_block in tile(K, V):
        # All computation in SRAM (fast!)
        scores_block = q_block @ k_block.T / sqrt(d)
        attn_block = softmax(scores_block)
        output_block += attn_block @ v_block
    # Write output_block to HBM (once per Q block)
```

**Speedup**: 2-4× for long sequences (2048+ tokens), 7-15× memory reduction

**FlashDecoding** (decode-phase optimization):

During generation (decode phase), each step processes 1 new token but attends to all previous tokens:
- **Prefill phase**: Process entire prompt (parallel)
- **Decode phase**: Generate tokens one-by-one (sequential)

FlashDecoding optimizes decode by parallelizing across sequence length dimension:
```python
# Standard decode: sequential across sequence
for i in range(seq_length):
    output[i] = attention(Q[new_token], K[:i], V[:i])

# FlashDecoding: parallel across sequence
output = flash_decode_attention(Q[new_token], K_cache, V_cache)
# Internally parallelizes across sequence dimension
```

**Speedup**: 2-3× decode throughput for long contexts (4096+ tokens)

### Paged KV Cache — Efficient Memory Management

**Problem**: KV cache for variable-length sequences wastes memory:

```
Request A: 512 tokens → allocate 2048-token buffer → 75% wasted
Request B: 1024 tokens → allocate 2048-token buffer → 50% wasted
Request C: 128 tokens → allocate 2048-token buffer → 94% wasted
```

Pre-allocating max-length buffers is extremely wasteful.

**Solution**: Paged KV Cache (inspired by vLLM's PagedAttention)

Treat KV cache like virtual memory:
- **Divide cache into fixed-size blocks** (e.g., 16 tokens per block)
- **Allocate blocks on-demand** as sequence grows
- **Free blocks when sequence completes**, reuse for new requests
- **Non-contiguous storage** (blocks can be scattered in memory)

```python
# Pseudocode for paged KV cache
class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=10000):
        self.block_size = block_size
        self.kv_blocks = allocate_gpu_memory(num_blocks, block_size, hidden_dim)
        self.free_blocks = set(range(num_blocks))
        self.request_blocks = {}  # request_id → list of block indices
    
    def allocate(self, request_id, num_tokens):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop()
            allocated.append(block_idx)
        self.request_blocks[request_id] = allocated
        return allocated
    
    def free(self, request_id):
        blocks = self.request_blocks.pop(request_id)
        self.free_blocks.update(blocks)
```

**Benefits**:
- **80% memory reduction** vs pre-allocation
- **Higher batch sizes** (fit more concurrent requests)
- **Eliminates fragmentation** (blocks reused dynamically)

### In-Flight Batching (Continuous Batching)

**Traditional Static Batching Problem:**

Batch requests together, wait for **all** to complete:
```
Request A: 10 tokens generation
Request B: 500 tokens generation
Request C: 50 tokens generation

Static batch: Wait for B (500 tokens) before starting next batch
→ A and C sit idle for 490 and 450 generation steps!
```

**In-Flight Batching Solution:**

Remove completed requests from batch, add new ones **during generation**:

```python
# Continuous batching pseudocode
active_requests = []
request_queue = Queue()

while True:
    # Remove completed requests
    active_requests = [r for r in active_requests if not r.is_complete()]
    
    # Add new requests to fill batch
    while len(active_requests) < max_batch_size and not request_queue.empty():
        active_requests.append(request_queue.get())
    
    if not active_requests:
        break
    
    # Generate one token for all active requests
    generate_step(active_requests)
```

**Benefits**:
- **2-3× throughput improvement** (GPU always fully utilized)
- **Lower latency** for short requests (don't wait for long ones)
- **Better resource utilization** (no idle GPU cycles)

### Multi-GPU Parallelism — Scaling Beyond Single GPU

**Tensor Parallelism** (intra-layer parallelism):

Split weight matrices across GPUs:

```python
# Example: Split linear layer across 4 GPUs
# Original: Y = X @ W  (X: [batch, 4096], W: [4096, 16384])

# Tensor Parallel (column split):
# GPU 0: W0 = W[:, 0:4096]     → Y0 = X @ W0  [batch, 4096]
# GPU 1: W1 = W[:, 4096:8192]  → Y1 = X @ W1  [batch, 4096]
# GPU 2: W2 = W[:, 8192:12288] → Y2 = X @ W2  [batch, 4096]
# GPU 3: W3 = W[:, 12288:]     → Y3 = X @ W3  [batch, 4096]

# All-Reduce: Y = [Y0, Y1, Y2, Y3]  [batch, 16384]
```

**Pipeline Parallelism** (inter-layer parallelism):

Split layers across GPUs:
```
GPU 0: Layers 0-19
GPU 1: Layers 20-39
GPU 2: Layers 40-59
GPU 3: Layers 60-79

Forward pass: GPU0 → GPU1 → GPU2 → GPU3
Micro-batching to keep all GPUs busy
```

**Hybrid Parallelism:**
```python
# LLaMA-70B on 8× H100 GPUs
# Tensor Parallel: 8-way (split each layer 8 ways)
# Pipeline Parallel: 1-way (all layers on all GPUs via TP)
# Total model parallel: 8 GPUs

# Alternative: LLaMA-175B on 16× H100
# Tensor Parallel: 4-way
# Pipeline Parallel: 4-way
# Total: 16 GPUs (4 pipeline stages × 4 tensor parallel groups)
```

### Quantization Integration — SmoothQuant, AWQ, GPTQ, FP8

**SmoothQuant** (INT8 weight + activation quantization):

Problem: Activations have outliers that cause large quantization error.

Solution: Migrate difficulty from activations to weights (which are easier to quantize):

```python
# SmoothQuant transformation
# Original: Y = X @ W
# X has outliers → hard to quantize
# W is smooth → easy to quantize

# Apply per-channel scaling:
s = smoothing_factor(X)  # Detect outlier channels
X_smooth = X / s          # Reduce outliers
W_adjusted = W * s        # Compensate in weights

# Now quantize:
X_int8 = quantize(X_smooth)  # Less error (no outliers)
W_int8 = quantize(W_adjusted)
Y = dequantize(X_int8 @ W_int8)
```

**AWQ (Activation-Aware Weight Quantization):**

Protect important weights (high activation magnitude) from quantization error:

```python
# Identify important weights based on activation statistics
importance = compute_activation_magnitude(weights, calibration_data)

# Quantize less important weights more aggressively
for weight, imp in zip(weights, importance):
    if imp > threshold:
        weight_quant = quantize_4bit_high_precision(weight)
    else:
        weight_quant = quantize_4bit_low_precision(weight)
```

**GPTQ (Post-Training Quantization):**

Layer-wise quantization with error compensation:
1. Quantize weights one layer at a time
2. Compensate quantization error in subsequent layers
3. Minimize reconstruction error via Hessian-based optimization

**FP8 (Hopper H100 Native):**

- **E4M3 format**: 4 exponent bits, 3 mantissa bits (range: 448)
- **E5M2 format**: 5 exponent bits, 2 mantissa bits (range: 57,344)
- **Hardware-accelerated** on Hopper Tensor Cores
- **2× throughput** vs FP16, **minimal accuracy loss** (<0.5% for most LLMs)

### Building & Running a Model — Step-by-Step

**Step 1: Install TensorRT-LLM**

```bash
# Install from source (recommended for latest features)
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
pip install -r requirements.txt
python setup.py develop
```

**Step 2: Convert Model Weights**

```python
# Convert HuggingFace LLaMA-7B to TensorRT-LLM format
from tensorrt_llm.models import LLaMAForCausalLM
from transformers import AutoModelForCausalLM

# Load HuggingFace model
hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Convert to TensorRT-LLM
trt_llm_model = LLaMAForCausalLM.from_hugging_face(
    hf_model,
    dtype="float16",
    mapping={"tp_size": 1, "pp_size": 1}  # Single GPU
)

# Save checkpoint
trt_llm_model.save_checkpoint("llama-7b-trtllm")
```

**Step 3: Build TensorRT-LLM Engine**

```bash
# Build optimized engine with all optimizations
trtllm-build \
    --checkpoint_dir llama-7b-trtllm \
    --output_dir llama-7b-engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --use_paged_kv_cache \
    --use_inflight_batching \
    --workers 4  # Parallel build on 4 GPUs
```

**Step 4: Run Inference**

```python
from tensorrt_llm.runtime import ModelRunner

# Load engine
runner = ModelRunner.from_dir("llama-7b-engine")

# Run inference
prompts = [
    "The capital of France is",
    "Artificial intelligence will",
    "The meaning of life is"
]

outputs = runner.generate(
    prompts,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")
```

**Step 5: Integrate with Triton**

```python
# Deploy TensorRT-LLM engine to Triton Inference Server
# See Section 5 for Triton integration details
```

### TensorRT-LLM Performance — LLaMA-7B Benchmark

**NVIDIA A100 80GB, Batch=8, Input=128 tokens, Output=128 tokens:**

| Implementation | Throughput (tokens/sec) | Latency (ms/token) | Memory (GB) |
|----------------|------------------------|-------------------|-------------|
| PyTorch FP16 (HF) | 580 | 110 | 28 |
| vLLM FP16 | 1,450 | 44 | 18 |
| TensorRT-LLM FP16 | 2,100 | 30 | 16 |
| TensorRT-LLM FP8 (H100) | 4,800 | 13 | 10 |
| TensorRT-LLM INT4 AWQ | 6,200 | 10 | 7 |

**Key Results:**
- **3.6× faster** than PyTorch eager mode
- **45% less memory** than PyTorch
- **FP8 quantization** (H100) delivers **8× speedup** with <1% accuracy loss
- **INT4 AWQ** achieves **10× speedup** for latency-critical applications

---

## 3. ModelOpt — Quantization, Pruning & Distillation

### What is ModelOpt (NVIDIA Model Optimizer)?

**ModelOpt** (formerly Torch-TensorRT and Quantization Toolkit) is NVIDIA's model optimization library for **post-training** and **quantization-aware training**. While TensorRT-LLM focuses on inference optimization, ModelOpt focuses on **model compression** techniques applied *before* inference:

- **Quantization**: Reduce precision (FP32 → FP16 → INT8 → INT4)
- **Pruning**: Remove redundant weights/neurons
- **Distillation**: Train smaller model to mimic larger model
- **Sparsity**: Structured sparsity (2:4) for Ampere+ GPUs

ModelOpt is **framework-integrated** (PyTorch, TensorFlow) and generates optimized models that feed directly into TensorRT/TensorRT-LLM.

<div class="diagram">
<div class="diagram-title">ModelOpt Optimization Techniques</div>
<div class="diagram-grid cols-2">
<div class="diagram-card accent">
<div class="card-icon">🔢</div>
<div class="card-title">Quantization</div>
<div class="card-desc"><strong>PTQ</strong>: Post-training quantization (calibration on small dataset)<br/>
<strong>QAT</strong>: Quantization-aware training (fine-tune with quantization)<br/>
<strong>Methods</strong>: SmoothQuant, AWQ, GPTQ, FP8</div>
</div>
<div class="diagram-card green">
<div class="card-icon">✂️</div>
<div class="card-title">Pruning</div>
<div class="card-desc"><strong>Unstructured</strong>: Remove individual weights (requires sparse kernels)<br/>
<strong>Structured</strong>: Remove entire channels/filters (hardware-friendly)<br/>
<strong>2:4 Sparsity</strong>: 50% sparsity with 2× Ampere+ speedup</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">🎓</div>
<div class="card-title">Knowledge Distillation</div>
<div class="card-desc">Train compact student model to match teacher outputs<br/>
Typical compression: 4× smaller, 90-95% accuracy retention<br/>
BERT-base → TinyBERT (14M params, 3× faster)</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">🧩</div>
<div class="card-title">Neural Architecture Search</div>
<div class="card-desc">Automated model architecture optimization<br/>
Find optimal layer sizes, skip connections, operator choices<br/>
Balances accuracy vs latency/memory constraints</div>
</div>
</div>
</div>

### Quantization — From FP32 to INT8/INT4

**Post-Training Quantization (PTQ):**

No retraining required — quantize pre-trained model with calibration:

```python
import modelopt.torch.quantization as mtq

# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True).cuda()
model.eval()

# Define quantization configuration
quant_cfg = mtq.INT8_DEFAULT_CFG  # INT8 weight + activation

# Calibrate on representative dataset (100-1000 samples)
def calibration_data_loader():
    for i, (images, _) in enumerate(calibration_loader):
        if i >= 100:  # 100 batches for calibration
            break
        yield images.cuda()

# Quantize model
with mtq.set_quantizer_by_cfg(quant_cfg):
    mtq.quantize(model, calibration_data_loader())

# Export to ONNX → TensorRT
torch.onnx.export(model, dummy_input, "resnet50_int8.onnx")
```

**Quantization-Aware Training (QAT):**

Fine-tune model with simulated quantization (higher accuracy than PTQ):

```python
import modelopt.torch.quantization as mtq

# Start from pre-trained FP32 model
model = load_pretrained_model()

# Insert fake quantization nodes
quant_model = mtq.quantize(model, mtq.QAT_DEFAULT_CFG)

# Fine-tune for 5-10 epochs with standard training loop
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = quant_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Export quantized model
mtq.export(quant_model, "model_qat.onnx")
```

**SmoothQuant for LLMs:**

```python
from modelopt.torch.quantization import SmoothQuantConfig, quantize

# Load LLaMA model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Apply SmoothQuant
config = SmoothQuantConfig(
    quant_mode="int8",  # INT8 weights + activations
    alpha=0.5,  # Smoothing factor (balance between weight/activation difficulty)
    calibration_dataset=calibration_data,
    num_calibration_samples=512
)

quantized_model = quantize(model, config)

# Export to TensorRT-LLM
export_to_trtllm(quantized_model, "llama-7b-sq-int8")
```

**AWQ (Activation-Aware Weight Quantization):**

```python
from modelopt.torch.quantization import AWQConfig

config = AWQConfig(
    w_bit=4,  # 4-bit weights
    group_size=128,  # Quantize in groups of 128 (finer granularity)
    zero_point=True,  # Use zero-point for asymmetric quantization
    calibration_dataset=calibration_data
)

awq_model = quantize(model, config)
# 4× memory reduction, 2-3× speedup, ~1% accuracy loss
```

### Pruning — Removing Redundant Parameters

**Magnitude-Based Pruning:**

Remove weights with smallest absolute values:

```python
from modelopt.torch.pruning import prune_model, MagnitudePruner

# Prune 50% of weights globally
pruner = MagnitudePruner(model, sparsity=0.5)
pruned_model = pruner.prune()

# Fine-tune to recover accuracy
for epoch in range(5):
    train_one_epoch(pruned_model, train_loader, optimizer)

# Result: 50% fewer parameters, 1.5-2× speedup (with sparse kernels)
```

**Structured Pruning (Channel Pruning):**

Remove entire channels/filters (hardware-friendly, no special kernels needed):

```python
from modelopt.torch.pruning import ChannelPruner

# Prune 30% of channels per layer
pruner = ChannelPruner(
    model,
    sparsity_per_layer=0.3,
    importance_metric="l1_norm"  # L1 norm of channel weights
)

pruned_model = pruner.prune()
# Result: 30% fewer FLOPs, 30% less memory, runs on standard kernels
```

**2:4 Structured Sparsity (Ampere+):**

NVIDIA Ampere+ GPUs have **hardware acceleration** for 2:4 sparsity (2 zeros per 4 values):

```python
from modelopt.torch.sparsity import apply_2_4_sparsity

# Apply 2:4 sparsity pattern
sparse_model = apply_2_4_sparsity(
    model,
    calibration_data=calibration_loader
)

# Result: 50% sparsity, 2× Tensor Core throughput on Ampere+, <1% accuracy loss
```

Visualization of 2:4 pattern:
```
Original weights: [0.8, 0.3, -0.6, 0.1, 0.9, -0.4, 0.2, -0.7]
2:4 sparse:       [0.8, 0.0, -0.6, 0.0, 0.9, -0.4, 0.0, -0.7]
                   └───────┬─────────┘ └───────┬─────────┘
                     2 non-zero per 4    2 non-zero per 4
```

### Knowledge Distillation — Compress via Teacher-Student

Train small "student" model to mimic large "teacher" model:

```python
from modelopt.torch.distillation import Distiller

# Teacher: Large pre-trained model (BERT-large, 340M params)
teacher = BertLargeModel.from_pretrained("bert-large-uncased")
teacher.eval()

# Student: Smaller architecture (BERT-small, 60M params)
student = BertSmallModel(num_layers=6, hidden_size=512)

# Distillation training
distiller = Distiller(
    teacher=teacher,
    student=student,
    temperature=4.0,  # Soften teacher predictions
    alpha=0.5  # Balance distillation loss vs task loss
)

for epoch in range(20):
    for batch in train_loader:
        # Compute teacher outputs (no gradient)
        with torch.no_grad():
            teacher_logits = teacher(batch)
        
        # Student forward pass
        student_logits = student(batch)
        
        # Combined loss: KL divergence + task loss
        distill_loss = distiller.loss(student_logits, teacher_logits, batch.labels)
        distill_loss.backward()
        optimizer.step()

# Result: 5× smaller model, 3× faster, 95% of teacher accuracy
```

### ModelOpt → TensorRT Pipeline

Complete workflow:

```python
# Step 1: Quantize with ModelOpt
from modelopt.torch.quantization import quantize, INT8_DEFAULT_CFG

model = load_model()
quantized_model = quantize(model, INT8_DEFAULT_CFG, calibration_data)

# Step 2: Apply 2:4 sparsity
from modelopt.torch.sparsity import apply_2_4_sparsity
sparse_model = apply_2_4_sparsity(quantized_model, calibration_data)

# Step 3: Export to ONNX
torch.onnx.export(sparse_model, dummy_input, "model_optimized.onnx")

# Step 4: Build TensorRT engine
import tensorrt as trt
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(...)
parser = trt.OnnxParser(network, trt.Logger())
parser.parse_from_file("model_optimized.onnx")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

engine = builder.build_engine(network, config)

# Result: INT8 + 2:4 sparse = 8× speedup, 4× memory reduction
```

---

## 4. AITune — Automated Configuration Optimization

### What is AITune?

**AITune** is NVIDIA's automated tuning framework for optimizing inference configurations. While TensorRT optimizes *kernels*, AITune optimizes *deployment parameters*:

- **Batch size tuning**: Find optimal batch size for latency/throughput trade-off
- **Memory allocation**: Optimize workspace sizes, buffer allocations
- **Concurrency settings**: Tune thread counts, stream counts
- **Dynamic batching**: Configure Triton's dynamic batching policies

AITune uses **Bayesian optimization** to explore configuration space efficiently (avoiding exhaustive grid search).

### Why AITune?

Manual tuning is tedious and suboptimal:

```python
# Manual approach: Try different batch sizes
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    latency, throughput = benchmark(model, batch_size)
    print(f"Batch {batch_size}: {latency} ms, {throughput} QPS")

# Problems:
# 1. Time-consuming (need to test many configs)
# 2. Ignores interactions (batch size × stream count × memory allocation)
# 3. May miss optimal configuration (discrete search)
```

AITune automates this with **multi-objective optimization** (minimize latency AND maximize throughput):

```python
from aitune import Tuner

tuner = Tuner(
    model_path="resnet50.plan",
    objectives=["latency", "throughput"],
    constraints={"memory": "12GB", "latency_p99": "10ms"}
)

# Runs Bayesian optimization (30-100 trials)
best_config = tuner.tune(max_trials=50)

print(f"Optimal batch size: {best_config['batch_size']}")
print(f"Optimal stream count: {best_config['streams']}")
print(f"Expected latency: {best_config['latency']} ms")
print(f"Expected throughput: {best_config['throughput']} QPS")
```

### Batch Size Optimization — Latency vs Throughput

**The Trade-off:**

- **Small batches** (1-4): Low latency, poor GPU utilization → good for real-time apps
- **Large batches** (32-128): High latency, excellent throughput → good for offline processing

AITune finds the **Pareto frontier**:

<div class="diagram">
<div class="diagram-title">Batch Size Pareto Frontier</div>
<div class="flow">
<div class="flow-node accent">Batch=1: Latency=5ms, Throughput=200 QPS, GPU Util=15%</div>
<div class="flow-arrow"></div>
<div class="flow-node green">Batch=8: Latency=12ms, Throughput=667 QPS, GPU Util=45%</div>
<div class="flow-arrow"></div>
<div class="flow-node purple">Batch=32: Latency=35ms, Throughput=914 QPS, GPU Util=75%</div>
<div class="flow-arrow"></div>
<div class="flow-node orange">Batch=128: Latency=110ms, Throughput=1164 QPS, GPU Util=92%</div>
</div>
</div>

**AITune Selection Strategy:**

```python
# Optimize for latency-constrained scenario (SLA: p99 < 15ms)
config_low_latency = tuner.tune(
    objectives=["throughput"],  # Maximize throughput
    constraints={"latency_p99": "15ms"}  # Subject to latency constraint
)
# Result: batch=8, 667 QPS

# Optimize for throughput-maximization (batch processing)
config_high_throughput = tuner.tune(
    objectives=["throughput"],  # Maximize throughput
    constraints={"memory": "16GB"}  # Subject to memory constraint
)
# Result: batch=128, 1164 QPS
```

### Memory Management Tuning

**TensorRT Workspace Optimization:**

TensorRT uses "workspace" for temporary buffers during kernel execution. Larger workspace → more kernel options → potentially faster kernels, but consumes memory.

```python
# AITune finds optimal workspace size
config = tuner.tune_workspace(
    model="model.plan",
    workspace_range=[256_000_000, 8_000_000_000],  # 256 MB - 8 GB
    metric="throughput"
)

print(f"Optimal workspace: {config['workspace_size'] / 1e9} GB")
# Typical result: 2-4 GB provides best throughput/memory balance
```

**KV Cache Memory (LLMs):**

For TensorRT-LLM, tune KV cache block sizes:

```python
config = tuner.tune_kv_cache(
    model="llama-7b-engine",
    max_batch_size=32,
    max_sequence_length=2048,
    gpu_memory="40GB"
)

print(f"KV cache block size: {config['kv_cache_block_size']}")
print(f"Max concurrent requests: {config['max_concurrent_requests']}")
# Balances memory usage vs request capacity
```

### Integration with Triton Inference Server

AITune can optimize Triton server configurations:

```python
from aitune import TritonTuner

tuner = TritonTuner(
    model_repository="/models",
    model_name="resnet50",
    backend="tensorrt"
)

# Tune dynamic batching parameters
config = tuner.tune_dynamic_batching(
    max_queue_delay_microseconds=100_000,  # Search range
    objectives=["throughput"],
    constraints={"latency_p99": "20ms"}
)

# Generates optimized config.pbtxt:
# dynamic_batching {
#   preferred_batch_size: [8, 16]
#   max_queue_delay_microseconds: 5000
# }
```

---

## 5. Triton Inference Server — Production Model Serving

### What is Triton Inference Server?

**Triton Inference Server** (formerly TensorRT Inference Server) is NVIDIA's open-source inference serving platform. Think of it as "Nginx for AI models" — a production-grade HTTP/gRPC server optimized for deploying ML models at scale.

<div class="diagram">
<div class="diagram-title">Triton Inference Server Architecture</div>
<div class="flow">
<div class="flow-node accent wide">Client Applications (HTTP/gRPC/C API)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">
<strong>Triton Server Core</strong><br/>
├─ Request Router & Load Balancer<br/>
├─ Dynamic Batcher<br/>
├─ Model Scheduler<br/>
└─ Metrics & Monitoring (Prometheus)
</div>
<div class="flow-arrow green"></div>
<div class="flow-node purple wide">
<strong>Backend Engines</strong><br/>
├─ TensorRT (optimized inference)<br/>
├─ PyTorch (Python models)<br/>
├─ TensorFlow (TF SavedModel)<br/>
├─ ONNX Runtime<br/>
├─ OpenVINO (Intel)<br/>
└─ Python Backend (custom logic)
</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">GPU / CPU Execution</div>
</div>
</div>

### Why Triton?

**Problem**: Deploying ML models to production involves challenges beyond inference optimization:

- **Dynamic batching**: Batch variable-size requests to maximize throughput
- **Multi-model serving**: Run multiple models on single server, share GPU efficiently
- **A/B testing**: Route traffic between model versions
- **Model ensembles**: Chain models (preprocessing → inference → postprocessing)
- **Monitoring**: Latency metrics, throughput, GPU utilization
- **Autoscaling**: Scale instances based on load

Triton handles all of this out-of-the-box.

**Key Features:**

<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<div class="card-icon">📦</div>
<div class="card-title">Multi-Framework</div>
<div class="card-desc">TensorRT, PyTorch, TensorFlow, ONNX, TensorRT-LLM — all in one server</div>
</div>
<div class="diagram-card green">
<div class="card-icon">⚡</div>
<div class="card-title">Dynamic Batching</div>
<div class="card-desc">Automatically batch requests for optimal GPU utilization</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">🔄</div>
<div class="card-title">Model Ensemble</div>
<div class="card-desc">Chain models into pipelines (data prep → inference → post-process)</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">🎯</div>
<div class="card-title">Model Versioning</div>
<div class="card-desc">A/B testing, canary deployments, gradual rollouts</div>
</div>
<div class="diagram-card cyan">
<div class="card-icon">📊</div>
<div class="card-title">Metrics & Monitoring</div>
<div class="card-desc">Prometheus metrics, DCGM GPU stats, custom metrics</div>
</div>
<div class="diagram-card teal">
<div class="card-icon">🌐</div>
<div class="card-title">Kubernetes Native</div>
<div class="card-desc">Helm charts, autoscaling, GPU scheduling integration</div>
</div>
</div>

### Setting Up Triton — Model Repository

Triton uses a **model repository** (file system or cloud storage) to load models:

```
model_repository/
├── resnet50/
│   ├── config.pbtxt          # Model configuration
│   └── 1/                    # Version 1
│       └── model.plan        # TensorRT engine
├── bert_tokenizer/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # Python backend
└── llama_7b/
    ├── config.pbtxt
    └── 1/
        ├── config.json
        ├── model.engine
        └── tokenizer.json
```

**Example config.pbtxt** (ResNet-50 TensorRT):

```protobuf
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP16
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP16
    dims: [1000]
  }
]

# Dynamic batching configuration
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
}

# Instance groups (how many model instances to run)
instance_group [
  {
    count: 2  # Run 2 concurrent instances
    kind: KIND_GPU
    gpus: [0]  # On GPU 0
  }
]
```

### Dynamic Batching — Maximizing Throughput

**How It Works:**

Triton collects incoming requests and batches them before sending to model:

```
Time 0ms:   Request A arrives (batch=1)
Time 1ms:   Request B arrives (batch=2)
Time 2ms:   Request C arrives (batch=3)
Time 5ms:   Max delay reached → execute batch of 3
Time 10ms:  Requests D,E,F arrive → batch of 3
Time 15ms:  Execute batch of 3
```

**Configuration:**

```protobuf
dynamic_batching {
  # Preferred batch sizes (Triton tries to form these)
  preferred_batch_size: [8, 16, 32]
  
  # Max time to wait for full batch (microseconds)
  max_queue_delay_microseconds: 5000  # 5ms
  
  # Preserve ordering of responses
  preserve_ordering: false
  
  # Priority levels for different requests
  priority_levels: 3
  default_priority_level: 2
}
```

**Benefits:**
- **2-5× throughput** vs no batching (for batch-friendly models)
- **Automatic adaptation** to request rate (small batches when idle, large when busy)
- **Latency control** via `max_queue_delay`

### Model Ensemble — Chaining Models

Combine multiple models into a pipeline:

```
Ensemble: "image_classification"
  ↓
Step 1: "preprocessing" (Python backend)
  - Resize image
  - Normalize pixels
  ↓
Step 2: "resnet50" (TensorRT backend)
  - CNN inference
  ↓
Step 3: "postprocessing" (Python backend)
  - Softmax
  - Top-5 classes
  ↓
Output: JSON with top-5 predictions
```

**Ensemble config.pbtxt:**

```protobuf
name: "image_classification_ensemble"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "raw_image"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3]  # Variable size
  }
]

output [
  {
    name: "predictions"
    data_type: TYPE_STRING
    dims: [1]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: 1
      input_map { key: "raw_image" value: "raw_image" }
      output_map { key: "preprocessed" value: "preprocessed_image" }
    },
    {
      model_name: "resnet50"
      model_version: 1
      input_map { key: "input" value: "preprocessed_image" }
      output_map { key: "output" value: "logits" }
    },
    {
      model_name: "postprocessing"
      model_version: 1
      input_map { key: "logits" value: "logits" }
      output_map { key: "predictions" value: "predictions" }
    }
  ]
}
```

### Client Usage — HTTP & gRPC

**Python Client (HTTP):**

```python
import tritonclient.http as httpclient
import numpy as np

# Connect to Triton server
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
image = preprocess_image("cat.jpg")  # [3, 224, 224] FP16
input_tensor = httpclient.InferInput("input", image.shape, "FP16")
input_tensor.set_data_from_numpy(image)

# Prepare output
output = httpclient.InferRequestedOutput("output")

# Inference request
response = client.infer(
    model_name="resnet50",
    inputs=[input_tensor],
    outputs=[output]
)

# Parse results
logits = response.as_numpy("output")
predicted_class = np.argmax(logits)
print(f"Predicted class: {predicted_class}")
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/v2/models/resnet50/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3, 224, 224],
        "datatype": "FP16",
        "data": [...]
      }
    ]
  }'
```

### Metrics & Monitoring

Triton exposes **Prometheus metrics** at `/metrics`:

```
# Request count per model
nv_inference_request_success{model="resnet50",version="1"} 15234

# Inference latency (P50, P90, P99)
nv_inference_request_duration_us{model="resnet50",quantile="0.99"} 8500

# GPU utilization
nv_gpu_utilization{gpu="0"} 0.85

# Queue time (dynamic batching)
nv_inference_queue_duration_us{model="resnet50",quantile="0.99"} 3200

# GPU memory usage
nv_gpu_memory_used_bytes{gpu="0"} 12884901888
```

**Grafana Dashboard Example:**

```yaml
# Monitor throughput and latency
Queries:
  - rate(nv_inference_request_success[1m])  # QPS
  - nv_inference_request_duration_us{quantile="0.99"}  # P99 latency
  - nv_gpu_utilization  # GPU %
  - nv_inference_queue_duration_us{quantile="0.99"}  # Queue time
```

### Kubernetes Deployment

**Helm Chart:**

```yaml
# values.yaml
replicaCount: 3  # 3 Triton instances

image:
  repository: nvcr.io/nvidia/tritonserver
  tag: 24.03-py3

resources:
  limits:
    nvidia.com/gpu: 1  # 1 GPU per pod

modelRepositoryPath: "s3://my-bucket/models"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

Deploy:
```bash
helm install triton-server nvidia/triton-inference-server -f values.yaml
```

---

## 6. The Complete Optimization Pipeline

### End-to-End Workflow — Training to Production

<div class="diagram">
<div class="diagram-title">Complete NVIDIA Optimization Pipeline</div>
<div class="flow">
<div class="flow-node accent wide">
<strong>Step 1: Training</strong><br/>
PyTorch/TensorFlow model training<br/>
LLaMA-7B trained on 1T tokens
</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">
<strong>Step 2: ModelOpt Optimization</strong><br/>
├─ SmoothQuant INT8 quantization<br/>
├─ 2:4 structured sparsity<br/>
└─ Fine-tune 1 epoch to recover accuracy
</div>
<div class="flow-arrow green"></div>
<div class="flow-node purple wide">
<strong>Step 3: TensorRT-LLM Build</strong><br/>
├─ Convert to TensorRT-LLM format<br/>
├─ Enable FlashAttention + PagedKV<br/>
├─ Build optimized engine (.engine file)<br/>
└─ Auto-tune kernels for target GPU (H100)
</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">
<strong>Step 4: AITune Configuration</strong><br/>
├─ Optimize batch size (sweet spot: 16)<br/>
├─ Tune KV cache block size<br/>
└─ Configure in-flight batching parameters
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan wide">
<strong>Step 5: Triton Deployment</strong><br/>
├─ Deploy to Triton Inference Server<br/>
├─ Configure dynamic batching<br/>
├─ Set up monitoring & metrics<br/>
└─ Kubernetes autoscaling
</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node pink wide">
<strong>Production Result</strong><br/>
8× faster than PyTorch | 4× less memory | 10× lower cost
</div>
</div>
</div>

### Case Study — Optimizing LLaMA-7B

**Initial State:**
- **Model**: LLaMA-7B (7 billion parameters)
- **Framework**: PyTorch + HuggingFace Transformers
- **Hardware**: NVIDIA A100 80GB
- **Workload**: Text generation, avg 128 input tokens, 128 output tokens
- **Performance**: 45 tokens/sec, 18 GB memory

**Step 1: Apply SmoothQuant INT8**

```bash
# Quantize with ModelOpt
python modelopt_quantize.py \
  --model meta-llama/Llama-2-7b-hf \
  --method smoothquant \
  --output llama-7b-sq-int8

# Result: 14 GB → 8 GB memory (weights: FP16 → INT8)
```

**Step 2: Build TensorRT-LLM Engine**

```bash
# Convert to TensorRT-LLM
python convert_checkpoint.py \
  --model_dir llama-7b-sq-int8 \
  --output_dir llama-7b-trtllm-ckpt \
  --dtype float16 \
  --use_smooth_quant

# Build engine
trtllm-build \
  --checkpoint_dir llama-7b-trtllm-ckpt \
  --output_dir llama-7b-engine \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --max_batch_size 16 \
  --max_input_len 512 \
  --max_output_len 512 \
  --use_paged_kv_cache \
  --use_inflight_batching \
  --paged_kv_cache_max_tokens 10000

# Result: FlashAttention + PagedKV enabled
```

**Step 3: AITune Optimization**

```python
from aitune import TRTLLMTuner

tuner = TRTLLMTuner(engine_dir="llama-7b-engine")

config = tuner.tune(
    objectives=["throughput"],
    constraints={"latency_p99": "100ms", "memory": "20GB"},
    max_trials=30
)

# Optimal config found:
# - Batch size: 16
# - KV cache blocks: 2048
# - Max concurrent requests: 24
```

**Step 4: Deploy to Triton**

```protobuf
# model_repository/llama_7b/config.pbtxt
name: "llama_7b"
backend: "tensorrtllm"
max_batch_size: 16

model_transaction_policy {
  decoupled: true  # For streaming responses
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

parameters {
  key: "engine_dir"
  value { string_value: "/models/llama-7b-engine" }
}

parameters {
  key: "max_tokens_in_paged_kv_cache"
  value { string_value: "10000" }
}
```

**Results:**

| Metric | PyTorch Baseline | Optimized (TRT-LLM + Triton) | Improvement |
|--------|------------------|------------------------------|-------------|
| **Throughput (tokens/sec)** | 45 | 380 | **8.4× faster** |
| **Latency P99 (ms/token)** | 148 | 19 | **7.8× faster** |
| **Memory (GB)** | 18 | 6.5 | **2.8× reduction** |
| **Cost per 1M tokens** | $12.50 | $1.49 | **8.4× cheaper** |

**Key Optimizations Applied:**
1. ✅ SmoothQuant INT8 (2× speedup)
2. ✅ FlashAttention-2 (1.8× speedup)
3. ✅ Paged KV cache (2.7× memory reduction)
4. ✅ In-flight batching (1.5× throughput boost)
5. ✅ Kernel auto-tuning (1.3× speedup)

**Total multiplicative effect**: 2 × 1.8 × 1.5 × 1.3 ≈ **7× speedup**

---

## 7. Performance Benchmarks

### Computer Vision Models

**NVIDIA A100 80GB, FP16 Precision, Batch=32**

| Model | PyTorch Eager | TorchScript | torch.compile | TensorRT FP16 | TensorRT INT8 |
|-------|---------------|-------------|---------------|---------------|---------------|
| **ResNet-50** | 1,127 img/s | 1,667 img/s | 2,270 img/s | **4,706 img/s** | **10,000 img/s** |
| **EfficientNet-B0** | 890 img/s | 1,234 img/s | 1,789 img/s | **3,456 img/s** | **7,890 img/s** |
| **Vision Transformer (ViT-B/16)** | 456 img/s | 678 img/s | 892 img/s | **2,134 img/s** | **3,678 img/s** |
| **YOLOv8-L (detection)** | 234 img/s | 345 img/s | 512 img/s | **1,234 img/s** | **2,456 img/s** |
| **Mask R-CNN (segmentation)** | 89 img/s | 123 img/s | 167 img/s | **456 img/s** | **789 img/s** |

### Language Models (Transformers)

**NVIDIA H100 80GB, Batch=8, Sequence Length=128**

| Model | HuggingFace FP16 | TensorRT FP16 | TensorRT-LLM FP16 | TensorRT-LLM FP8 | TensorRT-LLM INT4 |
|-------|------------------|---------------|-------------------|------------------|-------------------|
| **BERT-Base** | 899 seq/s | 2,500 seq/s | 2,800 seq/s | **5,200 seq/s** | **7,100 seq/s** |
| **BERT-Large** | 234 seq/s | 712 seq/s | 890 seq/s | **1,890 seq/s** | **2,670 seq/s** |
| **GPT-2 (1.5B)** | 124 tok/s | 456 tok/s | 890 tok/s | **2,100 tok/s** | **3,400 tok/s** |
| **LLaMA-7B** | 45 tok/s | 189 tok/s | 380 tok/s | **920 tok/s** | **1,450 tok/s** |
| **LLaMA-13B** | 28 tok/s | 98 tok/s | 210 tok/s | **540 tok/s** | **890 tok/s** |
| **LLaMA-70B (2×H100)** | 7 tok/s | 24 tok/s | 67 tok/s | **178 tok/s** | **289 tok/s** |

### Optimization Impact Breakdown

**LLaMA-7B on H100, Cumulative Speedup Analysis:**

| Optimization Step | Throughput (tokens/sec) | Speedup vs Previous | Cumulative Speedup |
|-------------------|------------------------|---------------------|-------------------|
| **Baseline (PyTorch)** | 45 | 1.0× | 1.0× |
| + TensorRT-LLM (FP16) | 189 | 4.2× | 4.2× |
| + FlashAttention-2 | 340 | 1.8× | 7.6× |
| + Paged KV Cache | 380 | 1.1× | 8.4× |
| + In-Flight Batching | 570 | 1.5× | 12.7× |
| + FP8 Quantization | 920 | 1.6× | **20.4×** |
| + INT4 AWQ | 1,450 | 1.6× | **32.2×** |

### Memory Efficiency

**Model Memory Footprint Comparison (Single GPU)**

| Model | PyTorch FP32 | PyTorch FP16 | TensorRT FP16 | TensorRT INT8 | TRT-LLM FP8 | TRT-LLM INT4 |
|-------|--------------|--------------|---------------|---------------|-------------|--------------|
| **ResNet-50** | 392 MB | 196 MB | 128 MB | **64 MB** | N/A | N/A |
| **BERT-Base** | 1.8 GB | 900 MB | 720 MB | **450 MB** | **360 MB** | **280 MB** |
| **LLaMA-7B** | 56 GB | 28 GB | 18 GB | **14 GB** | **10 GB** | **7 GB** |
| **LLaMA-70B** | 560 GB | 280 GB | 180 GB | **140 GB** | **100 GB** | **70 GB** |

**Key Insight**: INT8 quantization provides **2× memory reduction**, FP8 provides **2.8×**, and INT4 provides **4×** — critical for fitting large models on consumer GPUs.

### Cost Analysis — AWS p5.48xlarge (8× H100)

**LLaMA-70B Inference, 1 Million Tokens Generated:**

| Configuration | Throughput (tok/s) | Time (hrs) | Cost @ $98.32/hr | Cost per 1M Tokens |
|---------------|-------------------|-----------|------------------|-------------------|
| PyTorch FP16 (8 GPUs) | 56 | 4.96 | $487.67 | **$487.67** |
| TensorRT-LLM FP16 | 536 | 0.52 | $51.13 | **$51.13** |
| TensorRT-LLM FP8 | 1,424 | 0.19 | $18.68 | **$18.68** |

**Savings**: TensorRT-LLM FP8 is **26× cheaper** than PyTorch baseline!

---

## Summary — The Optimization Stack

NVIDIA's optimization stack transforms raw model performance:

1. **TensorRT**: Layer fusion, precision calibration, kernel auto-tuning → **3-5× speedup** for CNNs
2. **TensorRT-LLM**: FlashAttention, Paged KV, In-Flight Batching → **8-20× speedup** for LLMs
3. **ModelOpt**: Quantization (SmoothQuant, AWQ), pruning, sparsity → **2-4× compression**
4. **AITune**: Automated batch size, memory tuning → **1.5-2× additional gains**
5. **Triton**: Dynamic batching, model ensemble, production serving → **2-3× throughput**

**Combined Impact**: **10-50× end-to-end improvement** from training framework to production deployment.

For organizations deploying AI at scale, mastering this stack is not optional — it's the difference between feasible and infeasible, profitable and unprofitable, fast and unusable.

---

**Next: [Chapter 6 — NVIDIA Models →](./06_nvidia_models.md)**

---

*Last updated: April 2026*
