---
title: "Chapter 7 — Parallelism on GPUs"
---

[← Back to Table of Contents](./README.md)

# Chapter 7: Parallelism on GPUs — Every Technique Explained

## 1. Why Parallelism?

Modern deep learning models have grown far beyond the capacity of a single GPU. Understanding the memory requirements and the motivation for parallelism is essential.

### The Single-GPU Memory Limit Problem

A GPU's memory is finite. Even NVIDIA's flagship H100 SXM5 offers "only" 80GB of HBM3 memory. Meanwhile, state-of-the-art language models demand orders of magnitude more:

- **GPT-3 (175B parameters)**: ~350GB in FP16 (2 bytes per parameter)
- **LLaMA-2 70B**: ~140GB in FP16
- **GPT-4 (rumored 1.8T MoE)**: Multiple terabytes

This creates an immediate problem: **the model simply does not fit on a single GPU**.

### Memory Breakdown

Training a neural network requires storing more than just the model parameters:

<div class="diagram">
<div class="diagram-title">Training Memory Components</div>
<div class="diagram-grid cols-4">
<div class="diagram-card accent">
<div class="card-icon">⚖️</div>
<div class="card-title">Parameters</div>
<div class="card-desc">Model weights Ψ — Base memory requirement</div>
</div>
<div class="diagram-card green">
<div class="card-icon">∇</div>
<div class="card-title">Gradients</div>
<div class="card-desc">Same size as parameters: Ψ</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">📈</div>
<div class="card-title">Optimizer States</div>
<div class="card-desc">Adam: 2Ψ (momentum + variance)</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">🔥</div>
<div class="card-title">Activations</div>
<div class="card-desc">Batch size × layers × sequence length</div>
</div>
</div>
</div>

**Total memory for mixed-precision training (FP16 weights, FP32 optimizer)**:

$$
M_{\text{total}} = \underbrace{2\Psi}_{\text{FP16 params}} + \underbrace{2\Psi}_{\text{FP16 grads}} + \underbrace{4\Psi}_{\text{FP32 master}} + \underbrace{8\Psi}_{\text{Adam states}} + \underbrace{M_{\text{act}}}_{\text{activations}}
$$

$$
M_{\text{total}} = 16\Psi + M_{\text{act}}
$$

For a 175B parameter model:
- **Parameters & optimizer states alone**: 16 × 175B = 2.8TB
- **Activation memory** (batch=1024, seq=2048): ~500GB additional

This is **42× larger than a single H100's 80GB**.

**Solution**: Distribute the workload across multiple GPUs using various parallelism strategies.

---

## 2. Data Parallelism (DP)

The simplest and most commonly used parallelism technique.

### Concept

**Replicate the entire model** on each GPU. Split the training batch across GPUs. Each GPU:
1. Processes its portion of the batch independently
2. Computes gradients
3. Synchronizes gradients across all GPUs via **AllReduce**
4. Updates its local model copy (which remains identical across GPUs)

<div class="diagram">
<div class="diagram-title">Data Parallelism Architecture</div>
<div class="flow">
<div class="flow-node accent wide">Input Batch (size B)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">Split into N chunks (B/N each)</div>
<div class="flow-arrow green"></div>
<div class="flow-h">
<div class="flow-node purple">GPU 0<br/>Model Replica<br/>Batch 0</div>
<div class="flow-node purple">GPU 1<br/>Model Replica<br/>Batch 1</div>
<div class="flow-node purple">GPU 2<br/>Model Replica<br/>Batch 2</div>
<div class="flow-node purple">GPU N-1<br/>Model Replica<br/>Batch N-1</div>
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node orange wide">Forward Pass (Independent)</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan wide">Backward Pass → Gradients</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node accent wide">AllReduce Gradients (Ring or Tree)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">Optimizer Step (Synchronized Models)</div>
</div>
</div>

### PyTorch DistributedDataParallel (DDP)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl', init_method='env://')

# Wrap model
model = YourModel().cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass (independent per GPU)
    output = model(batch)
    loss = criterion(output, labels)
    
    # Backward pass
    loss.backward()  # DDP automatically handles gradient AllReduce
    
    # Optimizer step
    optimizer.step()
```

### Communication Cost Analysis

**AllReduce operation**: Each GPU sends and receives gradients for all Ψ parameters.

For a ring-based AllReduce with $N$ GPUs:
- **Data transferred per GPU**: $2 \cdot \frac{N-1}{N} \cdot \Psi$ elements
- **Time**: $T_{\text{comm}} = \frac{2(N-1)\Psi}{N \cdot B}$, where $B$ is the interconnect bandwidth

For large $N$: $T_{\text{comm}} \approx \frac{2\Psi}{B}$

### Scaling Efficiency

Ideal speedup with $N$ GPUs is $N$×. Actual speedup:

$$
S(N) = \frac{T_1}{T_N} = \frac{T_{\text{compute}} + T_{\text{comm}}(1)}{T_{\text{compute}}/N + T_{\text{comm}}(N)}
$$

**Efficiency**:

$$
E(N) = \frac{S(N)}{N} = \frac{1}{1 + \frac{N \cdot T_{\text{comm}}}{T_{\text{compute}}}}
$$

**Key insight**: Efficiency degrades as communication time becomes significant relative to compute time.

**Scaling strategies**:
- Increase batch size proportionally with $N$ (maintains compute-to-communication ratio)
- Use faster interconnects (NVLink, InfiniBand)
- Gradient accumulation (reduce communication frequency)

### Limitations

✅ **Advantages**:
- Simple to implement
- Perfect for small-to-medium models that fit in GPU memory
- Linear scaling with fast interconnects

❌ **Limitations**:
- Each GPU must hold the **entire model** → Doesn't solve memory problem for large models
- Gradient synchronization overhead grows with model size
- No parallelism within a single sample's forward/backward pass

---

## 3. Model Parallelism

When a model is too large to fit on a single GPU, split the model itself across devices.

### Simple Vertical Partitioning

Divide the model's layers across GPUs sequentially:

<div class="diagram">
<div class="diagram-title">Naive Model Parallelism</div>
<div class="flow">
<div class="flow-node accent">Input</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">GPU 0: Layers 0-3</div>
<div class="flow-arrow green"></div>
<div class="flow-node purple wide">Transfer to GPU 1</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">GPU 1: Layers 4-7</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan wide">Transfer to GPU 2</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node pink wide">GPU 2: Layers 8-11</div>
<div class="flow-arrow pink"></div>
<div class="flow-node accent">Output</div>
</div>
</div>

```python
class NaiveModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_0_3 = nn.Sequential(...).to('cuda:0')
        self.layers_4_7 = nn.Sequential(...).to('cuda:1')
        self.layers_8_11 = nn.Sequential(...).to('cuda:2')
    
    def forward(self, x):
        x = self.layers_0_3(x)
        x = x.to('cuda:1')
        x = self.layers_4_7(x)
        x = x.to('cuda:2')
        x = self.layers_8_11(x)
        return x
```

### The Idle GPU Problem (Pipeline Bubbles)

**Critical flaw**: At any given time, only **one GPU is active**. Others sit idle waiting for data.

For a model split across 4 GPUs:
- GPU 0 processes input → GPUs 1,2,3 idle
- GPU 1 processes → GPUs 0,2,3 idle
- GPU 2 processes → GPUs 0,1,3 idle
- GPU 3 processes → GPUs 0,1,2 idle

**GPU utilization**: ~25% (1/4 GPUs active at a time)

This is unacceptable. **Solution**: Pipeline Parallelism.

---

## 4. Pipeline Parallelism (PP)

Split the model into stages AND the batch into micro-batches, creating a pipeline.

### GPipe: Synchronous Pipeline

**Key idea**: While GPU 1 processes micro-batch 1, GPU 0 can process micro-batch 2.

<div class="diagram">
<div class="diagram-title">GPipe Pipeline Schedule (4 stages, 8 micro-batches)</div>
<div class="flow">
<div class="flow-node accent wide">Input: Batch of 8 samples → 8 micro-batches</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide" style="font-family: monospace; white-space: pre; text-align: left;">
Time →
GPU 0: F0 F1 F2 F3 F4 F5 F6 F7 __ __ __ __ __ __ __ __ __ __ __ __ __ B0 B1 B2 B3 B4 B5 B6 B7
GPU 1: __ F0 F1 F2 F3 F4 F5 F6 F7 __ __ __ __ __ __ __ __ __ __ __ B0 B1 B2 B3 B4 B5 B6 B7 __
GPU 2: __ __ F0 F1 F2 F3 F4 F5 F6 F7 __ __ __ __ __ __ __ __ __ B0 B1 B2 B3 B4 B5 B6 B7 __ __
GPU 3: __ __ __ F0 F1 F2 F3 F4 F5 F6 F7 __ __ __ __ __ __ __ B0 B1 B2 B3 B4 B5 B6 B7 __ __ __

(F = Forward, B = Backward, __ = Bubble/Idle)
</div>
</div>
</div>

**Bubble ratio** (fraction of time spent idle):

$$
\text{Bubble fraction} = \frac{p - 1}{m + p - 1} \approx \frac{p - 1}{m}
$$

where:
- $p$ = number of pipeline stages (GPUs)
- $m$ = number of micro-batches

**Example**: 4 stages, 8 micro-batches → Bubble = 3/8 = 37.5% idle time

**Reducing bubbles**: Increase $m$ (more micro-batches). But this increases activation memory!

### PipeDream: 1F1B Schedule

**One-Forward-One-Backward**: Interleave forward and backward passes to reduce memory.

<div class="diagram">
<div class="diagram-title">PipeDream 1F1B Schedule</div>
<div class="flow-node green wide" style="font-family: monospace; white-space: pre; text-align: left;">
GPU 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
GPU 1: __ F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7 __
GPU 2: __ __ F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7 __ __
GPU 3: __ __ __ F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 __ __ __
</div>
</div>

**Advantages**:
- Lower memory: Only stores activations for in-flight micro-batches
- Similar bubble fraction to GPipe
- Steadier GPU utilization

### Interleaved Pipeline Schedules

**Concept**: Each GPU handles **multiple non-contiguous stages** (e.g., GPU 0 handles stages 0, 4, 8, 12).

Benefits:
- Reduces bubble fraction to $\frac{p-1}{m+p-1} \cdot \frac{1}{v}$ where $v$ is the number of stages per GPU
- More balanced memory usage

### Pipeline Parallelism Implementation

```python
# Using DeepSpeed Pipeline Parallelism
from deepspeed.pipe import PipelineModule, LayerSpec

# Define model as a list of layers
layers = [
    LayerSpec(TransformerLayer, hidden_size=1024, ...),
    LayerSpec(TransformerLayer, hidden_size=1024, ...),
    # ... 96 total layers
]

# Wrap with PipelineModule
model = PipelineModule(
    layers=layers,
    num_stages=4,  # Split across 4 GPUs
    partition_method='uniform',  # Equal layers per stage
)

# DeepSpeed configuration
ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,  # 16 micro-batches
    "pipeline": {
        "pipe_partitioned": True,
        "grad_partitioned": True,
    }
}
```

---

## 5. Tensor Parallelism (TP)

Split individual tensors (weight matrices) across GPUs. **Megatron-LM** pioneered this approach.

### Column-Parallel Linear Layer

Split weight matrix $W \in \mathbb{R}^{h \times 4h}$ column-wise across $N$ GPUs:

$$
Y = XW = X [W_0 \mid W_1 \mid \cdots \mid W_{N-1}]
$$

Each GPU computes:
$$
Y_i = X W_i
$$

Then **AllGather** or **Concatenate** results: $Y = [Y_0 \mid Y_1 \mid \cdots \mid Y_{N-1}]$

<div class="diagram">
<div class="diagram-title">Column-Parallel Linear Layer</div>
<div class="flow">
<div class="flow-node accent wide">Input X (replicated on all GPUs)</div>
<div class="flow-arrow accent"></div>
<div class="flow-h">
<div class="flow-node green">GPU 0<br/>W₀<br/>Y₀ = XW₀</div>
<div class="flow-node green">GPU 1<br/>W₁<br/>Y₁ = XW₁</div>
<div class="flow-node green">GPU 2<br/>W₂<br/>Y₂ = XW₂</div>
<div class="flow-node green">GPU N-1<br/>Wₙ₋₁<br/>Yₙ₋₁ = XWₙ₋₁</div>
</div>
<div class="flow-arrow purple"></div>
<div class="flow-node purple wide">AllGather: Y = [Y₀ | Y₁ | Y₂ | ... | Yₙ₋₁]</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">Output Y (replicated on all GPUs)</div>
</div>
</div>

### Row-Parallel Linear Layer

Split weight matrix $W \in \mathbb{R}^{4h \times h}$ row-wise:

$$
W = \begin{bmatrix} W_0 \\ W_1 \\ \vdots \\ W_{N-1} \end{bmatrix}
$$

Split input $X$ correspondingly: $X = [X_0 \mid X_1 \mid \cdots \mid X_{N-1}]$

Each GPU computes: $Y_i = X_i W_i$

Then **ReduceScatter** or **AllReduce**: $Y = \sum_{i=0}^{N-1} Y_i$

<div class="diagram">
<div class="diagram-title">Row-Parallel Linear Layer</div>
<div class="flow">
<div class="flow-node accent wide">Input X (split across GPUs: X₀, X₁, ..., Xₙ₋₁)</div>
<div class="flow-arrow accent"></div>
<div class="flow-h">
<div class="flow-node cyan">GPU 0<br/>W₀, X₀<br/>Y₀ = X₀W₀</div>
<div class="flow-node cyan">GPU 1<br/>W₁, X₁<br/>Y₁ = X₁W₁</div>
<div class="flow-node cyan">GPU 2<br/>W₂, X₂<br/>Y₂ = X₂W₂</div>
<div class="flow-node cyan">GPU N-1<br/>Wₙ₋₁, Xₙ₋₁<br/>Yₙ₋₁ = Xₙ₋₁Wₙ₋₁</div>
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node orange wide">AllReduce: Y = Y₀ + Y₁ + Y₂ + ... + Yₙ₋₁</div>
<div class="flow-arrow orange"></div>
<div class="flow-node green wide">Output Y (replicated on all GPUs)</div>
</div>
</div>

### Transformer Layer with Tensor Parallelism

**MLP block**:
1. First linear (h → 4h): **Column-parallel** (no communication, output split)
2. GeLU activation: **Applied independently** (no communication)
3. Second linear (4h → h): **Row-parallel** (AllReduce on output)

**Self-Attention block**:
- **Q, K, V projections**: Column-parallel (split attention heads across GPUs)
- **Attention computation**: Each GPU computes attention for its heads independently
- **Output projection**: Row-parallel (AllReduce)

```python
# Megatron-LM style tensor-parallel layer
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = torch.distributed.get_world_size(tp_group)
        
        # Each GPU holds output_size // tp_size columns
        self.weight = nn.Parameter(
            torch.empty(input_size, output_size // self.tp_size)
        )
    
    def forward(self, x):
        # x is replicated across all GPUs
        # Each GPU computes its portion
        output_parallel = F.linear(x, self.weight)
        
        # AllGather to get full output
        output = all_gather(output_parallel, self.tp_group)
        return output

class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = torch.distributed.get_world_size(tp_group)
        
        # Each GPU holds input_size // tp_size rows
        self.weight = nn.Parameter(
            torch.empty(input_size // self.tp_size, output_size)
        )
    
    def forward(self, x):
        # x is split across GPUs along last dimension
        # Each GPU gets a chunk
        output_parallel = F.linear(x, self.weight)
        
        # AllReduce to sum results
        output = all_reduce(output_parallel, self.tp_group)
        return output
```

### Communication Volume

**Per transformer layer** with tensor parallelism degree $N$:
- Forward: 2 AllReduce operations (MLP output + attention output)
- Backward: 2 AllReduce operations (gradient w.r.t. input)

**Total**: 4 AllReduce per layer

For a model with $L$ layers and hidden size $h$, batch size $b$, sequence length $s$:

$$
\text{Communication volume} = 4L \cdot b \cdot s \cdot h \cdot 2 \text{ bytes (FP16)}
$$

**Critical**: Tensor parallelism requires **fast GPU-to-GPU interconnect** (NVLink, NVSwitch). Not suitable across nodes without high-bandwidth InfiniBand.

### Multi-Head Attention Splitting

For $H$ attention heads across $N$ GPUs:
- Each GPU handles $H/N$ heads
- Parallelizes QKV computation and attention independently
- No cross-GPU communication during attention computation itself

$$
\text{Attention}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

Each GPU computes attention for its subset of heads $\{i \cdot H/N, \ldots, (i+1) \cdot H/N - 1\}$.

---

## 6. Sequence Parallelism (SP)

**Problem**: Tensor parallelism leaves some operations **replicated** (LayerNorm, Dropout). Activations for these ops are duplicated across all TP GPUs.

**Solution**: Split activations along the **sequence dimension** for non-TP operations.

<div class="diagram">
<div class="diagram-title">Sequence Parallelism Extension to Tensor Parallelism</div>
<div class="diagram-grid cols-2">
<div class="diagram-card accent">
<div class="card-icon">❌</div>
<div class="card-title">Without Sequence Parallelism</div>
<div class="card-desc">LayerNorm and Dropout replicate full sequence on each GPU → TP-fold memory waste</div>
</div>
<div class="diagram-card green">
<div class="card-icon">✅</div>
<div class="card-title">With Sequence Parallelism</div>
<div class="card-desc">Split sequence dimension across TP GPUs → Reduces activation memory by TP factor</div>
</div>
</div>
</div>

### Mechanism

For a sequence of length $S$ split across $N$ TP GPUs:
- **Each GPU processes**: $S/N$ tokens
- **Operations affected**: LayerNorm, Dropout (element-wise ops)
- **Communication required**: AllGather before column-parallel linear, ReduceScatter after row-parallel linear

### Memory Savings

**Activation memory reduction**:

Without SP: $M_{\text{act}} = b \cdot s \cdot h \cdot L \cdot N_{\text{TP}}$ (replicated across TP GPUs)

With SP: $M_{\text{act}} = b \cdot s \cdot h \cdot L$ (evenly distributed)

**Savings factor**: $N_{\text{TP}}$

### Megatron-LM v3 Implementation

Introduced in Megatron-LM v3 (2022). Seamlessly integrates with tensor parallelism:

```python
# Pseudocode for sequence-parallel region
def transformer_layer_with_sp(x, tp_group):
    # x: [batch, seq_len, hidden] split along seq_len
    
    # LayerNorm in sequence-parallel mode (each GPU does seq_len/TP)
    x = layer_norm_sp(x)
    
    # AllGather before column-parallel (need full sequence)
    x_full = all_gather(x, dim=1, group=tp_group)  # [batch, seq_len, hidden]
    
    # Attention (column-parallel Q,K,V)
    attn_out = attention_column_parallel(x_full)  # [batch, seq_len, hidden/TP]
    
    # Row-parallel output projection + ReduceScatter
    attn_out = output_proj_row_parallel(attn_out)  # AllReduce
    attn_out = reduce_scatter(attn_out, dim=1, group=tp_group)  # [batch, seq_len/TP, hidden]
    
    # Residual + LayerNorm (sequence-parallel)
    x = x + attn_out
    x = layer_norm_sp(x)
    
    # MLP: similar pattern
    x_full = all_gather(x, dim=1, group=tp_group)
    mlp_out = mlp_column_parallel(x_full)
    mlp_out = mlp_row_parallel(mlp_out)
    mlp_out = reduce_scatter(mlp_out, dim=1, group=tp_group)
    
    x = x + mlp_out
    return x
```

---

## 7. Context Parallelism (CP)

For **extremely long sequences** (100K - 1M+ tokens), even sequence parallelism isn't enough.

**Context Parallelism** (aka **Ring Attention**): Split input sequence into chunks, distribute across GPUs, use ring communication for attention.

### Ring Attention Mechanism

<div class="diagram">
<div class="diagram-title">Ring Attention: Long Context Processing</div>
<div class="flow">
<div class="flow-node accent wide">Input Sequence (1M tokens) → Split into 8 chunks of 125K each</div>
<div class="flow-arrow accent"></div>
<div class="flow-h">
<div class="flow-node green">GPU 0<br/>Chunk 0<br/>Q₀, K₀, V₀</div>
<div class="flow-node green">GPU 1<br/>Chunk 1<br/>Q₁, K₁, V₁</div>
<div class="flow-node green">GPU 2<br/>Chunk 2<br/>Q₂, K₂, V₂</div>
<div class="flow-node green">GPU 7<br/>Chunk 7<br/>Q₇, K₇, V₇</div>
</div>
<div class="flow-arrow purple"></div>
<div class="flow-node purple wide">Ring Rotation: Each GPU receives K,V blocks from neighbors</div>
<div class="flow-arrow purple"></div>
<div class="flow-node orange wide">Compute Attention: Q_i × (K₀, K₁, ..., K₇) → Full context attention</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan wide">Output: Each GPU has attended to entire 1M-token sequence</div>
</div>
</div>

### Algorithm

For $N$ GPUs, each holding sequence chunk $i$:

1. **Compute local Q, K, V**: $Q_i, K_i, V_i = \text{Linear}(X_i)$
2. **Ring communication** (N-1 steps):
   - Step $t$: GPU $i$ receives $K_j, V_j$ where $j = (i - t) \mod N$
   - Compute partial attention: $A_i^{(t)} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) V_j$
   - Accumulate: $\text{Out}_i \mathrel{+}= A_i^{(t)}$
3. **Normalize**: Re-normalize attention scores across all chunks

### Differences from Sequence Parallelism

| Aspect | Sequence Parallelism | Context Parallelism |
|--------|---------------------|---------------------|
| **Splits** | Sequence dimension for non-TP ops | Sequence dimension for attention itself |
| **Use case** | Moderate sequences (2K-32K) | Ultra-long sequences (100K-1M+) |
| **Communication** | AllGather/ReduceScatter | Ring communication |
| **Attention** | Full attention on gathered sequence | Blocked attention with ring |
| **Memory** | Reduces activation memory | Enables otherwise-impossible sequences |

### Enabling Million-Token Contexts

Without CP: A 1M-token sequence with attention requires $O(S^2)$ memory:

$$
M_{\text{attn}} = \text{batch} \times \text{heads} \times S^2 \times 2 \text{ bytes}
$$

For $S = 1{,}000{,}000$: ~2 TB just for attention scores (batch=1, 32 heads, FP16).

With CP across 1024 GPUs: Each GPU handles $S/1024 \approx 1000$ tokens → Memory per GPU: ~2 GB.

---

## 8. Expert Parallelism (EP)

For **Mixture-of-Experts (MoE)** models, which contain many specialized sub-networks (experts).

### MoE Basics

A MoE layer replaces a dense FFN with $E$ expert networks:

$$
y = \sum_{i=1}^{E} G(x)_i \cdot \text{Expert}_i(x)
$$

where $G(x)$ is a gating function (router) that determines expert weights.

**Sparse activation**: Each token routed to top-$k$ experts (typically $k=1$ or $k=2$).

### Expert Parallelism Strategy

**Distribute experts across GPUs**: Each GPU holds a subset of experts.

**All-to-All communication**: Route tokens to the GPU containing the chosen expert.

<div class="diagram">
<div class="diagram-title">Expert Parallelism in MoE</div>
<div class="flow">
<div class="flow-node accent wide">Input: Tokens distributed across GPUs</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green wide">Router: Determine which expert(s) for each token</div>
<div class="flow-arrow green"></div>
<div class="flow-node purple wide">All-to-All: Route tokens to GPUs holding selected experts</div>
<div class="flow-arrow purple"></div>
<div class="flow-h">
<div class="flow-node orange">GPU 0<br/>Expert 0,1<br/>Process tokens</div>
<div class="flow-node orange">GPU 1<br/>Expert 2,3<br/>Process tokens</div>
<div class="flow-node orange">GPU 2<br/>Expert 4,5<br/>Process tokens</div>
<div class="flow-node orange">GPU N-1<br/>Expert E-2,E-1<br/>Process tokens</div>
</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node cyan wide">All-to-All: Route expert outputs back to original GPUs</div>
<div class="flow-arrow cyan"></div>
<div class="flow-node accent wide">Output: Combine expert outputs</div>
</div>
</div>

### Load Balancing Challenge

**Problem**: Gating network might route most tokens to a few "popular" experts → Load imbalance.

**Solutions**:

1. **Capacity factor**: Limit tokens per expert:
   $$
   \text{Capacity} = \frac{\text{tokens per batch}}{E} \times \text{capacity factor}
   $$
   Drop tokens exceeding capacity (or route to overflow expert).

2. **Load balancing loss**: Add auxiliary loss to encourage uniform expert usage:
   $$
   L_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i
   $$
   where $f_i$ = fraction of tokens routed to expert $i$, $P_i$ = expert probability.

3. **Expert choice routing** (DeepSeek-MoE): Experts select tokens instead of tokens selecting experts.

### Communication Volume

**All-to-All** for token routing (twice per MoE layer: to experts, from experts):

$$
\text{Comm volume} = 2 \times \text{batch size} \times \text{seq len} \times \text{hidden dim} \times k
$$

where $k$ = number of experts per token.

### Implementations

**GShard** (Google, 2020): Top-2 routing with capacity factor.

**DeepSeek-MoE** (2024): Fine-grained experts with expert-choice routing.

```python
# Simplified MoE layer with expert parallelism
class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert_parallel_group):
        super().__init__()
        self.ep_group = expert_parallel_group
        self.ep_size = torch.distributed.get_world_size(expert_parallel_group)
        
        # Each GPU holds num_experts // ep_size experts
        self.num_local_experts = num_experts // self.ep_size
        self.experts = nn.ModuleList([
            FFN(hidden_size) for _ in range(self.num_local_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)
    
    def forward(self, x):
        # x: [batch, seq_len, hidden]
        # Gating
        router_logits = self.gate(x)  # [batch, seq_len, num_experts]
        routing_weights, selected_experts = torch.topk(router_logits, k=2, dim=-1)
        
        # All-to-All: route tokens to expert GPUs
        x_routed = all_to_all_route_to_experts(x, selected_experts, self.ep_group)
        
        # Compute expert outputs locally
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_inputs = x_routed[i]  # Tokens routed to this expert
            expert_outputs.append(expert(expert_inputs))
        
        # All-to-All: route outputs back
        output = all_to_all_route_from_experts(expert_outputs, self.ep_group)
        return output
```

---

## 9. ZeRO (Zero Redundancy Optimizer)

**DeepSpeed's breakthrough**: Eliminate memory redundancy in data-parallel training.

### The Redundancy Problem

In standard data parallelism:
- **Model parameters**: Replicated on all $N$ GPUs → $N \times \Psi$ total
- **Gradients**: Replicated on all $N$ GPUs → $N \times \Psi$ total
- **Optimizer states** (Adam): Replicated on all $N$ GPUs → $N \times 2\Psi$ total

**Total memory across cluster**: $N \times (16\Psi + M_{\text{act}})$

**But only need**: $1 \times (16\Psi + M_{\text{act}})$ logically!

**Insight**: Partition optimizer states, gradients, and parameters across GPUs. Gather only when needed.

### ZeRO Stage 1: Optimizer State Partitioning

Each GPU stores only $1/N$ of optimizer states.

**Memory per GPU**:

$$
M_{\text{GPU}} = 2\Psi + 2\Psi + \frac{8\Psi}{N} + M_{\text{act}} = 4\Psi + \frac{8\Psi}{N} + M_{\text{act}}
$$

For Adam (8 bytes per parameter for momentum + variance in FP32):
- Without ZeRO-1: $8\Psi$ per GPU
- With ZeRO-1: $8\Psi / N$ per GPU

**Savings**: ~$8\Psi \cdot (N-1)/N$ per GPU for large $N$.

**Communication**: AllGather updated parameters after optimizer step (same as DDP).

### ZeRO Stage 2: Gradient Partitioning

Additionally partition gradients.

**Memory per GPU**:

$$
M_{\text{GPU}} = 2\Psi + \frac{2\Psi}{N} + \frac{8\Psi}{N} + M_{\text{act}} = 2\Psi + \frac{10\Psi}{N} + M_{\text{act}}
$$

**Communication**: ReduceScatter instead of AllReduce during backward pass. Each GPU accumulates only its partition of gradients.

### ZeRO Stage 3: Parameter Partitioning

Partition model parameters themselves. Each GPU stores only $\Psi / N$ parameters.

**Memory per GPU**:

$$
M_{\text{GPU}} = \frac{2\Psi}{N} + \frac{2\Psi}{N} + \frac{8\Psi}{N} + M_{\text{act}} = \frac{12\Psi}{N} + M_{\text{act}}
$$

**For mixed precision with FP32 master weights**:

$$
M_{\text{GPU}} = \frac{16\Psi}{N} + M_{\text{act}}
$$

**Communication**:
- **Forward**: AllGather parameters for each layer before computing
- **Backward**: AllGather parameters for each layer, then ReduceScatter gradients
- **Optimizer step**: Local update on partitioned parameters

<div class="diagram">
<div class="diagram-title">ZeRO Stages Memory Comparison (175B Model, FP16+FP32)</div>
<div class="diagram-grid cols-3">
<div class="diagram-card accent">
<div class="card-icon">1️⃣</div>
<div class="card-title">ZeRO Stage 1</div>
<div class="card-desc">Partition optimizer states<br/>~4Ψ + 8Ψ/N per GPU<br/>175B → ~1.4TB (N=1) → ~800GB (N=8)</div>
</div>
<div class="diagram-card green">
<div class="card-icon">2️⃣</div>
<div class="card-title">ZeRO Stage 2</div>
<div class="card-desc">+ Partition gradients<br/>~2Ψ + 10Ψ/N per GPU<br/>175B → ~700GB (N=1) → ~570GB (N=8)</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">3️⃣</div>
<div class="card-title">ZeRO Stage 3</div>
<div class="card-desc">+ Partition parameters<br/>~16Ψ/N per GPU<br/>175B → 2.8TB/N<br/>N=8 → 350GB per GPU ✅</div>
</div>
</div>
</div>

### ZeRO-Offload and ZeRO-Infinity

**ZeRO-Offload**: Offload optimizer states to CPU memory. Reduces GPU memory further but adds CPU-GPU transfer overhead.

**ZeRO-Infinity**: Offload to NVMe SSD for truly massive models. Enables trillion-parameter models.

### PyTorch FSDP (Fully Sharded Data Parallel)

PyTorch's native implementation of ZeRO-3:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# Configure mixed precision
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Wrap model with FSDP (ZeRO-3 equivalent)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=mp_policy,
    device_id=torch.cuda.current_device(),
)

# Training loop (same as regular DDP)
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch)  # Auto AllGather params
    loss = criterion(output, labels)
    loss.backward()  # Auto ReduceScatter gradients
    optimizer.step()  # Update local shard
```

### Communication Volume Analysis

**ZeRO-3 vs DDP**:

| Operation | DDP | ZeRO-3 |
|-----------|-----|--------|
| **Forward** | None | AllGather params ($2(N-1)\Psi/N$ per layer) |
| **Backward** | AllReduce grads ($2(N-1)\Psi/N$) | AllGather params + ReduceScatter grads |
| **Total per iteration** | $2(N-1)\Psi/N \approx 2\Psi$ | $4(N-1)\Psi/N \approx 4\Psi$ |

**Trade-off**: ZeRO-3 uses **2× more communication** than DDP, but enables training models that wouldn't fit in memory otherwise.

---

## 10. 3D Parallelism

Combine **Data Parallelism (DP) × Tensor Parallelism (TP) × Pipeline Parallelism (PP)** for massive-scale training.

### Dimensionality Decomposition

For $G$ total GPUs, decompose as:

$$
G = D \times T \times P
$$

where:
- $D$ = data parallel degree
- $T$ = tensor parallel degree (TP)
- $P$ = pipeline parallel degree (PP)

**GPU allocation**:
1. **Tensor parallel group**: $T$ GPUs (must be in same node with fast NVLink)
2. **Pipeline parallel stages**: $P$ stages
3. **Data parallel replicas**: $D$ copies of the entire (TP+PP) model

<div class="diagram">
<div class="diagram-title">3D Parallelism: DP × TP × PP</div>
<div class="flow">
<div class="flow-node accent wide">Total: 64 GPUs = DP:4 × TP:4 × PP:4</div>
<div class="flow-arrow accent"></div>
<div class="flow-h">
<div class="flow-node green" style="padding: 15px;">
<strong>Data Replica 0</strong><br/>
PP Stage 0: [TP: GPU 0-3]<br/>
PP Stage 1: [TP: GPU 4-7]<br/>
PP Stage 2: [TP: GPU 8-11]<br/>
PP Stage 3: [TP: GPU 12-15]
</div>
<div class="flow-node green" style="padding: 15px;">
<strong>Data Replica 1</strong><br/>
PP Stage 0: [TP: GPU 16-19]<br/>
PP Stage 1: [TP: GPU 20-23]<br/>
PP Stage 2: [TP: GPU 24-27]<br/>
PP Stage 3: [TP: GPU 28-31]
</div>
<div class="flow-node green" style="padding: 15px;">
<strong>Data Replica 2</strong><br/>
PP Stage 0: [TP: GPU 32-35]<br/>
PP Stage 1: [TP: GPU 36-39]<br/>
PP Stage 2: [TP: GPU 40-43]<br/>
PP Stage 3: [TP: GPU 44-47]
</div>
<div class="flow-node green" style="padding: 15px;">
<strong>Data Replica 3</strong><br/>
PP Stage 0: [TP: GPU 48-51]<br/>
PP Stage 1: [TP: GPU 52-55]<br/>
PP Stage 2: [TP: GPU 56-59]<br/>
PP Stage 3: [TP: GPU 60-63]
</div>
</div>
</div>
</div>

### Example: Training GPT-3 175B on 1024 GPUs

**Configuration**: TP=8, PP=16, DP=8 (8 × 16 × 8 = 1024)

**Why this configuration?**
- **TP=8**: Fits within a single DGX node (8× A100 with NVLink)
- **PP=16**: Splits 96-layer model into 6 layers per stage
- **DP=8**: Creates 8 data-parallel replicas (effective batch size × 8)

**Memory per GPU** (simplified):
- Model size: 175B parameters → 350GB in FP16
- With TP=8: 350GB / 8 = 43.75GB per GPU (just parameters)
- With PP=16: 43.75GB / 16 = 2.7GB per stage per GPU
- Plus optimizer states, gradients, activations → fits in 80GB A100

**Communication patterns**:
- **TP group** (intra-node NVLink): 4 AllReduce per layer (fast, low latency)
- **PP group** (inter-node InfiniBand): Point-to-point activation/gradient transfer
- **DP group**: AllReduce gradients after backward (inter-node)

### Megatron-LM and DeepSpeed Integration

```python
# Megatron-LM 3D parallelism setup
import torch.distributed as dist

# Initialize process groups
world_size = dist.get_world_size()
rank = dist.get_rank()

# Configuration
TP_SIZE = 8
PP_SIZE = 16
DP_SIZE = world_size // (TP_SIZE * PP_SIZE)

# Create process groups
# Tensor parallel group: GPUs within same node
tp_group = create_tensor_parallel_group(TP_SIZE)

# Pipeline parallel group: GPUs forming pipeline stages
pp_group = create_pipeline_parallel_group(PP_SIZE)

# Data parallel group: Corresponding GPUs across data replicas
dp_group = create_data_parallel_group(DP_SIZE)

# Model initialization
model = GPTModel(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    tensor_parallel_group=tp_group,
    pipeline_parallel_group=pp_group,
)

# Wrap with data parallelism (DDP or FSDP)
model = DistributedDataParallel(model, process_group=dp_group)
```

---

## 11. 5D Parallelism

For frontier models (GPT-4, Gemini, etc.), add **Expert Parallelism (EP)** and **Context Parallelism (CP)**:

$$
G = D \times T \times P \times E \times C
$$

<div class="diagram">
<div class="diagram-title">5D Parallelism: All Techniques Combined</div>
<div class="diagram-grid cols-5">
<div class="diagram-card accent">
<div class="card-icon">📊</div>
<div class="card-title">Data Parallel</div>
<div class="card-desc">D replicas<br/>Batch splitting<br/>Gradient sync</div>
</div>
<div class="diagram-card green">
<div class="card-icon">🔀</div>
<div class="card-title">Tensor Parallel</div>
<div class="card-desc">T GPUs/node<br/>Matrix sharding<br/>NVLink comm</div>
</div>
<div class="diagram-card purple">
<div class="card-icon">⛓️</div>
<div class="card-title">Pipeline Parallel</div>
<div class="card-desc">P stages<br/>Layer partitioning<br/>Micro-batches</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">🎓</div>
<div class="card-title">Expert Parallel</div>
<div class="card-desc">E expert groups<br/>MoE routing<br/>All-to-All</div>
</div>
<div class="diagram-card cyan">
<div class="card-icon">📏</div>
<div class="card-title">Context Parallel</div>
<div class="card-desc">C sequence chunks<br/>Ring attention<br/>Long context</div>
</div>
</div>
</div>

### Example: Training a 1.8T MoE with 1M Context

**Model**: 1.8 trillion parameters (MoE with 256 experts, top-2 routing)

**Configuration**: 
- **TP** = 8 (intra-node)
- **PP** = 16 (64 layers → 4 per stage)
- **EP** = 32 (256 experts → 8 per GPU group)
- **CP** = 8 (1M tokens → 125K per GPU)
- **DP** = 4 (data replicas)

**Total GPUs**: 8 × 16 × 32 × 8 × 4 = 131,072 GPUs (hypothetical exascale cluster)

### How to Assign Dimensions

**General heuristics**:

1. **Tensor Parallelism (T)**: 
   - Size: 2, 4, or 8 (must fit within a node)
   - Constraint: Requires fastest interconnect (NVLink)
   - Use: When model layers don't fit in single GPU

2. **Pipeline Parallelism (P)**:
   - Size: Number of layers / 4-8 layers per stage
   - Use: When model is too deep even with TP

3. **Expert Parallelism (E)**:
   - Size: Number of experts / 4-16 experts per GPU
   - Use: Only for MoE models

4. **Context Parallelism (C)**:
   - Size: Sequence length / 32K-128K tokens per GPU
   - Use: Only when sequence > 100K tokens

5. **Data Parallelism (D)**:
   - Size: Remaining GPUs = Total GPUs / (T × P × E × C)
   - Use: Always (for batch parallelism)

### Memory Calculation for 5D Parallelism

**Parameters per GPU**:

$$
M_{\text{params}} = \frac{\Psi}{T \times P \times E \times \text{sharing factor}}
$$

**With ZeRO-3 across DP dimension**:

$$
M_{\text{params}} = \frac{\Psi}{T \times P \times E \times D}
$$

**Activations per GPU**:

$$
M_{\text{act}} = \frac{b \times s \times h \times L}{D \times C \times \text{micro-batches}}
$$

---

## 12. Practical Decision Guide

How do you choose the right parallelism strategy for your model?

### Decision Table

| Model Size | GPU Memory | Sequence Length | Model Type | Recommended Strategy |
|------------|-----------|-----------------|------------|---------------------|
| < 1B | < 20GB | < 8K | Dense | **DDP only** |
| 1-7B | 20-40GB | < 8K | Dense | **DDP or FSDP** |
| 7-13B | 40-60GB | < 16K | Dense | **FSDP (ZeRO-3)** |
| 13-70B | 60GB+ | < 16K | Dense | **DP + TP (2-4)** or **FSDP** |
| 70-175B | N/A | < 32K | Dense | **3D: DP + TP (4-8) + PP (4-16)** |
| 175B+ | N/A | < 32K | Dense | **3D: DP + TP (8) + PP (16+)** |
| Any | N/A | 100K-1M | Dense | **+ Context Parallelism (CP)** |
| Any | N/A | Any | MoE | **+ Expert Parallelism (EP)** |

### Decision Flowchart

<div class="diagram">
<div class="diagram-title">Parallelism Strategy Selection</div>
<div class="flow">
<div class="flow-node accent wide">Start: Model size Ψ, GPU memory M</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green">Does Ψ fit in single GPU?<br/>(16Ψ + activations < M)</div>
<div class="flow-arrow green"></div>
<div class="flow-h">
<div class="flow-node purple">YES → Use DDP<br/>(Simple & Fast)</div>
<div class="flow-node orange">NO → Continue</div>
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node cyan">Does Ψ fit with ZeRO-3?<br/>(16Ψ/N < M)</div>
<div class="flow-arrow cyan"></div>
<div class="flow-h">
<div class="flow-node purple">YES → Use FSDP/ZeRO-3</div>
<div class="flow-node orange">NO → Continue</div>
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node pink">Is it MoE?</div>
<div class="flow-arrow pink"></div>
<div class="flow-h">
<div class="flow-node teal">YES → Add EP</div>
<div class="flow-node yellow">NO → Continue</div>
</div>
<div class="flow-arrow yellow"></div>
<div class="flow-node accent">Add TP (2-8) + PP (as needed)</div>
<div class="flow-arrow accent"></div>
<div class="flow-node green">Sequence > 100K?</div>
<div class="flow-arrow green"></div>
<div class="flow-h">
<div class="flow-node purple">YES → Add CP</div>
<div class="flow-node orange">NO → Done</div>
</div>
<div class="flow-arrow orange"></div>
<div class="flow-node accent wide">Final: 3D/4D/5D Parallelism</div>
</div>
</div>

### Interconnect Requirements

Different parallelism strategies have different bandwidth needs:

| Strategy | Communication | Bandwidth Required | Recommended Hardware |
|----------|---------------|-------------------|---------------------|
| **DP/DDP** | AllReduce gradients | Medium (100-400 Gbps) | InfiniBand HDR 200, RoCE |
| **TP** | AllReduce per layer | **Very High** (900+ Gbps) | **NVLink, NVSwitch** |
| **PP** | Point-to-point activations | Low-Medium (50-200 Gbps) | InfiniBand, PCIe |
| **EP** | All-to-All routing | Medium-High (200-400 Gbps) | InfiniBand HDR |
| **CP** | Ring KV transfer | Medium (100-200 Gbps) | InfiniBand |
| **ZeRO-3** | AllGather/ReduceScatter | High (400-800 Gbps) | InfiniBand HDR 200+ |

**Critical rule**: **Never use TP across nodes** without ultra-fast interconnect (800+ Gbps). Latency kills TP performance.

### Communication Overhead Analysis

<div class="diagram">
<div class="diagram-title">Communication vs Compute Trade-offs</div>
<div class="diagram-grid cols-3">
<div class="diagram-card green">
<div class="card-icon">⚡</div>
<div class="card-title">Low Overhead</div>
<div class="card-desc"><strong>DDP</strong>: 1 AllReduce/iteration<br/><strong>PP</strong>: P2P transfers only<br/>Best scaling efficiency</div>
</div>
<div class="diagram-card yellow">
<div class="card-icon">⚖️</div>
<div class="card-title">Medium Overhead</div>
<div class="card-desc"><strong>ZeRO-2</strong>: ReduceScatter<br/><strong>EP</strong>: All-to-All per MoE layer<br/>Acceptable with good interconnect</div>
</div>
<div class="diagram-card orange">
<div class="card-icon">🔥</div>
<div class="card-title">High Overhead</div>
<div class="card-desc"><strong>TP</strong>: 4 AllReduce/layer<br/><strong>ZeRO-3</strong>: 2 AllGather/layer<br/>Needs NVLink-class bandwidth</div>
</div>
</div>
</div>

### Practical Example Configurations

**Configuration 1: LLaMA-2 70B on 8× A100 (80GB)**
- Model: 70B params = 140GB FP16
- Strategy: **FSDP (ZeRO-3)** only
- Memory per GPU: 140GB/8 + 20GB (optimizer states/8) + 30GB (activations) = ~68GB ✅
- Scaling: Can increase to 16-32 GPUs for larger batches

**Configuration 2: GPT-3 175B on 256× A100**
- Model: 175B params = 350GB FP16
- Strategy: **DP=8, TP=4, PP=8** (8 × 4 × 8 = 256)
- Within node: 4-way TP (connected via NVSwitch)
- Across nodes: 8-way PP (32 layers → 4 per stage)
- Data parallel: 8 replicas (batch size × 8)

**Configuration 3: 1T Dense Model on 4096× H100**
- Model: 1T params
- Strategy: **DP=16, TP=8, PP=32** (16 × 8 × 32 = 4096)
- TP=8: Within each node
- PP=32: Deep pipeline for 128-layer model
- DP=16: Moderate data parallelism

**Configuration 4: MoE 1.8T on 8192× H100**
- Model: 1.8T params, 256 experts
- Strategy: **DP=4, TP=8, PP=16, EP=16** (4 × 8 × 16 × 16 = 8192)
- EP=16: 256 experts / 16 = 16 experts per GPU group
- Other dimensions: Standard 3D parallelism

### Memory Estimation Formulas

**For dense models with mixed precision (FP16 params, FP32 optimizer)**:

Without parallelism:
$$
M = 16\Psi + 12 \cdot b \cdot s \cdot h \cdot L
$$

With 3D parallelism (DP × TP × PP):
$$
M = \frac{16\Psi}{TP \times PP} + \frac{12 \cdot b \cdot s \cdot h \cdot L}{DP \times \text{micro-batches}}
$$

With 3D + ZeRO-3:
$$
M = \frac{16\Psi}{DP \times TP \times PP} + \frac{12 \cdot b \cdot s \cdot h \cdot L}{DP \times \text{micro-batches}}
$$

### Optimization Tips

**1. Maximize TP within nodes**
- Use TP=4 or TP=8 on 4-GPU or 8-GPU nodes
- Leverage NVLink/NVSwitch for minimal latency

**2. Use PP sparingly**
- Pipeline bubbles waste compute
- Prefer ZeRO-3 if memory is the only issue

**3. Scale DP for throughput**
- After fitting model with TP+PP+ZeRO, add more DP
- DP scales almost linearly with good interconnect

**4. Activation checkpointing**
- Trade compute for memory: recompute activations during backward
- Reduces memory by ~40% with ~20% slowdown

```python
# PyTorch activation checkpointing
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # Use checkpointing to save memory
        return checkpoint(self._forward, x)
    
    def _forward(self, x):
        # Actual layer computation
        x = self.attention(x)
        x = self.mlp(x)
        return x
```

**5. Gradient accumulation**
- Simulate larger batches without memory overhead
- Reduces communication frequency

```python
# Gradient accumulation
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Timeline of Parallelism Innovations

<div class="timeline">
<div class="timeline-item">
<div class="timeline-year">2012</div>
<div class="timeline-title">AlexNet Data Parallelism</div>
<div class="timeline-desc">Pioneered model-parallel training across 2 GPUs</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2014</div>
<div class="timeline-title">Model Parallelism</div>
<div class="timeline-desc">Simple layer-wise splitting for large models</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2018</div>
<div class="timeline-title">GPipe (Pipeline Parallelism)</div>
<div class="timeline-desc">Google introduces micro-batch pipelining</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2019</div>
<div class="timeline-title">Megatron-LM (Tensor Parallelism)</div>
<div class="timeline-desc">NVIDIA's intra-layer tensor parallelism</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2020</div>
<div class="timeline-title">ZeRO (DeepSpeed)</div>
<div class="timeline-desc">Microsoft eliminates memory redundancy</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2021</div>
<div class="timeline-title">3D Parallelism</div>
<div class="timeline-desc">Megatron-DeepSpeed combines DP+TP+PP</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2022</div>
<div class="timeline-title">Sequence Parallelism</div>
<div class="timeline-desc">Megatron-LM v3 adds sequence-dimension splitting</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2023</div>
<div class="timeline-title">Context Parallelism</div>
<div class="timeline-desc">Ring Attention enables million-token contexts</div>
</div>
<div class="timeline-item">
<div class="timeline-year">2024</div>
<div class="timeline-title">5D Parallelism</div>
<div class="timeline-desc">Frontier models use DP+TP+PP+EP+CP</div>
</div>
</div>

---

## Summary

Parallelism is **essential** for modern deep learning at scale. Key takeaways:

### Fundamental Principles

1. **No single solution**: Different models and scales require different strategies
2. **Memory vs Communication**: Trade-offs exist between memory efficiency and communication overhead
3. **Hierarchical approach**: Start simple (DDP), add complexity as needed (TP → PP → ZeRO → 3D/5D)

### Strategy Selection

| Goal | Best Approach |
|------|---------------|
| **Simple small model** | DDP |
| **Medium model (fits with sharding)** | FSDP/ZeRO-3 |
| **Large model (doesn't fit even sharded)** | 3D Parallelism |
| **MoE model** | Add Expert Parallelism |
| **Long context** | Add Context Parallelism |
| **Maximum throughput** | Maximize data parallelism |
| **Maximum model size** | Maximize all dimensions |

### Critical Constraints

✅ **Do**:
- Use TP within nodes (with NVLink)
- Combine strategies for large models
- Monitor communication overhead
- Use activation checkpointing for memory

❌ **Don't**:
- Use TP across nodes (without ultra-fast interconnect)
- Over-parallelize small models
- Ignore pipeline bubbles
- Forget about activation memory

### The Future

As models continue to grow:
- **More dimensions**: Beyond 5D (mixture-of-depths, mixture-of-resolutions)
- **Better algorithms**: Reducing communication, improving load balancing
- **Hardware co-design**: Custom interconnects for specific parallelism patterns
- **Automated search**: ML to find optimal parallelism configurations

Parallelism has evolved from a niche technique to the **foundation of all large-scale training**. Understanding these techniques is no longer optional—it's essential for anyone working with modern AI.

---

**Next: [Chapter 8 — GPU Generations →](./08_gpu_generations.md)**

---

*Last updated: April 2026*
