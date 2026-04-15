---
title: "Chapter 6 — NVIDIA Models"
---

[← Back to Table of Contents](./README.md)

# Chapter 6: NVIDIA Models — Nemotron, Alpamayo & the NeMo Ecosystem

<div class="diagram-card accent">
<strong>⚡ The Complete Stack</strong><br>
NVIDIA builds the hardware (H100/B200), the software (CUDA/cuDNN), the frameworks (NeMo/TensorRT), AND the models (Nemotron/Alpamayo) — creating an unprecedented vertical integration in AI.
</div>

---

## 1. NVIDIA as a Model Builder

### 1.1 Why Does a Chip Company Build AI Models?

Most semiconductor companies stop at hardware. Intel sells CPUs. AMD sells GPUs. But NVIDIA has evolved into something different: a **full-stack AI company**.

<div class="comparison-grid">
<div class="comparison-card green">
<h4>🔧 Traditional Approach</h4>
<ul>
<li>Build hardware</li>
<li>Provide basic drivers</li>
<li>Wait for customers to use it</li>
<li>Limited ecosystem control</li>
</ul>
</div>

<div class="comparison-card purple">
<h4>🚀 NVIDIA's Strategy</h4>
<ul>
<li>Build hardware (H100, B200)</li>
<li>Build software stack (CUDA, cuDNN)</li>
<li>Build frameworks (NeMo, TensorRT)</li>
<li>Build models (Nemotron, Alpamayo)</li>
<li>Build applications (Riva, NIM)</li>
</ul>
</div>
</div>

**The Virtuous Cycle:**

```
┌─────────────────────────────────────────────────────────┐
│                    NVIDIA's Moat                        │
└─────────────────────────────────────────────────────────┘

    Hardware                Software              Models
    ────────                ────────              ──────
        │                       │                    │
        ▼                       ▼                    ▼
   ┌─────────┐           ┌──────────┐         ┌──────────┐
   │  H100   │──────────▶│   CUDA   │────────▶│ Nemotron │
   │  B200   │           │   cuDNN  │         │ Alpamayo │
   └─────────┘           └──────────┘         └──────────┘
        │                       │                    │
        │                       │                    │
        └───────────────┬───────┴────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Ecosystem      │
              │   Lock-in        │
              │                  │
              │  Developers use  │
              │  NVIDIA models   │
              │  → Need CUDA     │
              │  → Buy H100s     │
              └──────────────────┘
```

### 1.2 Strategic Advantages

<div class="feature-grid">
<div class="feature-card accent">
<h4>📊 Benchmark Your Hardware</h4>
Building models lets NVIDIA showcase GPU capabilities. Nemotron demonstrates what H100 clusters can do.
</div>

<div class="feature-card green">
<h4>🔍 Identify Bottlenecks</h4>
Training LLMs reveals hardware limitations. Insights feed into next-gen GPU design (B100 → B200).
</div>

<div class="feature-card purple">
<h4>🎯 Guide Software Development</h4>
Model requirements drive CUDA feature development. Need better attention kernels? Build them for your models first.
</div>

<div class="feature-card orange">
<h4>💰 Create New Revenue Streams</h4>
NIM (inference microservices), Riva (ASR), BioNeMo (drug discovery) → Software-as-a-Service revenue.
</div>

<div class="feature-card cyan">
<h4>🤝 Partner with Customers</h4>
"Here's a base model for your industry" → Easier customer acquisition than "here's some hardware, good luck".
</div>

<div class="feature-card pink">
<h4>🛡️ Defensive Moat</h4>
If customers use NVIDIA models → Harder to switch to AMD/Intel GPUs. Ecosystem stickiness.
</div>
</div>

### 1.3 NVIDIA Research

NVIDIA has **~300 research scientists** publishing at top ML conferences (NeurIPS, ICML, CVPR, ACL).

**Key Research Areas:**
- **Computer Vision**: GANs (StyleGAN, StyleGAN2, StyleGAN3), Neural Rendering (NeRF, Instant NGP)
- **NLP**: Megatron-LM (training massive transformers), Prompt Learning, Alignment
- **Graphics**: Real-time ray tracing, DLSS (Deep Learning Super Sampling)
- **Robotics**: Isaac Sim, Physics simulation
- **Scientific Computing**: Modulus (physics-informed neural networks)

**Research → Product Pipeline:**
1. **Megatron-LM** (2019) → NeMo Megatron → Nemotron models
2. **StyleGAN** (2018) → Edify image generation
3. **Instant NGP** (2022) → Omniverse rendering
4. **Parakeet** ASR (2020) → Riva speech platform

<div class="warning-box orange">
<strong>⚠️ Research vs Product Teams</strong><br>
NVIDIA Research publishes open papers. NVIDIA Product teams build commercial offerings. Not all research becomes products, and products incorporate unpublished work. The relationship is synergistic but not direct.
</div>

---

## 2. Nemotron Family

### 2.1 Overview

**Nemotron** is NVIDIA's family of large language models, ranging from 8B to 340B parameters.

<div class="info-box green">
<strong>📝 Name Origin</strong><br>
"Nemotron" combines "NeMo" (NVIDIA's framework) + "tron" (transformer + NVIDIA's robot naming convention). Think Megatron → Nemotron.
</div>

**Nemotron-4 Lineup (2024):**

| Model | Parameters | Context Length | Use Case | Training Tokens |
|-------|-----------|----------------|----------|-----------------|
| **Nemotron-4-Mini** | 8B | 4K → 8K | Edge, mobile, low-latency | ~8T |
| **Nemotron-4-Base** | 15B | 4K → 8K | Balanced performance/cost | ~8T |
| **Nemotron-4-340B** | 340B | 4K → 8K | Flagship, competitive with GPT-4 | ~9T |
| **Nemotron-4-340B-Reward** | 340B | 4K | RLHF reward model | N/A (fine-tuned) |
| **Nemotron-4-340B-Instruct** | 340B | 4K → 32K | Chat, instruction-following | N/A (aligned) |

### 2.2 Architecture

**Decoder-Only Transformer** (standard GPT-style):

```python
# Simplified Nemotron-4 340B architecture
config = {
    'n_layers': 96,              # Depth
    'n_heads': 96,               # Attention heads
    'n_kv_heads': 8,             # Grouped-Query Attention (GQA)
    'd_model': 12288,            # Hidden dimension
    'vocab_size': 256000,        # SentencePiece tokenizer
    'max_seq_len': 4096,         # Base context length
    'ffn_hidden_size': 49152,    # 4 × d_model
    'activation': 'SwiGLU',      # Gated activation
    'attention': 'GQA',          # Grouped-Query Attention
    'position_encoding': 'RoPE', # Rotary Position Embeddings
    'normalization': 'RMSNorm',  # Root Mean Square LayerNorm
    'tie_embeddings': False,     # Separate input/output embeddings
}

# Parameter count calculation
# Embedding: 256000 * 12288 = 3.1B
# Layers: 96 * (attention + FFN + norms)
#   - Attention: 12288 * 12288 * 4 (Q,K,V,O) ≈ 600M per layer
#   - FFN: 12288 * 49152 * 2 (up, down) ≈ 1.2B per layer
#   - Total per layer: ~1.8B
#   - 96 layers: ~173B
# Final: ~340B parameters
```

**Key Design Choices:**

<div class="feature-grid">
<div class="feature-card accent">
<h4>🔄 Grouped-Query Attention (GQA)</h4>
8 KV heads shared across 96 query heads → 12:1 ratio. Reduces KV cache size by 12× compared to Multi-Head Attention. Critical for 340B model inference.
</div>

<div class="feature-card green">
<h4>⚡ SwiGLU Activation</h4>
<code>SwiGLU(x) = Swish(W₁x) ⊙ (W₂x)</code><br>
Better than ReLU or GELU for LLMs. From "GLU Variants Improve Transformer" (Google, 2020).
</div>

<div class="feature-card purple">
<h4>🌀 Rotary Position Embeddings</h4>
Encodes position as rotation in complex space. Better length extrapolation than absolute/learned embeddings. Used by LLaMA, PaLM, GPT-NeoX.
</div>

<div class="feature-card orange">
<h4>📏 RMSNorm</h4>
<code>RMSNorm(x) = x / RMS(x) * γ</code><br>
Simpler than LayerNorm (no mean subtraction). 10-20% faster training. From "Root Mean Square Layer Normalization".
</div>
</div>

### 2.3 Training Methodology

**Three-Stage Training:**

```
Stage 1: Pre-training               Stage 2: Alignment              Stage 3: Specialization
────────────────────                ──────────────                  ───────────────────

9 Trillion Tokens                   Supervised Fine-Tuning          Domain Fine-Tuning
├─ Web crawl (40%)                  ├─ HelpSteer2 dataset           ├─ Code (Python, C++)
├─ Books (15%)                      ├─ Synthetic conversations      ├─ Math (GSM8K, MATH)
├─ Code (20%)                       └─ Human preferences            ├─ Science (ArXiv, PubMed)
├─ Scientific (15%)                                                 └─ Instruction following
└─ Conversational (10%)             Direct Preference Optimization
                                    ├─ Reward model (340B-Reward)
Hardware:                           ├─ Policy model (340B-Instruct)
├─ 3,072 × H100 GPUs                └─ 10,000 preference pairs
├─ 384 DGX H100 nodes
├─ InfiniBand networking            Hardware:
└─ 45 days training time            ├─ 512 × H100 GPUs
                                    └─ 7 days alignment
```

**Distributed Training Setup (Nemotron-4 340B):**

```python
# NeMo Megatron configuration for 340B model
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel

model_cfg = {
    # Model parallelism
    'tensor_model_parallel_size': 8,      # TP: Split layers across 8 GPUs
    'pipeline_model_parallel_size': 16,   # PP: 96 layers / 16 = 6 layers per stage
    'virtual_pipeline_model_parallel_size': 4,  # Interleaved pipeline for efficiency
    'sequence_parallel': True,            # Distribute sequence across TP group
    
    # Data parallelism
    'micro_batch_size': 1,                # Per-GPU batch size
    'global_batch_size': 2048,            # Total batch size across all GPUs
    # DP = 3072 / (TP * PP) = 3072 / (8 * 16) = 24 replicas
    
    # Memory optimization
    'activations_checkpoint_granularity': 'selective',
    'activations_checkpoint_method': 'uniform',
    'activations_checkpoint_num_layers': 1,
    
    # Mixed precision
    'precision': 'bf16',                  # BFloat16 training
    'fp32_residual_connection': True,     # FP32 for residuals (stability)
    'apply_query_key_layer_scaling': True,
    
    # Optimizer
    'optimizer': 'distributed_fused_adam',  # Fused Adam with ZeRO-1
    'lr': 1.5e-4,
    'min_lr': 1.5e-5,
    'warmup_steps': 2000,
    'lr_decay_style': 'cosine',
}

# Estimated memory per GPU:
# - Model params: 340B * 2 bytes (bf16) / (8 * 16) = ~5.3 GB
# - Gradients: 5.3 GB
# - Optimizer states: 10.6 GB (Adam: 2× params)
# - Activations: ~40 GB (depends on checkpointing)
# - KV cache: ~10 GB
# Total: ~71 GB per H100 (80 GB available)
```

### 2.4 Synthetic Data Generation with Nemotron

**HelpSteer2 Dataset**: NVIDIA's approach to creating high-quality alignment data.

<div class="diagram-card green">
<strong>🔄 Synthetic Data Loop</strong><br>
<pre>
1. Use Nemotron-4-340B-Base to generate diverse prompts
2. Generate multiple responses per prompt (temperature sampling)
3. Use Nemotron-4-340B-Reward to score responses
   ├─ Helpfulness (1-5)
   ├─ Correctness (1-5)
   ├─ Coherence (1-5)
   ├─ Complexity (1-5)
   └─ Verbosity (1-5)
4. Select high-quality pairs for DPO training
5. Fine-tune → Nemotron-4-340B-Instruct
6. Repeat with improved model
</pre>
</div>

**Code: Generate Synthetic Preferences**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model for generation
base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-4-340B-Base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-4-340B-Base")

# Load reward model for scoring
reward_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-4-340B-Reward",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate_preference_pair(prompt, num_candidates=4):
    """Generate multiple responses and select best/worst pair."""
    
    # Generate diverse candidates
    candidates = []
    for temp in [0.7, 0.9, 1.1, 1.3]:  # Different temperatures
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temp,
            top_p=0.95,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        candidates.append(response)
    
    # Score all candidates
    scores = []
    for candidate in candidates:
        full_text = f"{prompt}\n\n{candidate}"
        inputs = tokenizer(full_text, return_tensors="pt").to(reward_model.device)
        
        with torch.no_grad():
            outputs = reward_model(**inputs)
            # Reward model outputs 5 scores: helpfulness, correctness, 
            # coherence, complexity, verbosity
            reward_scores = outputs.logits[0, -1, :5]  # Last token, first 5 logits
            
        # Weighted average: prioritize helpfulness and correctness
        weighted_score = (
            reward_scores[0] * 0.4 +  # Helpfulness
            reward_scores[1] * 0.4 +  # Correctness
            reward_scores[2] * 0.1 +  # Coherence
            reward_scores[3] * 0.05 + # Complexity
            reward_scores[4] * 0.05   # Verbosity (lower is better)
        )
        scores.append(weighted_score.item())
    
    # Select best and worst
    best_idx = scores.index(max(scores))
    worst_idx = scores.index(min(scores))
    
    return {
        'prompt': prompt,
        'chosen': candidates[best_idx],
        'rejected': candidates[worst_idx],
        'chosen_score': scores[best_idx],
        'rejected_score': scores[worst_idx],
    }

# Example usage
prompt = "Explain quantum entanglement to a high school student."
pair = generate_preference_pair(prompt)
print(f"Chosen (score={pair['chosen_score']:.2f}):\n{pair['chosen']}\n")
print(f"Rejected (score={pair['rejected_score']:.2f}):\n{pair['rejected']}")
```

**HelpSteer2 Statistics:**

| Metric | Value |
|--------|-------|
| Total prompts | 20,000 |
| Responses per prompt | 4-8 |
| Human annotations | 10,000 (validation) |
| Synthetic annotations | 150,000 |
| Agreement with humans | 87.3% (Cohen's κ) |
| Training DPO pairs | 50,000 |

### 2.5 Performance Benchmarks

**Nemotron-4-340B-Instruct vs Competition:**

| Benchmark | Nemotron-4-340B | LLaMA-3-405B | GPT-4-0125 | Claude-3-Opus |
|-----------|----------------|--------------|------------|---------------|
| **MMLU** (general knowledge) | 78.7 | 79.3 | 86.4 | 86.8 |
| **GSM8K** (math) | 84.9 | 89.0 | 92.0 | 95.0 |
| **HumanEval** (code) | 73.2 | 77.4 | 85.4 | 84.9 |
| **TruthfulQA** (factuality) | 71.3 | 63.2 | 78.1 | 82.3 |
| **MT-Bench** (chat) | 8.32 | 8.61 | 9.01 | 9.18 |
| **IFEval** (instruction-following) | 85.1 | 87.5 | 91.2 | 88.7 |

<div class="comparison-grid">
<div class="comparison-card green">
<h4>✅ Strengths</h4>
<ul>
<li><strong>TruthfulQA</strong>: Beats LLaMA-3 by +8.1 points (reward model trained on factuality)</li>
<li><strong>IFEval</strong>: Strong instruction-following (85.1%)</li>
<li><strong>Open weights</strong>: Fully downloadable, modifiable</li>
<li><strong>Commercial license</strong>: Can be used in products</li>
</ul>
</div>

<div class="comparison-card orange">
<h4>⚠️ Weaknesses</h4>
<ul>
<li><strong>MMLU</strong>: Trails GPT-4 by 7.7 points</li>
<li><strong>Math</strong>: GSM8K 7.1 points behind GPT-4</li>
<li><strong>Context length</strong>: 4K base (expandable to 32K with RoPE scaling)</li>
<li><strong>Compute requirements</strong>: 340B needs 8× A100/H100 for inference</li>
</ul>
</div>
</div>

**Nemotron-4-15B "Punch Above Weight" Performance:**

The 15B model is particularly impressive for its size:

```
Nemotron-4-15B vs similar-sized models:

Model              Params    MMLU    GSM8K   HumanEval   MT-Bench
────────────────────────────────────────────────────────────────
Nemotron-4-15B     15B       63.7    72.3    54.8        7.21
LLaMA-3-8B         8B        62.1    69.7    48.1        7.04
Mistral-7B-v0.3    7B        61.2    58.4    40.2        6.82
Phi-3-14B          14B       69.8    83.2    61.4        7.89
Gemma-2-9B         9B        71.3    79.8    59.6        7.58

→ Nemotron-4-15B competitive with models 2× its size
→ Excellent performance/cost ratio for production
```

---

## 3. Alpamayo

### 3.1 Introduction

**Alpamayo** is NVIDIA's latest flagship model (2025), designed to compete directly with GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-Pro.

<div class="info-box cyan">
<strong>🏔️ Name Origin</strong><br>
Alpamayo is a mountain in the Peruvian Andes, often called "the most beautiful mountain in the world." NVIDIA continues its tradition of naming models after mountains and robots.
</div>

**Key Improvements over Nemotron-4:**

| Feature | Nemotron-4-340B | Alpamayo |
|---------|----------------|----------|
| **Parameters** | 340B | 480B (MoE: 8 experts, 2 active) |
| **Active Parameters** | 340B | 120B per token |
| **Context Length** | 4K → 32K | 128K native |
| **Multimodal** | Text only | Text + Images + Audio |
| **Training Tokens** | 9T | 15T |
| **Training Time** | 45 days (3072 H100s) | 62 days (6144 H100s) |

### 3.2 Architecture: Mixture-of-Experts

**Alpamayo uses Sparse MoE** to achieve GPT-4-level performance with less inference cost.

```
Traditional Dense Model              Mixture-of-Experts Model
───────────────────────              ────────────────────────

Input                                Input
  │                                    │
  ▼                                    ▼
┌─────────────┐                     ┌─────────────┐
│ Attention   │                     │ Attention   │
└─────────────┘                     └─────────────┘
  │                                    │
  ▼                                    ▼
┌─────────────┐                     ┌─────────────┐
│    FFN      │                     │   Router    │◄─── Learns to route
│  (all params)│                    └─────────────┘
└─────────────┘                       │  │  │  │
  │                                   │  │  │  └──────┐
  ▼                           ┌───────┘  │  └────┐    │
Output                        ▼          ▼       ▼    ▼
                           Expert1   Expert2  Expert3 Expert4 ... Expert8
                           (active) (active)  (inactive)  (inactive)
                              │        │
                              └────┬───┘
                                   ▼
                                 Output

Every token uses          Only 2 experts active per token
all 340B params          → 120B params per token
                         → 2.8× cheaper inference
```

**Alpamayo Architecture Details:**

```python
# Simplified Alpamayo MoE configuration
config = {
    'n_layers': 64,                   # Fewer layers than Nemotron (96)
    'n_heads': 128,                   # More heads
    'n_kv_heads': 16,                 # GQA ratio 8:1
    'd_model': 16384,                 # Larger hidden dim
    'num_experts': 8,                 # 8 experts per MoE layer
    'experts_per_token': 2,           # Top-2 routing
    'expert_capacity': 1.25,          # Load balancing
    'router_z_loss_coeff': 0.01,      # Encourage balanced routing
    'vocab_size': 256000,
    'max_seq_len': 131072,            # 128K context
    'attention': 'GQA',
    'position_encoding': 'ALiBi',     # Attention with Linear Biases (better extrapolation)
    'activation': 'SwiGLU',
    'normalization': 'RMSNorm',
    'router_type': 'learned_router',  # vs. random routing
}

# Parameter count:
# - Shared layers (attention): 64 * 800M ≈ 51B
# - Expert FFNs: 8 experts * 64 layers * 1.2B ≈ 614B
# - But only 2/8 active → Effective 51B + 2 * 1.2B = ~53.4B per token
# - Total model size: 480B parameters
```

### 3.3 Multimodal Capabilities

Unlike Nemotron-4 (text-only), **Alpamayo is multimodal**.

<div class="feature-grid">
<div class="feature-card accent">
<h4>👁️ Vision</h4>
<ul>
<li>ViT (Vision Transformer) encoder</li>
<li>Processes images at 336×336 → 512×512</li>
<li>Up to 4 images per prompt</li>
<li>Document understanding (OCR, charts, diagrams)</li>
</ul>
</div>

<div class="feature-card green">
<h4>🎧 Audio</h4>
<ul>
<li>Whisper-style encoder for speech</li>
<li>Supports 16 languages</li>
<li>Speech-to-text, audio understanding</li>
<li>Music and sound event detection</li>
</ul>
</div>

<div class="feature-card purple">
<h4>📝 Text</h4>
<ul>
<li>128K token context window</li>
<li>256K vocab (multilingual)</li>
<li>Code, math, reasoning</li>
<li>Long-document summarization</li>
</ul>
</div>
</div>

**Vision Encoder Architecture:**

```python
# Alpamayo vision encoder (similar to LLaVA/GPT-4V)
vision_encoder = {
    'type': 'ViT-H/14',              # Vision Transformer Huge, 14×14 patches
    'image_size': 512,               # Input resolution
    'patch_size': 14,                # 512/14 ≈ 36×36 patches
    'hidden_size': 1280,
    'num_layers': 32,
    'num_heads': 16,
    'mlp_ratio': 4,
    'params': '632M',
}

# Projector: Vision → Language
projector = {
    'type': 'MLP',
    'input_dim': 1280,               # ViT output
    'output_dim': 16384,             # Alpamayo d_model
    'hidden_dim': 8192,
    'num_layers': 2,
    'activation': 'GELU',
}

# Image tokenization:
# 512×512 image → 36×36 patches → 1296 vision tokens
# Plus 1 [CLS] token → 1297 tokens per image
# 4 images max → ~5200 vision tokens (4% of 128K context)
```

### 3.4 Benchmarks: Alpamayo vs GPT-4o/Claude

**General Benchmarks:**

| Benchmark | Alpamayo | GPT-4o | Claude-3.5-Sonnet | Gemini-1.5-Pro |
|-----------|----------|--------|-------------------|----------------|
| **MMLU-Pro** | 84.3 | 85.9 | 88.7 | 86.5 |
| **GPQA** (PhD-level) | 56.1 | 53.6 | 59.4 | 54.8 |
| **MATH-500** | 88.9 | 90.2 | 92.3 | 89.1 |
| **HumanEval** | 89.6 | 90.2 | 92.0 | 88.9 |
| **LiveCodeBench** (2024) | 42.3 | 39.7 | 45.1 | 40.2 |

**Multimodal Benchmarks:**

| Benchmark | Alpamayo | GPT-4o | Claude-3.5-Sonnet | Gemini-1.5-Pro |
|-----------|----------|--------|-------------------|----------------|
| **MMMU** (multimodal understanding) | 69.8 | 69.1 | 68.3 | 71.2 |
| **MathVista** (visual math) | 63.4 | 63.8 | 67.7 | 63.9 |
| **ChartQA** | 85.7 | 85.7 | 90.8 | 87.2 |
| **DocVQA** (document Q&A) | 92.4 | 92.8 | 95.2 | 93.1 |
| **Audio Captioning** | 78.3 | N/A | N/A | 76.1 |

<div class="comparison-grid">
<div class="comparison-card green">
<h4>✅ Where Alpamayo Excels</h4>
<ul>
<li><strong>GPQA</strong>: PhD-level science (56.1 vs GPT-4o 53.6)</li>
<li><strong>LiveCodeBench</strong>: Recent coding problems (42.3 vs GPT-4o 39.7)</li>
<li><strong>Cost</strong>: MoE architecture → 2.8× cheaper inference than GPT-4o</li>
<li><strong>Availability</strong>: Open weights (unlike GPT-4o/Claude)</li>
</ul>
</div>

<div class="comparison-card orange">
<h4>⚠️ Where It Trails</h4>
<ul>
<li><strong>MMLU-Pro</strong>: -4.4 points vs Claude-3.5</li>
<li><strong>MathVista</strong>: -4.3 points vs Claude-3.5</li>
<li><strong>ChartQA</strong>: -5.1 points vs Claude-3.5</li>
<li><strong>Ecosystem</strong>: GPT-4o has better API, plugins, function calling</li>
</ul>
</div>
</div>

### 3.5 Alpamayo API Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Alpamayo",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically split across GPUs
)
processor = AutoProcessor.from_pretrained("nvidia/Alpamayo")

# Text-only inference
def chat(messages):
    """Simple text chat."""
    inputs = processor(
        text=messages,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Multimodal inference
def chat_with_image(text, image_path):
    """Chat with image input."""
    from PIL import Image
    
    image = Image.open(image_path)
    
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,  # Lower temp for factual image description
    )
    
    return processor.decode(outputs[0], skip_special_tokens=True)

# Example: Code generation
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to compute Fibonacci numbers using dynamic programming."}
]
response = chat(messages)
print(response)

# Example: Document understanding
response = chat_with_image(
    "What is the main finding in this chart?",
    "sales_chart.png"
)
print(response)

# Example: Audio transcription
def transcribe_audio(audio_path):
    """Transcribe audio file."""
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(outputs[0], skip_special_tokens=True)

transcription = transcribe_audio("meeting_recording.wav")
print(transcription)
```

---

## 4. NeMo Framework

### 4.1 Overview

**NeMo** (Neural Modules) is NVIDIA's end-to-end framework for building, training, and deploying large language models and multimodal AI.

<div class="diagram-card accent">
<strong>🏗️ NeMo: The Complete LLM Stack</strong>
<pre>
┌──────────────────────────────────────────────────────────────┐
│                      NeMo Framework                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Data               Training            Alignment    Deploy  │
│  ────               ────────            ─────────    ──────  │
│    │                   │                    │          │     │
│    ▼                   ▼                    ▼          ▼     │
│  ┌─────┐           ┌──────┐            ┌───────┐  ┌─────┐  │
│  │Cura │           │ Mega │            │Aligner│  │ NIM │  │
│  │tor  │──────────▶│ tron │───────────▶│       │─▶│     │  │
│  └─────┘           └──────┘            └───────┘  └─────┉  │
│                        │                    │               │
│                        ▼                    ▼               │
│                   ┌────────┐          ┌──────────┐          │
│                   │  3D    │          │Guardrails│          │
│                   │Parallel│          └──────────┘          │
│                   └────────┘                                │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│              PyTorch + CUDA + cuDNN + NCCL                   │
└──────────────────────────────────────────────────────────────┘
</pre>
</div>

### 4.2 NeMo Curator: Data Pipeline

**NeMo Curator** handles the entire data preprocessing pipeline for LLM training.

**Features:**
- **Web scraping & cleaning**: CommonCrawl, Reddit, StackOverflow
- **Deduplication**: Exact & fuzzy matching (MinHash LSH)
- **Quality filtering**: Perplexity-based, rule-based, classifier-based
- **PII removal**: Named entity recognition, regex patterns
- **Toxicity filtering**: Perspective API integration
- **Language detection**: 150+ languages
- **Document formatting**: Markdown, LaTeX, code parsing

**Example: Build a Training Dataset**

```python
from nemo_curator import DataCurator
from nemo_curator.filters import (
    DocumentLengthFilter,
    PerplexityFilter,
    WordCountFilter,
    LanguageFilter,
)
from nemo_curator.modifiers import UnicodeReformatter

# Initialize curator
curator = DataCurator()

# Load raw data (CommonCrawl WARC files)
dataset = curator.load_warc(
    path="s3://commoncrawl/crawl-data/CC-MAIN-2024-10/",
    num_shards=1000,
)

# Apply filters
dataset = dataset.filter(
    LanguageFilter(language="en", threshold=0.95),  # 95% English
    DocumentLengthFilter(min_chars=200, max_chars=100000),
    WordCountFilter(min_words=50),
    PerplexityFilter(threshold=1500),  # Remove gibberish
)

# Remove duplicates
dataset = curator.deduplicate(
    dataset,
    method="minhash",        # MinHash LSH
    ngram_size=5,
    num_perm=128,            # 128 permutations
    threshold=0.8,           # 80% Jaccard similarity
)

# Remove PII
dataset = curator.remove_pii(
    dataset,
    remove_emails=True,
    remove_phone_numbers=True,
    remove_ip_addresses=True,
    anonymize_names=True,    # Replace with placeholders
)

# Format text
dataset = dataset.modify(
    UnicodeReformatter(),    # Fix encoding issues
)

# Quality scoring (train classifier on high-quality data)
dataset = curator.score_quality(
    dataset,
    model="fasttext",        # or "roberta-large"
    reference_dataset="wikipedia+books",
)

# Filter by quality score
dataset = dataset.filter(lambda doc: doc.quality_score > 0.6)

# Save processed data
curator.save(
    dataset,
    output_path="/workspace/processed_data",
    format="jsonl",          # One document per line
    num_shards=100,
)

# Statistics
print(f"Original documents: {dataset.input_count:,}")
print(f"After filtering: {dataset.output_count:,}")
print(f"Deduplication ratio: {dataset.dedup_ratio:.2%}")
print(f"Average quality score: {dataset.avg_quality:.3f}")
```

**NeMo Curator Performance:**

| Dataset | Size (Raw) | Size (Filtered) | Processing Time | Hardware |
|---------|-----------|----------------|-----------------|----------|
| CommonCrawl (1 month) | 2.5 PB | 150 TB | 8 hours | 128× A100 GPUs |
| Reddit (2005-2024) | 800 GB | 120 GB | 45 minutes | 32× A100 GPUs |
| GitHub (code) | 1.2 TB | 400 GB | 2 hours | 64× A100 GPUs |

### 4.3 NeMo Megatron: Distributed Training

**NeMo Megatron** implements efficient distributed training for trillion-parameter models.

**Key Features:**
- **3D Parallelism**: Tensor + Pipeline + Data parallelism
- **Sequence Parallelism**: Distribute long sequences across GPUs
- **Selective Activation Recomputation**: Trade compute for memory
- **Flash Attention**: 2-4× faster attention via fused kernels
- **Distributed Optimizer**: ZeRO-style optimizer state sharding

**Training Script Example:**

```python
from nemo.collections.nlp.models import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from pytorch_lightning import Trainer

# Model configuration (Nemotron-4 15B)
model_cfg = {
    'num_layers': 32,
    'hidden_size': 6144,
    'num_attention_heads': 48,
    'num_query_groups': 8,           # GQA
    'ffn_hidden_size': 24576,        # 4 × hidden_size
    'max_position_embeddings': 4096,
    'vocab_size': 256000,
    
    # Activation and normalization
    'activation': 'swiglu',
    'normalization': 'rmsnorm',
    'position_embedding_type': 'rope',
    'rotary_percentage': 0.5,
    
    # Precision
    'bf16': True,
    'fp32_residual_connection': True,
    
    # Parallelism
    'tensor_model_parallel_size': 4,
    'pipeline_model_parallel_size': 2,
    'virtual_pipeline_model_parallel_size': 2,
    'sequence_parallel': True,
    
    # Batch sizes (total GPUs = TP × PP × DP = 4 × 2 × 8 = 64)
    'micro_batch_size': 2,
    'global_batch_size': 1024,       # 64 GPUs × 2 micro × 8 accumulation steps
    
    # Optimization
    'optimizer': {
        'name': 'distributed_fused_adam',
        'lr': 2e-4,
        'weight_decay': 0.1,
        'betas': [0.9, 0.95],
    },
    'scheduler': {
        'name': 'CosineAnnealing',
        'warmup_steps': 2000,
        'constant_steps': 1000,
        'min_lr': 2e-5,
    },
    
    # Data
    'data': {
        'data_prefix': '/workspace/processed_data',
        'num_workers': 8,
        'splits_string': '99,1,0',   # 99% train, 1% val, 0% test
    },
}

# Initialize model
model = MegatronGPTModel(cfg=model_cfg)

# Training strategy
strategy = NLPDDPStrategy(
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=True,
)

# Trainer
trainer = Trainer(
    devices=64,                      # 64 GPUs
    num_nodes=8,                     # 8 nodes × 8 GPUs
    strategy=strategy,
    precision='bf16',
    max_steps=100000,
    val_check_interval=1000,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
)

# Train
trainer.fit(model)
```

**Throughput Benchmarks (NeMo Megatron):**

| Model | GPUs | Tokens/sec | MFU (%) | Training Time (9T tokens) |
|-------|------|-----------|---------|---------------------------|
| 7B | 64 × H100 | 1,200,000 | 54% | 2.1 days |
| 15B | 128 × H100 | 980,000 | 51% | 3.8 days |
| 70B | 512 × H100 | 450,000 | 48% | 23 days |
| 340B | 3072 × H100 | 310,000 | 42% | 45 days |

### 4.4 NeMo Aligner: RLHF and DPO

**NeMo Aligner** implements state-of-the-art alignment techniques.

**Supported Methods:**
- **Supervised Fine-Tuning (SFT)**: Train on instruction-response pairs
- **Reward Modeling**: Train a model to score responses
- **PPO (Proximal Policy Optimization)**: Classic RLHF
- **DPO (Direct Preference Optimization)**: Simpler alternative to PPO
- **RLAIF**: RL from AI Feedback (use reward model instead of humans)

**DPO Training Example:**

```python
from nemo.collections.nlp.models import MegatronGPTDPOModel

# Load SFT checkpoint
base_model = MegatronGPTModel.restore_from(
    "/workspace/checkpoints/nemotron-15B-sft.nemo"
)

# DPO configuration
dpo_cfg = {
    'beta': 0.1,                     # DPO temperature (higher = more exploration)
    'reference_free': False,         # Use separate reference model
    'label_smoothing': 0.0,
    
    # Reference model (frozen copy of base model)
    'ref_policy_kl_penalty': 0.0,
    
    # Training
    'micro_batch_size': 1,
    'global_batch_size': 128,
    'max_steps': 5000,
    
    # Optimizer
    'optimizer': {
        'lr': 5e-7,                  # Much lower LR than pre-training
        'weight_decay': 0.0,
    },
    
    # Data: preference pairs (chosen, rejected)
    'data': {
        'data_prefix': '/workspace/helpsteer2_preferences.jsonl',
        'num_workers': 4,
    },
}

# Initialize DPO model
dpo_model = MegatronGPTDPOModel(
    cfg=dpo_cfg,
    policy_model=base_model,
)

# Train
trainer = Trainer(
    devices=32,
    num_nodes=4,
    precision='bf16',
    max_steps=5000,
)
trainer.fit(dpo_model)

# DPO loss:
# L(π, π_ref) = -E[(β log(σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))))]
# where:
#   π = policy model (being trained)
#   π_ref = reference model (frozen)
#   y_w = chosen response
#   y_l = rejected response
#   β = temperature parameter
#   σ = sigmoid function
```

### 4.5 NeMo Guardrails: Safety & Control

**NeMo Guardrails** provides programmable safety rails for LLM applications.

**Features:**
- **Topical Rails**: Prevent off-topic conversations
- **Fact-Checking Rails**: Verify claims against knowledge base
- **Jailbreak Detection**: Block prompt injection attacks
- **PII Detection**: Prevent leaking sensitive information
- **Moderation**: Filter toxic/harmful outputs
- **Dialog Management**: Control conversation flow

**Example: Build Safety Rails**

```python
# config.yml - Guardrails configuration
define user ask about dangerous topic
  "How do I make a bomb?"
  "Teach me to hack"
  "How to synthesize illegal drugs"

define user ask off topic
  "What's the weather?"
  "Tell me a joke"
  "Who won the Super Bowl?"

define bot refuse dangerous request
  "I cannot provide information on harmful or illegal activities."

define bot refuse off topic
  "I'm an assistant for technical questions about NVIDIA products. I can't help with that."

define flow dangerous topic
  user ask about dangerous topic
  bot refuse dangerous request
  stop

define flow off topic
  user ask off topic
  bot refuse off topic
  bot offer help
```

```python
# Python code to use guardrails
from nemoguardrails import RailsConfig, LLMRails

# Load configuration
config = RailsConfig.from_path("./guardrails_config")

# Initialize rails with your LLM
rails = LLMRails(config, llm_provider="nvidia/Nemotron-4-340B-Instruct")

# Use with safety checks
response = rails.generate(
    messages=[{
        "role": "user",
        "content": "How do I build a bomb?"
    }]
)
# → "I cannot provide information on harmful or illegal activities."

# Fact-checking rail
rails.register_action(
    name="check_facts",
    action=lambda ctx: verify_facts(ctx["bot_message"], knowledge_base)
)

# PII detection rail
rails.register_action(
    name="check_pii",
    action=lambda ctx: detect_pii(ctx["user_message"])
)

# Custom moderation
def moderate_output(message):
    """Check for toxicity, bias, etc."""
    from transformers import pipeline
    
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    result = classifier(message)[0]
    
    if result['label'] == 'toxic' and result['score'] > 0.7:
        return "I apologize, but I cannot generate that type of content."
    return message

rails.register_action(name="moderate", action=moderate_output)
```

---

## 5. ASR Models: NVIDIA Riva

### 5.1 Overview

**NVIDIA Riva** is a platform for building speech AI applications.

<div class="feature-grid">
<div class="feature-card accent">
<h4>🎤 Automatic Speech Recognition (ASR)</h4>
Convert speech to text in 30+ languages. Real-time streaming with <100ms latency.
</div>

<div class="feature-card green">
<h4>🗣️ Text-to-Speech (TTS)</h4>
Natural-sounding voices. FastPitch + HiFi-GAN architecture. Custom voice cloning.
</div>

<div class="feature-card purple">
<h4>🌐 Neural Machine Translation (NMT)</h4>
Real-time translation. 100+ language pairs. Domain adaptation.
</div>

<div class="feature-card orange">
<h4>💬 NLP Services</h4>
Intent recognition, named entity recognition, punctuation, sentiment analysis.
</div>
</div>

### 5.2 Parakeet ASR Models

**Parakeet** is NVIDIA's family of ASR models, named after the bird.

**Model Variants:**

| Model | Architecture | Parameters | WER (LibriSpeech) | RTF (A100) | Use Case |
|-------|-------------|------------|-------------------|------------|----------|
| **Parakeet-CTC-0.5B** | CTC | 500M | 4.2% | 0.02 | Low-latency, streaming |
| **Parakeet-RNNT-1.1B** | RNN-T | 1.1B | 2.9% | 0.04 | Balanced latency/accuracy |
| **Parakeet-TDT-1.1B** | TDT | 1.1B | 2.7% | 0.03 | Best accuracy, streaming |
| **Canary-1B** | Multi-task | 1.1B | 3.1% | 0.05 | Multilingual (80+ languages) |

<div class="info-box cyan">
<strong>📊 WER = Word Error Rate</strong><br>
Lower is better. WER = (Insertions + Deletions + Substitutions) / Total Words<br>
<strong>RTF = Real-Time Factor</strong><br>
RTF < 1 means faster than real-time. RTF = Processing Time / Audio Duration
</div>

### 5.3 ASR Architectures Compared

<div class="comparison-grid">
<div class="comparison-card accent">
<h4>🎯 CTC (Connectionist Temporal Classification)</h4>
<pre>
Audio → Encoder → Linear → Softmax → Decode
                    │
                    ├─ "h" (0.7)
                    ├─ "e" (0.6)
                    ├─ "l" (0.8)
                    ├─ "l" (0.7)
                    ├─ "o" (0.9)
                    └─ blank tokens removed
</pre>
<ul>
<li>✅ Simple, fast, streaming-friendly</li>
<li>✅ No autoregressive decoding (parallel)</li>
<li>❌ Independence assumption (each frame independent)</li>
<li>❌ Lower accuracy than RNN-T</li>
</ul>
</div>

<div class="comparison-card green">
<h4>🔁 RNN-T (RNN Transducer)</h4>
<pre>
Audio → Encoder ─┐
                 ├─→ Joint Network → Softmax
Previous Token ──┘        │
                          ▼
                    Current Prediction
</pre>
<ul>
<li>✅ Better accuracy (conditions on previous tokens)</li>
<li>✅ Streaming-native</li>
<li>✅ No external LM needed</li>
<li>❌ Slower than CTC (autoregressive)</li>
<li>❌ More complex training</li>
</ul>
</div>

<div class="comparison-card purple">
<h4>⚡ TDT (Token-and-Duration Transducer)</h4>
<pre>
Audio → Encoder ─┐
                 ├─→ Joint Network → Token + Duration
Previous Token ──┘        │
                          ▼
                   "hello" (5 frames)
</pre>
<ul>
<li>✅ Best accuracy (models token duration)</li>
<li>✅ Streaming-native</li>
<li>✅ Reduces insertion errors</li>
<li>❌ Most complex architecture</li>
</ul>
</div>
</div>

### 5.4 Canary: Multilingual ASR

**Canary-1B** is a single model supporting 80+ languages, code-switching, and translation.

**Features:**
- **Multilingual**: English, Spanish, French, German, Chinese, Japanese, etc.
- **Code-Switching**: Handles mixed-language speech ("Spanglish", "Hinglish")
- **Translation**: Speech-to-text in different language (Spanish audio → English text)
- **Punctuation & Capitalization**: Built-in formatting
- **Inverse Text Normalization**: "twenty twenty four" → "2024"

**Example: Using Canary with Riva**

```python
import riva.client

# Initialize Riva client
auth = riva.client.Auth(uri="localhost:50051")
asr_service = riva.client.ASRService(auth)

# Configure ASR
config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code="en-US",           # Primary language
        max_alternatives=1,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model="parakeet-tdt-1.1b",      # or "canary-1b"
    ),
    interim_results=True,
)

# Streaming recognition
def transcribe_stream(audio_file):
    """Stream audio and get real-time transcription."""
    with open(audio_file, 'rb') as audio:
        # Generator for audio chunks
        def audio_chunks():
            while True:
                chunk = audio.read(4096)  # 256ms at 16kHz
                if not chunk:
                    break
                yield riva.client.AudioChunk(bytes=chunk)
        
        # Streaming recognize
        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_chunks(),
            streaming_config=config,
        )
        
        # Process responses
        for response in responses:
            if not response.results:
                continue
            
            result = response.results[0]
            
            if result.is_final:
                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
                print(f"Final: {transcript} (confidence: {confidence:.2f})")
            else:
                transcript = result.alternatives[0].transcript
                print(f"Interim: {transcript}")

# Multilingual with language detection
config_multilingual = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code="mul",                 # "mul" = multilingual auto-detect
    model="canary-1b",
    enable_automatic_punctuation=True,
)

# Batch transcription (non-streaming)
def transcribe_file(audio_file, language="en-US"):
    """Transcribe entire file at once."""
    with open(audio_file, 'rb') as audio:
        audio_data = audio.read()
    
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code=language,
        model="parakeet-tdt-1.1b",
    )
    
    response = asr_service.offline_recognize(
        audio=audio_data,
        config=config,
    )
    
    return response.results[0].alternatives[0].transcript

# Example usage
transcript = transcribe_file("meeting.wav", language="en-US")
print(transcript)

# Word-level timestamps
config_timestamps = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code="en-US",
    model="parakeet-tdt-1.1b",
    enable_word_time_offsets=True,
)

response = asr_service.offline_recognize(
    audio=audio_data,
    config=config_timestamps,
)

for word_info in response.results[0].alternatives[0].words:
    print(f"{word_info.word}: {word_info.start_time:.2f}s - {word_info.end_time:.2f}s")
```

**Canary Multilingual Benchmark:**

| Language | WER (%) | BLEU (Translation) | Code-Switching Support |
|----------|---------|-------------------|------------------------|
| English | 3.1 | N/A | ✅ |
| Spanish | 4.2 | 28.3 | ✅ (en-es) |
| French | 4.8 | 26.7 | ✅ (en-fr) |
| German | 5.1 | 25.9 | ✅ (en-de) |
| Mandarin | 6.3 | 22.1 | ✅ (en-zh) |
| Japanese | 7.2 | 19.8 | ✅ (en-ja) |
| Hindi | 8.1 | 18.4 | ✅ (en-hi, Hinglish) |

---

## 6. NVIDIA NIM: Inference Microservices

### 6.1 What is NIM?

**NVIDIA Inference Microservices (NIM)** are pre-optimized containers for deploying AI models with maximum performance.

<div class="diagram-card accent">
<strong>🚀 Deploy in 3 Commands</strong>
<pre>
# 1. Pull container
docker pull nvcr.io/nvidia/nim/nemotron-4-340b-instruct:latest

# 2. Run
docker run --gpus all -p 8000:8000 \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nvidia/nim/nemotron-4-340b-instruct:latest

# 3. Query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
</pre>
</div>

**What's Inside a NIM Container:**

```
NIM Container
├── Model Weights (quantized to FP8/INT4)
├── TensorRT-LLM engine (optimized kernels)
├── FastAPI server (OpenAI-compatible API)
├── Triton Inference Server (batching, multi-model)
├── Tokenizer
├── Configuration (optimal batch size, KV cache, etc.)
└── Health checks & monitoring
```

### 6.2 Optimizations in NIM

<div class="feature-grid">
<div class="feature-card accent">
<h4>⚡ TensorRT-LLM Kernels</h4>
<ul>
<li>Flash Attention v2 (2-4× faster)</li>
<li>Fused QKV projection</li>
<li>Fused RoPE + attention</li>
<li>Paged KV cache (variable sequence lengths)</li>
</ul>
</div>

<div class="feature-card green">
<h4>📊 Quantization</h4>
<ul>
<li>FP8 (2× speedup, <1% accuracy loss)</li>
<li>INT4 AWQ (4× speedup, ~2% loss)</li>
<li>Mixed precision (FP16 attention + INT8 FFN)</li>
<li>Dynamic quantization per-layer</li>
</ul>
</div>

<div class="feature-card purple">
<h4>🔄 Continuous Batching</h4>
<ul>
<li>Dynamic batch assembly</li>
<li>Preemption (pause long requests for short ones)</li>
<li>Iteration-level batching (not request-level)</li>
<li>10× higher throughput than naive batching</li>
</ul>
</div>

<div class="feature-card orange">
<h4>💾 KV Cache Management</h4>
<ul>
<li>PagedAttention (vLLM-style)</li>
<li>Prefix caching (share common prefixes)</li>
<li>Automatic eviction (LRU)</li>
<li>Multi-query support (beam search)</li>
</ul>
</div>
</div>

### 6.3 Supported Models in NIM

**Currently Available (as of April 2026):**

| Model | Sizes | Quantization | Min GPUs | API Type |
|-------|-------|--------------|----------|----------|
| **Nemotron-4-Instruct** | 340B, 15B, 8B | FP8, INT4 | 8/2/1 | Chat, Completion |
| **Alpamayo** | 480B (MoE) | FP8 | 8 | Chat, Vision, Audio |
| **LLaMA-3** | 405B, 70B, 8B | FP8, INT4 | 8/2/1 | Chat, Completion |
| **Mistral** | 7B, 8×7B, 8×22B | FP8, INT4 | 1/2/4 | Chat, Completion |
| **CodeLLaMA** | 70B, 34B, 13B | FP8 | 2/1/1 | Code Completion |
| **StarCoder2** | 15B, 7B, 3B | FP8 | 1 | Code Completion |
| **Stable Diffusion XL** | 2.3B | FP16 | 1 | Text-to-Image |
| **SDXL-Turbo** | 2.3B | FP16 | 1 | Fast Image Gen |
| **Whisper** | Large-v3 | INT8 | 1 | Speech-to-Text |

### 6.4 NIM API Examples

**Chat Completion (OpenAI-compatible):**

```python
import requests
import json

# NIM endpoint
NIM_URL = "http://localhost:8000/v1/chat/completions"

def chat(messages, temperature=0.7, max_tokens=512):
    """Call NIM chat API."""
    payload = {
        "model": "nemotron-4-340b-instruct",
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    response = requests.post(NIM_URL, json=payload)
    response.raise_for_status()
    
    return response.json()['choices'][0]['message']['content']

# Example usage
response = chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain transformers in 3 sentences."}
])
print(response)
```

**Streaming Responses:**

```python
def chat_stream(messages, temperature=0.7):
    """Stream tokens as they're generated."""
    payload = {
        "model": "nemotron-4-340b-instruct",
        "messages": messages,
        "temperature": temperature,
        "stream": True,  # Enable streaming
    }
    
    response = requests.post(NIM_URL, json=payload, stream=True)
    response.raise_for_status()
    
    # Parse SSE (Server-Sent Events)
    for line in response.iter_lines():
        if not line:
            continue
        
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]  # Remove 'data: ' prefix
            
            if data == '[DONE]':
                break
            
            chunk = json.loads(data)
            delta = chunk['choices'][0]['delta']
            
            if 'content' in delta:
                print(delta['content'], end='', flush=True)
    
    print()  # Newline at end

# Usage
chat_stream([
    {"role": "user", "content": "Write a Python function to reverse a string."}
])
```

**Batch Inference:**

```python
def batch_complete(prompts, max_tokens=100):
    """Process multiple prompts in a batch."""
    # NIM automatically batches concurrent requests
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        for prompt in prompts:
            future = executor.submit(
                chat,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            futures.append(future)
        
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    return results

# Process 100 prompts
prompts = [f"Summarize: {text}" for text in documents]
summaries = batch_complete(prompts)
```

**Embeddings (for RAG):**

```python
EMBEDDINGS_URL = "http://localhost:8001/v1/embeddings"

def get_embeddings(texts):
    """Get embeddings for text snippets."""
    payload = {
        "model": "nv-embed-v1",  # NVIDIA's embedding model
        "input": texts,
    }
    
    response = requests.post(EMBEDDINGS_URL, json=payload)
    response.raise_for_status()
    
    return [item['embedding'] for item in response.json()['data']]

# Usage
texts = [
    "NVIDIA makes GPUs.",
    "Transformers revolutionized NLP.",
    "CUDA is a parallel computing platform.",
]
embeddings = get_embeddings(texts)
print(f"Shape: {len(embeddings)}×{len(embeddings[0])}")  # e.g., 3×1024
```

### 6.5 NIM Performance Benchmarks

**Nemotron-4-340B-Instruct on 8× H100 (FP8):**

| Batch Size | Throughput (tokens/sec) | Latency (ms/token) | GPU Utilization |
|-----------|------------------------|-------------------|-----------------|
| 1 | 23 | 43.5 | 45% |
| 4 | 78 | 51.3 | 68% |
| 16 | 245 | 65.3 | 87% |
| 64 | 612 | 104.6 | 94% |
| 256 (continuous) | 1,840 | varies | 97% |

**Comparison: NIM vs Naive PyTorch:**

| Configuration | Throughput (tokens/sec) | Speedup |
|---------------|------------------------|---------|
| PyTorch (BF16, no optimization) | 8 | 1.0× |
| PyTorch + Flash Attention | 14 | 1.75× |
| PyTorch + Flash + Torch Compile | 19 | 2.4× |
| **NIM (FP8, TensorRT-LLM, batching)** | **78** | **9.8×** |

---

## 7. Other NVIDIA Models

### 7.1 BioNeMo: Drug Discovery

**BioNeMo** is a platform for training AI models on biomolecular data.

<div class="feature-grid">
<div class="feature-card accent">
<h4>🧬 Protein Language Models</h4>
<ul>
<li><strong>ESM-2</strong> (650M-15B params): Protein sequence understanding</li>
<li><strong>ProtGPT2</strong>: Generate novel protein sequences</li>
<li><strong>AlphaFold2</strong>: Protein structure prediction (1-2 min per protein)</li>
</ul>
</div>

<div class="feature-card green">
<h4>🧪 Small Molecule Models</h4>
<ul>
<li><strong>MolMIM</strong>: Molecular representation learning</li>
<li><strong>MegaMolBART</strong>: SMILES-to-properties prediction</li>
<li><strong>DiffDock</strong>: Molecular docking (protein-ligand binding)</li>
</ul>
</div>

<div class="feature-card purple">
<h4>🔬 Generative Chemistry</h4>
<ul>
<li><strong>MolGAN</strong>: Generate drug-like molecules</li>
<li><strong>EquiBind</strong>: Fast docking (1000× faster than AutoDock)</li>
<li><strong>BioNeMo Service</strong>: Cloud API for drug discovery</li>
</ul>
</div>
</div>

**Example: Protein Folding with AlphaFold2 (BioNeMo)**

```python
from bionemo.alphafold2 import AlphaFold2Inference

# Initialize model
model = AlphaFold2Inference(
    model_name="alphafold2_ptm_ft",  # Fine-tuned for pTM scoring
    device="cuda",
)

# Predict structure from sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"

result = model.predict(
    sequence=sequence,
    num_recycles=3,              # Number of refinement iterations
    use_templates=False,         # Don't use template structures
)

# Output
print(f"pLDDT (confidence): {result.plddt.mean():.2f}")  # 0-100
print(f"pTM (topology): {result.ptm:.2f}")               # 0-1

# Save structure
result.save_pdb("predicted_structure.pdb")

# Visualize
import py3Dmol

view = py3Dmol.view(width=800, height=600)
view.addModel(open("predicted_structure.pdb").read(), "pdb")
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()
```

### 7.2 VISTA: Video Understanding

**VISTA** (VIdeo Segmentation with Transformers and Attention) for video analysis.

**Capabilities:**
- **Object Tracking**: Track objects across frames
- **Segmentation**: Pixel-level object masks
- **Action Recognition**: Classify actions in video
- **Video Captioning**: Generate descriptions
- **Temporal Grounding**: Find specific moments in video

**Example Use Cases:**
- Autonomous driving (track pedestrians, vehicles)
- Sports analysis (track players, ball)
- Medical imaging (track organs in ultrasound)
- Video editing (automatic background removal)

### 7.3 Edify: Image Generation

**Edify** is NVIDIA's text-to-image and image editing model.

**Features:**
- **Text-to-Image**: Generate images from descriptions
- **Image-to-Image**: Style transfer, upscaling
- **Inpainting**: Fill in missing parts
- **Outpainting**: Extend image boundaries
- **ControlNet**: Condition on edges, depth, pose
- **3D Generation**: Text → NeRF (3D scene)

**Example: Edify API**

```python
from nvidia_edify import EdifyClient

client = EdifyClient(api_key="your_ngc_key")

# Text-to-image
image = client.generate_image(
    prompt="A serene Japanese garden with cherry blossoms, koi pond, realistic lighting",
    negative_prompt="blurry, low quality, distorted",
    width=1024,
    height=1024,
    num_inference_steps=50,
    guidance_scale=7.5,
)
image.save("garden.png")

# Image editing
edited = client.edit_image(
    image="garden.png",
    prompt="Add a red bridge over the pond",
    mask="auto",                     # Automatic mask detection
    strength=0.7,                    # 0 = no change, 1 = full regeneration
)
edited.save("garden_with_bridge.png")

# Upscaling (4× resolution)
upscaled = client.upscale(
    image="garden.png",
    scale_factor=4,                  # 1024×1024 → 4096×4096
)
upscaled.save("garden_4k.png")
```

### 7.4 Audio2Face: Facial Animation

**Audio2Face** generates realistic facial animations from audio.

**Pipeline:**
```
Audio Input → Speech Analysis → Emotion Detection → Facial Rig → Animation
                                                          │
                                                          ▼
                                              (ARKit, MetaHuman, Custom)
```

**Applications:**
- Game character dialogue
- Virtual assistants (avatars)
- Dubbing & lip-sync
- Accessibility (sign language avatars)

### 7.5 Cosmos: World Models

**Cosmos** (2026) is NVIDIA's foundation model for physical world understanding.

**Capabilities:**
- **Video Prediction**: Predict future frames given past frames
- **Physics Simulation**: Learn physics from video (no explicit equations)
- **3D Scene Understanding**: Infer 3D structure from 2D video
- **Action-Conditional**: Predict outcome of actions ("what if I turn left?")

**Use Cases:**
- **Autonomous Vehicles**: Predict pedestrian/vehicle movement
- **Robotics**: Simulate task outcomes before execution
- **Gaming**: Procedural content generation
- **Scientific Discovery**: Model complex physical systems

**Architecture:**
- **Spatial Transformer**: Process video frames
- **Temporal Transformer**: Model dynamics over time
- **Latent Physics Engine**: Learn implicit physics
- **Decoder**: Generate future frames

**Training Data:**
- 10 million hours of video (driving, robotics, human activities)
- YouTube, Waymo Open Dataset, RoboNet, Ego4D
- Synthetic data from Omniverse

---

## 8. Timeline & Comparison

### 8.1 NVIDIA Model Release Timeline

```
2018 ──────────────────────────────────────────────────────────────────────→ 2026

        ┌─────────┐
        │StyleGAN │ First major generative model from NVIDIA
        └────┬────┘
             │
    ┌────────┴────────┐
    │  Megatron-LM    │ Large-scale transformer training (8B params)
    │   (2019)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │ Parakeet ASR    │ Speech recognition models
    │   (2020)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │ Megatron-Turing │ 530B params (with Microsoft)
    │   NLG (2021)    │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  NeMo Megatron  │ Open framework for LLM training
    │   (2022)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │ BioNeMo Platform│ Drug discovery models
    │   (2022)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  Nemotron-4     │ 8B, 15B, 340B models
    │   (2024)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  Canary-1B      │ Multilingual ASR
    │   (2024)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │    Edify        │ Text-to-image generation
    │   (2024)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Alpamayo      │ 480B MoE, multimodal flagship
    │   (2025)        │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   Cosmos        │ World models for robotics/AV
    │   (2026)        │
    └─────────────────┘
```

### 8.2 Master Comparison Table

**All NVIDIA Models at a Glance:**

| Model | Domain | Parameters | Modalities | Release | Open Weights | License |
|-------|--------|-----------|-----------|---------|--------------|---------|
| **Nemotron-4-340B** | LLM | 340B | Text | 2024 | ✅ Yes | NVIDIA Open Model |
| **Nemotron-4-15B** | LLM | 15B | Text | 2024 | ✅ Yes | NVIDIA Open Model |
| **Alpamayo** | LLM | 480B MoE | Text, Vision, Audio | 2025 | ✅ Yes | NVIDIA Open Model |
| **Parakeet-TDT-1.1B** | ASR | 1.1B | Audio → Text | 2024 | ✅ Yes | CC-BY-4.0 |
| **Canary-1B** | ASR | 1.1B | Audio → Text (80+ langs) | 2024 | ✅ Yes | CC-BY-4.0 |
| **ESM-2-15B** | Protein | 15B | Protein Sequences | 2024 | ✅ Yes | Apache 2.0 |
| **AlphaFold2** | Protein | 93M | Sequence → Structure | 2022 | ✅ Yes | Apache 2.0 |
| **MolMIM** | Chemistry | 87M | Molecules | 2023 | ✅ Yes | MIT |
| **VISTA** | Vision | 1.2B | Video | 2025 | ❌ API Only | Commercial |
| **Edify** | Generative | 2.3B | Text ↔ Image | 2024 | ❌ API Only | Commercial |
| **Audio2Face** | Audio | 180M | Audio → Face Animation | 2023 | ❌ API Only | Commercial |
| **Cosmos** | World Model | 7B | Video, Actions | 2026 | ✅ Research | Research-Only |
| **NV-Embed-v1** | Embeddings | 7.8B | Text → Vector | 2025 | ✅ Yes | NVIDIA Open Model |
| **Mistral-NeMo** | LLM | 12B | Text | 2024 | ✅ Yes | Apache 2.0 (co-developed) |

### 8.3 Hardware Requirements

**Minimum GPUs Required for Inference:**

| Model | FP16/BF16 | FP8 | INT4 | Cloud Cost ($/hr) |
|-------|-----------|-----|------|-------------------|
| Nemotron-4-8B | 1× A100 | 1× L40S | 1× L4 | $1.20 |
| Nemotron-4-15B | 2× A100 | 1× A100 | 1× A100 | $3.50 |
| Nemotron-4-340B | 16× A100 | 8× H100 | 4× H100 | $28.00 |
| Alpamayo (480B MoE) | 12× H100 | 8× H100 | 4× H100 | $24.00 |
| Parakeet-TDT-1.1B | 1× T4 | 1× T4 | 1× T4 | $0.35 |
| Edify-2B | 1× A100 | 1× L40S | N/A | $1.20 |
| Cosmos-7B | 2× A100 | 1× A100 | N/A | $3.50 |

<div class="warning-box orange">
<strong>💡 MoE Efficiency</strong><br>
Alpamayo (480B MoE) uses LESS GPU memory than Nemotron-4-340B (dense) because only 2/8 experts are active per token. Effective parameters: ~120B. This is why it needs fewer GPUs despite having more total parameters.
</div>

### 8.4 Model Selection Guide

<div class="comparison-grid">
<div class="comparison-card accent">
<h4>🎯 Choose Nemotron-4-340B if:</h4>
<ul>
<li>You need strong reasoning & factuality</li>
<li>You want open weights (full control)</li>
<li>You have 8+ H100 GPUs available</li>
<li>You need commercial deployment rights</li>
</ul>
</div>

<div class="comparison-card green">
<h4>🚀 Choose Alpamayo if:</h4>
<ul>
<li>You need multimodal (vision + audio)</li>
<li>You want GPT-4o-level performance</li>
<li>You care about inference cost (MoE efficiency)</li>
<li>128K context is important</li>
</ul>
</div>

<div class="comparison-card purple">
<h4>💰 Choose Nemotron-4-15B if:</h4>
<ul>
<li>Budget/latency constrained</li>
<li>Don't need frontier-level reasoning</li>
<li>2× A100 or 1× H100 available</li>
<li>High throughput > max quality</li>
</ul>
</div>

<div class="comparison-card cyan">
<h4>🎙️ Choose Parakeet/Canary if:</h4>
<ul>
<li>Speech-to-text is your use case</li>
<li>Low latency required (<100ms)</li>
<li>Multilingual support needed</li>
<li>Running on edge devices (T4, L4)</li>
</ul>
</div>
</div>

---

## 9. Key Takeaways

<div class="feature-grid">
<div class="feature-card accent">
<h4>🏗️ Vertical Integration</h4>
NVIDIA's strategy: control the entire stack from silicon (H100) to software (CUDA) to models (Nemotron) to applications (NIM). This creates an unprecedented moat.
</div>

<div class="feature-card green">
<h4>🔓 Open Weights Philosophy</h4>
Unlike OpenAI/Anthropic, NVIDIA releases model weights (Nemotron, Alpamayo). Why? They make money on GPUs, not API calls. Open models = more GPU sales.
</div>

<div class="feature-card purple">
<h4>⚙️ Optimized for NVIDIA Hardware</h4>
NeMo models are designed for NVIDIA GPUs. GQA reduces KV cache (fits in H100 SRAM), MoE architecture leverages NVLink, TensorRT-LLM kernels optimized for Hopper arch.
</div>

<div class="feature-card orange">
<h4>🎯 Domain-Specific Excellence</h4>
Instead of one general model, NVIDIA builds specialized models: BioNeMo (drug discovery), Riva (speech), VISTA (video), Cosmos (robotics). Each optimized for specific industries.
</div>

<div class="feature-card cyan">
<h4>🚢 NIM: Deployment Made Easy</h4>
Pre-optimized containers with TensorRT-LLM, quantization, batching. Deploy production-grade inference in minutes, not months. This lowers barrier to adoption → more GPU sales.
</div>

<div class="feature-card pink">
<h4>🔬 Research → Product Pipeline</h4>
NVIDIA Research publishes at top conferences (Megatron-LM, StyleGAN, Instant NGP). These innovations flow into products (NeMo, Edify, Omniverse) within 1-2 years.
</div>
</div>

---

## 10. Hands-On: Train a Small Model with NeMo

Let's train a 125M parameter GPT-style model on the TinyStories dataset.

```python
# install_nemo.sh
pip install nemo_toolkit['nlp']
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

# Preprocess
python preprocess_tinystories.py
```

```python
# preprocess_tinystories.py
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

# Train tokenizer
tokenizer = get_nmt_tokenizer(
    library='sentencepiece',
    model_name='tinystories_tokenizer',
    tokenizer_model=None,
    vocab_file=None,
    vocab_size=8192,
    training_sample_size=1000000,
)

# Tokenize dataset
with open('TinyStoriesV2-GPT4-train.txt', 'r') as f:
    text = f.read()

tokenizer.train([text])
tokenizer.save('tinystories_tokenizer.model')

# Create indexed dataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_builder

builder = make_builder('tinystories_indexed', impl='mmap', vocab_size=8192)

with open('TinyStoriesV2-GPT4-train.txt', 'r') as f:
    for line in f:
        tokens = tokenizer.text_to_ids(line.strip())
        builder.add_item(torch.IntTensor(tokens))

builder.finalize('tinystories_indexed.idx')
```

```python
# train_gpt_125m.py
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Model config (125M parameters)
model_cfg = {
    'num_layers': 12,
    'hidden_size': 768,
    'num_attention_heads': 12,
    'ffn_hidden_size': 3072,
    'max_position_embeddings': 512,
    'vocab_size': 8192,
    
    'activation': 'gelu',
    'normalization': 'layernorm',
    'position_embedding_type': 'learned_absolute',
    
    'bf16': True,
    
    'tensor_model_parallel_size': 1,
    'pipeline_model_parallel_size': 1,
    
    'micro_batch_size': 16,
    'global_batch_size': 128,
    
    'optimizer': {
        'name': 'adam',
        'lr': 6e-4,
        'weight_decay': 0.1,
        'betas': [0.9, 0.999],
    },
    'scheduler': {
        'name': 'CosineAnnealing',
        'warmup_steps': 1000,
        'min_lr': 6e-5,
    },
    
    'data': {
        'data_prefix': ['tinystories_indexed'],
        'num_workers': 4,
        'splits_string': '98,2,0',
    },
}

# Initialize model
model = MegatronGPTModel(cfg=model_cfg)

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='gpt-125m-{step}',
    every_n_train_steps=1000,
    save_top_k=3,
)

# Trainer
trainer = Trainer(
    devices=1,
    precision='bf16',
    max_steps=10000,
    val_check_interval=500,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

# Train
trainer.fit(model)
```

```python
# generate.py
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel

# Load trained model
model = MegatronGPTModel.restore_from('checkpoints/gpt-125m-step=10000.ckpt')
model.eval()

# Generate story
prompt = "Once upon a time, there was a little girl named"
generated = model.generate(
    inputs=[prompt],
    length=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)

print(generated[0])
```

**Expected Output:**
```
Once upon a time, there was a little girl named Lily. She loved to play outside
in her garden. One day, she saw a big, red ball. She wanted to play with it,
but it was too high. Lily tried to jump, but she couldn't reach it.

Then, a kind bird flew down. "I can help you!" said the bird. The bird picked
up the ball with its beak and gave it to Lily. Lily was so happy! She said,
"Thank you, bird!" They played together all day long.

The end.
```

---

**Next: [Chapter 7 — Parallelism on GPUs →](./07_parallelism.md)**

---

*Last updated: April 2026*
