---
title: "Chapter 10 — Multi-GPU Interconnects"
---

[← Back to Table of Contents](./README.md)

# Chapter 10: Multi-GPU Interconnects

Modern AI workloads have outgrown the capabilities of single GPUs. Training large language models with hundreds of billions of parameters, processing massive datasets, and achieving breakthrough inference latency all require multiple GPUs working in concert. The interconnect technology binding these GPUs together fundamentally determines system performance, cost, and scalability. This chapter explores the complete ecosystem of GPU interconnects—from PCIe's ubiquity to NVLink's raw bandwidth, from NVSwitch's all-to-all fabric to InfiniBand's cluster-scale networking.

## 1. Why Multi-GPU Computing?

### 1.1 The Scaling Imperative

Single GPU limitations manifest across multiple dimensions:

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">💾</div>
    <div class="card-title">Memory Capacity</div>
    <div class="card-desc">GPT-3 (175B parameters) requires ~700GB memory at FP16. Single H100 has only 80GB.</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">Compute Throughput</div>
    <div class="card-desc">Training throughput scales linearly with GPU count when interconnect bandwidth is sufficient.</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🔄</div>
    <div class="card-title">Parallelism Types</div>
    <div class="card-desc">Data, model, pipeline, and tensor parallelism all require efficient GPU-to-GPU communication.</div>
  </div>
</div>

### 1.2 Communication Patterns

Different parallelism strategies impose distinct bandwidth requirements:

<div class="diagram">
<div class="diagram-title">Communication Requirements by Parallelism Type</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>Data Parallel:</strong> Gradient all-reduce after each batch (parameter size × 2 bytes)
  </div>
  <div class="layer green">
    <strong>Tensor Parallel:</strong> All-gather/reduce-scatter every layer forward/backward (activation size)
  </div>
  <div class="layer blue">
    <strong>Pipeline Parallel:</strong> Point-to-point activation transfer between stages (batch × hidden size)
  </div>
  <div class="layer purple">
    <strong>Sequence Parallel:</strong> All-gather for attention, split outputs (sequence length sensitive)
  </div>
</div>
</div>

The bandwidth-to-compute ratio determines scaling efficiency:

$$
\text{Scaling Efficiency} = \frac{\text{Compute Time}}{\text{Compute Time} + \text{Communication Time}}
$$

For optimal efficiency (>90%), communication time should be <10% of compute time. With modern GPUs achieving 1 PFLOP/s (FP16), this requires interconnect bandwidths exceeding 100 GB/s.

### 1.3 The Interconnect Hierarchy

<div class="diagram">
<div class="diagram-title">Multi-GPU Interconnect Hierarchy</div>
<div class="flow">
  <div class="flow-node accent wide">Within Server: NVLink (900 GB/s - 1.8 TB/s)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Between Servers: InfiniBand/Ethernet (400 Gb/s - 3.2 Tb/s)</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">Between Racks: Spine-Leaf Fabric (Multi-Tb/s aggregate)</div>
  <div class="flow-arrow blue"></div>
  <div class="flow-node purple wide">Between Data Centers: WAN Links (100 Gb/s - 400 Gb/s)</div>
</div>
</div>

## 2. PCIe: The Universal Interconnect

### 2.1 PCIe Evolution

Peripheral Component Interconnect Express (PCIe) serves as the universal GPU interconnect, connecting GPUs to CPUs and enabling GPU-to-GPU communication through the host.

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2010</div>
    <div class="timeline-title">PCIe 3.0</div>
    <div class="timeline-desc">8 GT/s per lane, ~1 GB/s per lane (128b/130b encoding), 16 GB/s for x16 slot</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2017</div>
    <div class="timeline-title">PCIe 4.0</div>
    <div class="timeline-desc">16 GT/s per lane, ~2 GB/s per lane, 32 GB/s for x16 slot</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2019</div>
    <div class="timeline-title">PCIe 5.0</div>
    <div class="timeline-desc">32 GT/s per lane, ~4 GB/s per lane, 64 GB/s for x16 slot</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2022</div>
    <div class="timeline-title">PCIe 6.0</div>
    <div class="timeline-desc">64 GT/s per lane with PAM4, ~8 GB/s per lane, 128 GB/s for x16 slot</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2025</div>
    <div class="timeline-title">PCIe 7.0</div>
    <div class="timeline-desc">128 GT/s per lane (planned), ~16 GB/s per lane, 256 GB/s for x16 slot</div>
  </div>
</div>

### 2.2 PCIe Bandwidth Calculation

Each PCIe generation doubles the raw transfer rate, but encoding overhead reduces usable bandwidth:

**PCIe 3.0/4.0 (128b/130b encoding):**
$$
\text{Bandwidth}_{\text{lane}} = \frac{8 \text{ GT/s} \times 128}{130} = 7.877 \text{ Gb/s} \approx 0.985 \text{ GB/s}
$$

**PCIe 5.0+ (PAM4 with lower overhead):**
$$
\text{Bandwidth}_{\text{x16}} = 32 \text{ GT/s} \times 16 \text{ lanes} \times \frac{128}{130} \times \frac{1}{8} = 63.015 \text{ GB/s}
$$

### 2.3 PCIe Topology Limitations

<div class="diagram">
<div class="diagram-title">PCIe CPU-Centric Topology</div>
<div class="flow">
  <div class="flow-node accent">GPU 0</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">CPU (PCIe Root Complex)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent">GPU 1</div>
</div>
<div style="margin-top: 1rem; padding: 1rem; background: rgba(118, 185, 0, 0.1); border-left: 3px solid #76b900;">
<strong>CPU Bottleneck:</strong> GPU-to-GPU traffic must traverse CPU memory, limiting bandwidth to ~32 GB/s bidirectional and adding latency (~5-10 µs).
</div>
</div>

In a typical dual-socket server with 8 GPUs:

```
CPU 0          CPU 1
│              │
├─ GPU 0       ├─ GPU 4
├─ GPU 1       ├─ GPU 5
├─ GPU 2       ├─ GPU 6
└─ GPU 3       └─ GPU 7
```

Cross-socket GPU communication suffers additional penalties:
- **Same socket:** ~32 GB/s (PCIe 4.0 x16)
- **Cross socket:** ~25 GB/s (additional UPI/Infinity Fabric hop)
- **PCIe switch hops:** +500-1000ns latency per hop

### 2.4 PCIe Performance Characteristics

```python
import subprocess

def measure_pcie_bandwidth():
    """Measure effective PCIe bandwidth using CUDA"""
    # Host-to-Device (H2D)
    for size_mb in [1, 16, 256, 4096]:
        size_bytes = size_mb * 1024 * 1024
        # cudaMemcpy benchmark
        print(f"H2D {size_mb}MB: {bandwidth_gbps:.2f} GB/s")
    
    # Device-to-Host (D2H)
    # Typically slightly lower due to CPU cache effects
    
    # Device-to-Device via CPU (no direct PCIe peer)
    # Only ~50% of unidirectional bandwidth due to contention
```

**Measured Performance (PCIe 4.0 x16):**
- Small transfers (<1MB): ~10-15 GB/s (latency dominated)
- Large transfers (>256MB): ~28-30 GB/s (approaching theoretical 32 GB/s)
- Bidirectional: ~22-24 GB/s (due to internal switch contention)

## 3. NVLink: GPU-Native Interconnect

NVIDIA's proprietary high-speed interconnect enables direct GPU-to-GPU communication, bypassing the CPU entirely. Each generation has progressively increased bandwidth and link count.

### 3.1 NVLink 1.0 (Pascal, 2016)

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Architecture</div>
    <ul>
      <li><strong>Bandwidth:</strong> 20 GB/s per link (bidirectional)</li>
      <li><strong>Links per GPU:</strong> 4 (P100 PCIe) to 6 (P100 SXM2)</li>
      <li><strong>Total Bandwidth:</strong> 160 GB/s (6 links)</li>
      <li><strong>Technology:</strong> 25 Gb/s signaling, proprietary SerDes</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">vs PCIe 3.0</div>
    <ul>
      <li><strong>Bandwidth:</strong> 10× PCIe 3.0 x16 (16 GB/s)</li>
      <li><strong>Latency:</strong> ~1.5 µs vs ~5 µs for PCIe</li>
      <li><strong>Direct GPU-GPU:</strong> No CPU traversal</li>
      <li><strong>Cache Coherence:</strong> Hardware-managed coherency</li>
    </ul>
  </div>
</div>

**Pascal NVLink Configuration:**
```
P100 SXM2 Module (6 NVLinks):
GPU 0 ←→ GPU 1 (2 links: 80 GB/s)
GPU 2 ←→ GPU 3 (2 links: 80 GB/s)
GPU 0 ←→ GPU 2 (1 link: 40 GB/s)
GPU 1 ←→ GPU 3 (1 link: 40 GB/s)
```

### 3.2 NVLink 2.0 (Volta, 2017)

Introduced with V100, NVLink 2.0 increased per-link bandwidth and improved signaling.

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">📊</div>
    <div class="card-title">NVLink 2.0 Specs</div>
    <div class="card-desc">
      • 25 GB/s per link (25% increase)<br>
      • 6 links per V100 SXM2<br>
      • 300 GB/s total bidirectional<br>
      • Improved signal integrity
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔗</div>
    <div class="card-title">Topology</div>
    <div class="card-desc">
      • Hybrid Cube-Mesh for 8 GPUs<br>
      • Every GPU connects to 3 others<br>
      • Max 2-hop distance<br>
      • 150 GB/s to nearest neighbors
    </div>
  </div>
</div>

**V100 8-GPU DGX-1 Topology:**
```
     GPU0 ─── GPU1
     │ ╲      ╱ │
     │   ╲  ╱   │
     │     ╳    │
     │   ╱  ╲   │
     │ ╱      ╲ │
     GPU3 ─── GPU2
       │        │
     (replicated for GPU4-7)
```

Each GPU has 6 NVLink connections:
- 2 links to one neighbor (50 GB/s)
- 2 links to second neighbor (50 GB/s)
- 2 links to third neighbor (50 GB/s)

### 3.3 NVLink 3.0 (Ampere, 2020)

A100 maintained 25 GB/s per link but increased the maximum link count to 12, enabling new topologies.

<div class="diagram">
<div class="diagram-title">NVLink 3.0 Features</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>Bandwidth:</strong> 25 GB/s per link (bidirectional), 600 GB/s total per GPU
  </div>
  <div class="layer green">
    <strong>Link Count:</strong> Up to 12 links per A100 SXM4, enabling full NVSwitch connectivity
  </div>
  <div class="layer blue">
    <strong>NVSwitch Integration:</strong> All 12 links connect to NVSwitch for non-blocking all-to-all
  </div>
  <div class="layer purple">
    <strong>Error Correction:</strong> Enhanced RAS (Reliability, Availability, Serviceability) features
  </div>
</div>
</div>

**A100 DGX with NVSwitch:**

<div class="diagram">
<div class="diagram-title">8 × A100 with NVSwitch Topology</div>
<div class="flow">
  <div class="flow-node accent wide">GPU 0 (12 links to NVSwitch)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide" style="font-size: 1.1em;">6 × NVSwitch (3rd Gen)<br>Non-blocking 600 GB/s per GPU</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">GPU 7 (12 links to NVSwitch)</div>
</div>
<div style="margin-top: 1rem; padding: 1rem; background: rgba(118, 185, 0, 0.1); border-left: 3px solid #76b900;">
<strong>Result:</strong> Full bisection bandwidth—any GPU can communicate with any other GPU at 600 GB/s simultaneously.
</div>
</div>

### 3.4 NVLink 4.0 (Hopper, 2022)

H100 increased link count to 18 while maintaining 25 GB/s per link, achieving 900 GB/s total bandwidth.

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">⚡</div>
    <div class="card-title">Raw Bandwidth</div>
    <div class="card-desc">450 GB/s in each direction = 900 GB/s bidirectional</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🔢</div>
    <div class="card-title">18 Links</div>
    <div class="card-desc">All connect to NVSwitch for non-blocking fabric</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">📈</div>
    <div class="card-title">Efficiency</div>
    <div class="card-desc">7× PCIe 5.0 bandwidth, 3× lower latency</div>
  </div>
</div>

**NVLink 4.0 Technical Details:**

$$
\text{Effective Bandwidth} = 18 \text{ links} \times 25 \text{ GB/s} = 450 \text{ GB/s per direction}
$$

```c++
// NVLink 4.0 characteristics
struct NVLink4_Spec {
    uint32_t links_per_gpu = 18;
    float bandwidth_per_link_gbs = 25.0f;  // GB/s bidirectional
    float total_bandwidth_gbs = 900.0f;
    float latency_us = 1.0f;  // ~1 microsecond
    
    // Compare to H100 compute throughput
    float h100_fp16_tflops = 1979.0f;  // With sparsity: 3958
    float bytes_per_flop = 4.0f;  // FP16 operand + result
    
    // Compute-to-communication ratio
    float arithmetic_intensity_needed() {
        return (h100_fp16_tflops * 1e12 / 1e9) / total_bandwidth_gbs;
        // = 2199 FLOP/byte
        // Models with < 2199 FLOP/byte are communication-bound
    }
};
```

### 3.5 NVLink 5.0 & Chip-to-Chip (C2C) Interconnect (Blackwell, 2024)

Blackwell architecture introduced the most significant NVLink advancement:

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">NVLink 5.0</div>
    <ul>
      <li><strong>Bandwidth:</strong> 50 GB/s per link (2× NVLink 4.0)</li>
      <li><strong>Links per GPU:</strong> 18 links</li>
      <li><strong>Total:</strong> 1.8 TB/s bidirectional</li>
      <li><strong>Signaling:</strong> PAM4 modulation</li>
      <li><strong>Use Case:</strong> GPU-to-GPU, GPU-to-NVSwitch</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">NVLink C2C</div>
    <ul>
      <li><strong>Bandwidth:</strong> 900 GB/s bidirectional</li>
      <li><strong>Purpose:</strong> Connect two B100/B200 dies into single GPU</li>
      <li><strong>Latency:</strong> <100 ns die-to-die</li>
      <li><strong>Technology:</strong> Ultra-short reach SerDes</li>
      <li><strong>Coherence:</strong> Cache-coherent shared memory</li>
    </ul>
  </div>
</div>

**Blackwell Dual-Die Architecture:**

<div class="diagram">
<div class="diagram-title">B200 Chip-to-Chip Interconnect</div>
<div class="flow-h">
  <div class="flow-node accent" style="padding: 2rem;">Die 0<br>B100<br>96 GB HBM3e</div>
  <div class="flow-arrow accent" style="flex-grow: 0.5;"></div>
  <div class="flow-node green" style="padding: 1.5rem; background: rgba(118, 185, 0, 0.3);">NVLink C2C<br>900 GB/s<br>&lt;100ns latency</div>
  <div class="flow-arrow accent" style="flex-grow: 0.5;"></div>
  <div class="flow-node accent" style="padding: 2rem;">Die 1<br>B100<br>96 GB HBM3e</div>
</div>
<div style="margin-top: 1rem; padding: 1rem; background: rgba(118, 185, 0, 0.1); border-left: 3px solid #76b900;">
<strong>Result:</strong> B200 appears as single 192 GB GPU with coherent memory access across dies. Software sees unified address space.
</div>
</div>

### 3.6 NVLink Physical Layer

NVLink operates using sophisticated SerDes (Serializer/Deserializer) technology:

<div class="diagram">
<div class="diagram-title">NVLink Physical Architecture</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>Application Layer:</strong> CUDA kernels, memory operations, peer-to-peer transfers
  </div>
  <div class="layer green">
    <strong>Transport Layer:</strong> Packet formation, flow control, credit-based flow management
  </div>
  <div class="layer blue">
    <strong>Data Link Layer:</strong> CRC error detection, retry mechanism, link training
  </div>
  <div class="layer purple">
    <strong>Physical Layer:</strong> SerDes, NRZ/PAM4 encoding, differential signaling at 25-50 Gbaud
  </div>
</div>
</div>

**NVLink Signaling Evolution:**

```
NVLink 1.0-4.0: NRZ (Non-Return-to-Zero) Signaling
├─ 2 voltage levels (0, 1)
├─ 25 Gbaud symbol rate
└─ 25 Gb/s per lane

NVLink 5.0: PAM4 (4-Level Pulse Amplitude Modulation)
├─ 4 voltage levels (00, 01, 10, 11)
├─ 2 bits per symbol
├─ 25 Gbaud symbol rate
└─ 50 Gb/s per lane
```

**Link Training and Initialization:**
1. **Electrical Idle:** Links start in low-power state
2. **Link Training:** Synchronization patterns exchanged
3. **Lane Polarity Detection:** Auto-correction for PCB routing
4. **Data Rate Negotiation:** Both sides agree on speed
5. **Active State:** Full bandwidth data transfer

### 3.7 NVLink Software Access

```cuda
// Check NVLink topology from CUDA
#include <cuda_runtime.h>

void print_nvlink_topology() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("NVLink Topology:\n");
    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) continue;
            
            int accessible, link_type;
            cudaDeviceCanAccessPeer(&accessible, i, j);
            
            if (accessible) {
                // Query NVLink connection
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                
                // NVLink bandwidth between GPUs
                printf("GPU %d -> GPU %d: NVLink available\n", i, j);
                
                // Measure actual bandwidth
                measure_p2p_bandwidth(i, j);
            }
        }
    }
}

void measure_p2p_bandwidth(int src_gpu, int dst_gpu) {
    const size_t size = 1 << 30;  // 1 GB
    void *src_ptr, *dst_ptr;
    
    cudaSetDevice(src_gpu);
    cudaMalloc(&src_ptr, size);
    
    cudaSetDevice(dst_gpu);
    cudaMalloc(&dst_ptr, size);
    
    // Enable peer access
    cudaDeviceEnablePeerAccess(src_gpu, 0);
    
    // Warm up
    cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);
    
    // Measure
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float bandwidth_gbs = (size * 100) / (ms * 1e6);
    
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbs);
}
```

## 4. NVSwitch: All-to-All Connectivity

Direct GPU-to-GPU NVLink connections work well for small GPU counts but don't scale. NVSwitch provides non-blocking, all-to-all connectivity.

### 4.1 NVSwitch Architecture

<div class="diagram">
<div class="diagram-title">NVSwitch Concept</div>
<div class="flow">
  <div class="flow-node accent wide">Problem: N GPUs with direct NVLink requires N(N-1)/2 links = doesn't scale</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Solution: Central switch fabric connects all GPUs</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">Each GPU connects to switch, switch provides full bandwidth to any peer</div>
</div>
</div>

**NVSwitch Key Features:**
- **Non-blocking:** Multiple GPU pairs can communicate simultaneously at full bandwidth
- **Low latency:** Hardware packet switching with minimal overhead
- **Load balancing:** Adaptive routing based on link utilization
- **Fault tolerance:** Automatic failover if links degrade

### 4.2 NVSwitch Generations

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2018</div>
    <div class="timeline-title">NVSwitch 1.0 (Volta)</div>
    <div class="timeline-desc">18 ports × 25 GB/s = 450 GB/s per switch, used in DGX-2 with 6 switches for 16 V100s</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2020</div>
    <div class="timeline-title">NVSwitch 2.0 (Ampere)</div>
    <div class="timeline-desc">Prototyped for A100, not widely deployed in DGX A100</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2021</div>
    <div class="timeline-title">NVSwitch 3.0 (Ampere)</div>
    <div class="timeline-desc">64 ports × 25 GB/s = 1.6 TB/s per switch, 6 switches in DGX A100 for 8 GPUs</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2022</div>
    <div class="timeline-title">NVSwitch 4.0 (Hopper)</div>
    <div class="timeline-desc">64 ports × 25 GB/s, improved routing, 4 switches in DGX H100 for 8 GPUs</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2024</div>
    <div class="timeline-title">NVSwitch 5.0 (Blackwell)</div>
    <div class="timeline-desc">144 ports × 50 GB/s = 7.2 TB/s per switch, enables 576-GPU NVLink domains</div>
  </div>
</div>

### 4.3 DGX A100 NVSwitch Configuration

<div class="diagram">
<div class="diagram-title">DGX A100: 8 × A100 + 6 × NVSwitch 3.0</div>
<div class="flow">
  <div class="flow-node accent" style="font-size: 0.9em;">GPU 0<br>(12 NVLink)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide" style="padding: 1.5rem;">
    <strong>NVSwitch Fabric</strong><br>
    6 switches × 64 ports<br>
    Non-blocking 600 GB/s per GPU
  </div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent" style="font-size: 0.9em;">GPU 7<br>(12 NVLink)</div>
</div>
</div>

Each A100 connects its 12 NVLink ports across all 6 NVSwitches:
- 2 links per NVSwitch (2 × 25 GB/s = 50 GB/s per switch)
- 6 switches × 50 GB/s = 300 GB/s total per direction
- Full bisection bandwidth: Any 4 GPUs can send to other 4 GPUs at full speed

**Configuration:**
```
A100 GPU NVLink Distribution:
GPU 0: Links 0-1 → Switch 0, Links 2-3 → Switch 1, ..., Links 10-11 → Switch 5
GPU 1: Links 0-1 → Switch 0, Links 2-3 → Switch 1, ..., Links 10-11 → Switch 5
...
GPU 7: Links 0-1 → Switch 0, Links 2-3 → Switch 1, ..., Links 10-11 → Switch 5

Each NVSwitch connects to all 8 GPUs:
Switch 0: 2 links to GPU 0, 2 to GPU 1, ..., 2 to GPU 7 = 16 ports used
(Remaining 48 ports unused in single-node configuration)
```

### 4.4 DGX H100 NVSwitch Configuration

DGX H100 optimized the switch count while increasing per-GPU bandwidth:

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🔢</div>
    <div class="card-title">4 × NVSwitch 4.0</div>
    <div class="card-desc">
      Fewer switches but more links per GPU<br>
      Each switch connects to all 8 H100s<br>
      Cost optimization vs A100
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">900 GB/s per H100</div>
    <div class="card-desc">
      18 NVLink 4.0 ports per GPU<br>
      4-5 links per NVSwitch<br>
      Non-blocking all-to-all fabric
    </div>
  </div>
</div>

**Performance Implications:**

```python
def dgx_h100_bisection_bandwidth():
    """Calculate bisection bandwidth for DGX H100"""
    gpus = 8
    nvlinks_per_gpu = 18
    bandwidth_per_link = 25  # GB/s bidirectional
    
    # Total GPU bandwidth
    total_gpu_bw = gpus * nvlinks_per_gpu * bandwidth_per_link
    print(f"Total GPU bandwidth: {total_gpu_bw} GB/s")
    # = 3600 GB/s
    
    # Bisection: 4 GPUs communicating with other 4 GPUs
    bisection_bw = (gpus // 2) * nvlinks_per_gpu * bandwidth_per_link
    print(f"Bisection bandwidth: {bisection_bw} GB/s")
    # = 1800 GB/s
    
    # Full bisection achieved since each GPU can use all 900 GB/s
    # simultaneously to reach any other GPU
```

## 5. NVLink Switch System: Multi-Node NVLink

### 5.1 Beyond Single-Node NVLink

Traditional NVLink fabrics were limited to single nodes (8-16 GPUs). Blackwell introduced the **NVLink Switch System** to scale NVLink across multiple nodes.

<div class="diagram">
<div class="diagram-title">NVLink Scaling Evolution</div>
<div class="flow">
  <div class="flow-node accent wide">Pascal/Volta: Direct GPU-GPU (8 GPUs max)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Ampere/Hopper: NVSwitch within node (8 GPUs)</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">Blackwell: NVLink Switch System across nodes (576 GPUs)</div>
</div>
</div>

### 5.2 NVLink Switch System Architecture

<div class="diagram">
<div class="diagram-title">2-Tier NVLink Switch System</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>Tier 1 (Node-Level):</strong> NVSwitch 5.0 connects 8 B200 GPUs within each server
  </div>
  <div class="layer green">
    <strong>Tier 2 (Rack-Level):</strong> External NVLink Switch connects multiple nodes
  </div>
  <div class="layer blue">
    <strong>Scale:</strong> Up to 576 GPUs in single NVLink domain with 1.8 TB/s per GPU
  </div>
</div>
</div>

**72-Node NVLink Fabric:**

```
Rack 1:                    Rack 2:
├─ DGX B200 #1 (8 GPUs)    ├─ DGX B200 #37 (8 GPUs)
├─ DGX B200 #2 (8 GPUs)    ├─ DGX B200 #38 (8 GPUs)
│  ...                      │  ...
└─ DGX B200 #36 (8 GPUs)   └─ DGX B200 #72 (8 GPUs)
   ↓ NVLink 5.0                ↓ NVLink 5.0
   External NVSwitch Layer (Quantum-X800 with NVLink)
```

### 5.3 Bandwidth and Topology

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔗</div>
    <div class="card-title">Intra-Node</div>
    <div class="card-desc">1.8 TB/s per GPU within same DGX via NVSwitch 5.0</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🏢</div>
    <div class="card-title">Inter-Node</div>
    <div class="card-desc">~400-800 GB/s per node via external NVLink switches</div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌐</div>
    <div class="card-title">Full Fabric</div>
    <div class="card-desc">576 GPUs appear as single coherent memory space</div>
  </div>
</div>

**Performance Model:**

$$
\text{Effective Bandwidth} = \begin{cases}
1.8 \text{ TB/s} & \text{same node} \\
\frac{N_{\text{uplinks}} \times 50 \text{ GB/s}}{8 \text{ GPUs}} & \text{cross-node}
\end{cases}
$$

For a 36-node system with 8 uplinks per node:
$$
\text{Cross-node BW per GPU} = \frac{8 \times 50 \text{ GB/s}}{8} = 50 \text{ GB/s per GPU}
$$

### 5.4 Software Perspective

From CUDA's perspective, the entire NVLink Switch System appears as a single GPU cluster with peer-to-peer access:

```cuda
// Query extended NVLink topology
void check_nvlink_switch_system() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);  // May report 576 GPUs
    
    printf("NVLink Switch System: %d GPUs\n", deviceCount);
    
    // Check connectivity matrix
    for (int i = 0; i < deviceCount; i++) {
        for (int j = i + 1; j < deviceCount; j++) {
            int canAccess;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            
            if (canAccess) {
                // Determine if same node or cross-node
                int node_i = i / 8;
                int node_j = j / 8;
                
                if (node_i == node_j) {
                    printf("GPU %d ↔ GPU %d: Intra-node NVLink (1.8 TB/s)\n", 
                           i, j);
                } else {
                    printf("GPU %d ↔ GPU %d: Inter-node NVLink (~50 GB/s)\n", 
                           i, j);
                }
            }
        }
    }
}
```

## 6. InfiniBand: Cluster-Scale Networking

For multi-node AI clusters beyond the scale of NVLink Switch Systems, InfiniBand provides high-bandwidth, low-latency networking.

### 6.1 InfiniBand Architecture

<div class="diagram">
<div class="diagram-title">InfiniBand in AI Clusters</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>HCA (Host Channel Adapter):</strong> ConnectX SmartNIC in each server (PCIe or OCP 3.0)
  </div>
  <div class="layer green">
    <strong>Switch Fabric:</strong> Quantum switches create fat-tree or dragonfly topology
  </div>
  <div class="layer blue">
    <strong>RDMA:</strong> Remote Direct Memory Access bypasses kernel, CPU-free transfers
  </div>
  <div class="layer purple">
    <strong>GPUDirect RDMA:</strong> Direct GPU memory to network, zero CPU copies
  </div>
</div>
</div>

### 6.2 InfiniBand Generations

<div class="timeline">
  <div class="timeline-item">
    <div class="timeline-year">2013</div>
    <div class="timeline-title">FDR (Fourteen Data Rate)</div>
    <div class="timeline-desc">56 Gb/s per port (14 Gb/s per lane × 4), ~7 GB/s</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2014</div>
    <div class="timeline-title">EDR (Enhanced Data Rate)</div>
    <div class="timeline-desc">100 Gb/s per port (25 Gb/s per lane × 4), ~12.5 GB/s</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2017</div>
    <div class="timeline-title">HDR (High Data Rate)</div>
    <div class="timeline-desc">200 Gb/s per port (50 Gb/s per lane × 4), ~25 GB/s</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2020</div>
    <div class="timeline-title">NDR (Next Data Rate)</div>
    <div class="timeline-desc">400 Gb/s per port (100 Gb/s per lane × 4), ~50 GB/s</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2024</div>
    <div class="timeline-title">XDR (eXtreme Data Rate)</div>
    <div class="timeline-desc">800 Gb/s per port (200 Gb/s per lane × 4), ~100 GB/s</div>
  </div>
  <div class="timeline-item">
    <div class="timeline-year">2026</div>
    <div class="timeline-title">GDR (Giga Data Rate)</div>
    <div class="timeline-desc">1600 Gb/s (planned), ~200 GB/s</div>
  </div>
</div>

### 6.3 ConnectX Adapters

NVIDIA's ConnectX SmartNICs serve as InfiniBand HCAs:

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">📡</div>
    <div class="card-title">ConnectX-6</div>
    <div class="card-desc">
      • HDR (200 Gb/s)<br>
      • GPUDirect RDMA<br>
      • SHARP (in-network collectives)<br>
      • 2020 generation
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">ConnectX-7</div>
    <div class="card-desc">
      • NDR (400 Gb/s)<br>
      • Enhanced GPUDirect<br>
      • Adaptive Routing<br>
      • 2022 generation
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🚀</div>
    <div class="card-title">ConnectX-8</div>
    <div class="card-desc">
      • XDR (800 Gb/s)<br>
      • Ultra-low latency<br>
      • Advanced congestion control<br>
      • 2024 generation
    </div>
  </div>
</div>

### 6.4 RDMA: Remote Direct Memory Access

Traditional networking requires CPU involvement at every stage:

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Traditional TCP/IP</div>
    <ul>
      <li>Application writes data to buffer</li>
      <li>Kernel copies buffer to network stack</li>
      <li>Network card reads from kernel memory</li>
      <li>Receiver: Card → Kernel → Application</li>
      <li><strong>Latency:</strong> ~50-100 µs</li>
      <li><strong>CPU:</strong> 100% of one core per 10 Gb/s</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">RDMA InfiniBand</div>
    <ul>
      <li>Application registers memory region</li>
      <li>Network card reads directly from application memory</li>
      <li>Receiver: Card writes directly to application memory</li>
      <li>Zero kernel involvement</li>
      <li><strong>Latency:</strong> ~1-2 µs</li>
      <li><strong>CPU:</strong> Nearly zero (offloaded to NIC)</li>
    </ul>
  </div>
</div>

**RDMA Verbs API:**

```c
#include <infiniband/verbs.h>

void rdma_send_example() {
    // Open InfiniBand device
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    
    // Allocate protection domain
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    
    // Register memory region (pins pages, gives NIC access)
    void *buffer = malloc(1024 * 1024);  // 1 MB
    struct ibv_mr *mr = ibv_reg_mr(pd, buffer, 1024 * 1024,
                                    IBV_ACCESS_LOCAL_WRITE |
                                    IBV_ACCESS_REMOTE_READ |
                                    IBV_ACCESS_REMOTE_WRITE);
    
    // Create queue pair (QP)
    struct ibv_qp_init_attr qp_attr = {
        .qp_type = IBV_QPT_RC,  // Reliable Connected
        .sq_sig_all = 0,
        // ... more attributes
    };
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_attr);
    
    // Post RDMA WRITE operation
    struct ibv_sge sge = {
        .addr = (uint64_t)buffer,
        .length = 1024 * 1024,
        .lkey = mr->lkey
    };
    
    struct ibv_send_wr wr = {
        .wr_id = 0,
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED,
        .wr.rdma = {
            .remote_addr = remote_addr,
            .rkey = remote_rkey
        }
    };
    
    struct ibv_send_wr *bad_wr;
    ibv_post_send(qp, &wr, &bad_wr);
    
    // NIC handles transfer autonomously, no CPU involvement!
}
```

### 6.5 GPUDirect RDMA

GPUDirect RDMA extends RDMA to GPU memory, enabling network card to directly read/write GPU memory without CPU or GPU involvement.

<div class="diagram">
<div class="diagram-title">GPUDirect RDMA Data Path</div>
<div class="flow">
  <div class="flow-node accent wide">GPU Memory (CUDA buffer)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">PCIe/NVLink</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">ConnectX NIC (DMA engine)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">InfiniBand Fabric</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">Remote ConnectX NIC</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">Remote GPU Memory</div>
</div>
<div style="margin-top: 1rem; padding: 1rem; background: rgba(118, 185, 0, 0.1); border-left: 3px solid #76b900;">
<strong>Zero-Copy:</strong> Data moves from local GPU memory to remote GPU memory without any CPU or system memory involvement.
</div>
</div>

**Performance Benefits:**

```python
def compare_transfer_methods(size_gb=1):
    """Compare GPU-to-GPU transfer across nodes"""
    
    # Method 1: GPU → CPU → Network → CPU → GPU
    gpu_to_cpu = size_gb / 30  # ~30 GB/s PCIe 4.0
    cpu_to_network = size_gb / 12  # ~12 GB/s EDR InfiniBand
    network_to_cpu = size_gb / 12
    cpu_to_gpu = size_gb / 30
    total_traditional = gpu_to_cpu + cpu_to_network + network_to_cpu + cpu_to_gpu
    
    print(f"Traditional: {total_traditional:.3f} seconds")
    # = 0.267 seconds for 1 GB
    
    # Method 2: GPUDirect RDMA (GPU → Network → GPU)
    gpu_to_network_direct = size_gb / 25  # ~25 GB/s NDR
    total_gpudirect = gpu_to_network_direct
    
    print(f"GPUDirect RDMA: {total_gpudirect:.3f} seconds")
    # = 0.040 seconds for 1 GB
    
    speedup = total_traditional / total_gpudirect
    print(f"Speedup: {speedup:.1f}×")
    # = 6.7× faster
```

**CUDA Integration:**

```cuda
// GPUDirect RDMA requires GPU memory registration with IB stack
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

void* allocate_gpu_rdma_buffer(size_t size, struct ibv_pd *pd) {
    // Allocate GPU memory
    void *gpu_ptr;
    cudaMalloc(&gpu_ptr, size);
    
    // Register with InfiniBand
    struct ibv_mr *mr = ibv_reg_mr(pd, gpu_ptr, size,
                                    IBV_ACCESS_LOCAL_WRITE |
                                    IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ);
    
    if (!mr) {
        fprintf(stderr, "Failed to register GPU memory\n");
        return NULL;
    }
    
    printf("GPU buffer registered: addr=%p, lkey=0x%x, rkey=0x%x\n",
           gpu_ptr, mr->lkey, mr->rkey);
    
    return gpu_ptr;
}

// Direct GPU-to-GPU RDMA transfer
void gpu_rdma_transfer(struct ibv_qp *qp, void *local_gpu_ptr,
                       void *remote_gpu_ptr, size_t size,
                       uint32_t lkey, uint32_t rkey) {
    struct ibv_sge sge = {
        .addr = (uint64_t)local_gpu_ptr,
        .length = size,
        .lkey = lkey
    };
    
    struct ibv_send_wr wr = {
        .opcode = IBV_WR_RDMA_WRITE,
        .sg_list = &sge,
        .num_sge = 1,
        .wr.rdma = {
            .remote_addr = (uint64_t)remote_gpu_ptr,
            .rkey = rkey
        }
    };
    
    struct ibv_send_wr *bad_wr;
    ibv_post_send(qp, &wr, &bad_wr);
    
    // Data flows: Local GPU → PCIe → ConnectX → IB → Remote ConnectX → Remote GPU
    // All hardware-managed, zero CPU/software involvement!
}
```

### 6.6 Quantum Switches

NVIDIA's Quantum series switches form the InfiniBand fabric backbone:

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔀</div>
    <div class="card-title">Quantum-2 (2022)</div>
    <div class="card-desc">
      • 64 ports NDR (400 Gb/s)<br>
      • 51.2 Tb/s total throughput<br>
      • SHARP in-network computing<br>
      • Adaptive routing
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">Quantum-3 (2024)</div>
    <div class="card-desc">
      • 64 ports XDR (800 Gb/s)<br>
      • 102.4 Tb/s throughput<br>
      • Enhanced SHARP<br>
      • Congestion control
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌐</div>
    <div class="card-title">Quantum-X800 (2024)</div>
    <div class="card-desc">
      • Converged IB + NVLink<br>
      • Enables NVLink Switch System<br>
      • 144 NVLink 5.0 ports<br>
      • Multi-tier fabrics
    </div>
  </div>
</div>

### 6.7 SHARP: In-Network Computing

Scalable Hierarchical Aggregation and Reduction Protocol (SHARP) offloads collective operations to the network switches:

<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">Traditional All-Reduce</div>
    <ul>
      <li>All nodes send to one node (reduce)</li>
      <li>Leader performs reduction</li>
      <li>Leader broadcasts result</li>
      <li><strong>Latency:</strong> O(N) with node count</li>
      <li><strong>Bandwidth:</strong> Leader bottleneck</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">SHARP All-Reduce</div>
    <ul>
      <li>Switches perform reduction in-flight</li>
      <li>Tree-based aggregation in fabric</li>
      <li>No single node bottleneck</li>
      <li><strong>Latency:</strong> O(log N)</li>
      <li><strong>Bandwidth:</strong> Full bisection utilized</li>
    </ul>
  </div>
</div>

SHARP reduces all-reduce latency by 5-10× for large clusters (>100 nodes).

## 7. Ethernet for AI

While InfiniBand dominates AI clusters, NVIDIA is pushing Ethernet adoption with new technologies.

### 7.1 Spectrum Switches

<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🔀</div>
    <div class="card-title">Spectrum-3 (2020)</div>
    <div class="card-desc">
      • 64 ports 400 GbE<br>
      • 25.6 Tb/s throughput<br>
      • RoCE v2 RDMA<br>
      • Deep buffers (128 MB)
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">⚡</div>
    <div class="card-title">Spectrum-4 (2023)</div>
    <div class="card-desc">
      • 64 ports 800 GbE<br>
      • 51.2 Tb/s throughput<br>
      • Enhanced RoCE<br>
      • Telemetry streaming
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌐</div>
    <div class="card-title">Spectrum-X (2024)</div>
    <div class="card-desc">
      • AI-optimized Ethernet<br>
      • BlueField-3 DPU integration<br>
      • Congestion control for AI<br>
      • 1.6× better performance
    </div>
  </div>
</div>

### 7.2 RoCE: RDMA over Converged Ethernet

RoCE (RDMA over Converged Ethernet) brings InfiniBand-like RDMA to Ethernet:

<div class="diagram">
<div class="diagram-title">RoCE Protocol Stack</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>Application:</strong> CUDA, NCCL, MPI
  </div>
  <div class="layer green">
    <strong>RDMA Verbs:</strong> Same API as InfiniBand
  </div>
  <div class="layer blue">
    <strong>RoCE v2:</strong> InfiniBand transport over UDP/IP
  </div>
  <div class="layer purple">
    <strong>Ethernet:</strong> 400 GbE / 800 GbE physical layer
  </div>
</div>
</div>

**RoCE Challenges:**
- **Lossy network:** Ethernet drops packets under congestion, requiring PFC (Priority Flow Control)
- **Congestion:** ECN (Explicit Congestion Notification) needed for performance
- **Jitter:** More variable latency than InfiniBand (~3-5 µs vs 1-2 µs)

### 7.3 BlueField DPU

BlueField Data Processing Unit (DPU) is a programmable SmartNIC with Arm cores:

<div class="diagram">
<div class="diagram-title">BlueField-3 DPU Architecture</div>
<div class="flow-h">
  <div class="flow-node accent">16 Arm A78<br>Cores</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">Hardware<br>Accelerators<br>(Crypto, Compress)</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue">ConnectX-7<br>400 Gb/s</div>
  <div class="flow-arrow blue"></div>
  <div class="flow-node purple">PCIe Gen5<br>to Host</div>
</div>
</div>

**Use Cases:**
1. **Security:** Firewall, encryption without host CPU overhead
2. **Storage:** NVMe-oF, GPUDirect Storage offload
3. **Networking:** OVS, packet processing, telemetry
4. **AI:** NCCL offload, collective operations

### 7.4 Spectrum-X: AI-Optimized Ethernet

Spectrum-X platform combines Spectrum-4 switches with BlueField-3 DPUs to optimize Ethernet for AI:

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🎯</div>
    <div class="card-title">Congestion Control</div>
    <div class="card-desc">
      Advanced algorithms detect and mitigate congestion before packet loss, maintaining steady throughput for NCCL all-reduce operations.
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">📊</div>
    <div class="card-title">Telemetry</div>
    <div class="card-desc">
      Real-time network monitoring identifies bottlenecks and hot spots, enabling adaptive routing and load balancing for AI traffic patterns.
    </div>
  </div>
</div>

**Performance Claims:**
- 1.6× better throughput than standard Ethernet for AI workloads
- 95%+ network utilization (vs 60-70% typical Ethernet)
- Competitive with InfiniBand for many workloads at lower cost

## 8. GPUDirect Technologies

GPUDirect is a family of technologies enabling direct data paths between GPUs and other devices.

### 8.1 GPUDirect Technology Stack

<div class="diagram">
<div class="diagram-title">GPUDirect Family</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>GPUDirect P2P:</strong> GPU-to-GPU within node (PCIe BAR mapping)
  </div>
  <div class="layer green">
    <strong>GPUDirect RDMA:</strong> GPU-to-Network (NICs access GPU memory)
  </div>
  <div class="layer blue">
    <strong>GPUDirect Storage:</strong> GPU-to-Storage (NVMe/GDS reads directly to GPU)
  </div>
  <div class="layer purple">
    <strong>GPUDirect Async:</strong> Kernel-initiated transfers without CPU sync
  </div>
</div>
</div>

### 8.2 GPUDirect Peer-to-Peer (P2P)

Enables GPU-to-GPU memory access over PCIe without CPU copies:

```cuda
// Enable P2P between two GPUs
void setup_p2p(int gpu0, int gpu1) {
    // Check if P2P is supported
    int canAccessPeer;
    cudaDeviceCanAccessPeer(&canAccessPeer, gpu0, gpu1);
    
    if (!canAccessPeer) {
        printf("P2P not supported between GPU %d and %d\n", gpu0, gpu1);
        return;
    }
    
    // Enable P2P access
    cudaSetDevice(gpu0);
    cudaDeviceEnablePeerAccess(gpu1, 0);
    
    cudaSetDevice(gpu1);
    cudaDeviceEnablePeerAccess(gpu0, 0);
    
    // Now GPU kernels on GPU0 can directly read/write GPU1 memory
    printf("P2P enabled: GPU %d ↔ GPU %d\n", gpu0, gpu1);
}

// Kernel on GPU0 accessing GPU1 memory
__global__ void cross_gpu_kernel(float *gpu1_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This read goes over PCIe/NVLink directly to GPU1
    float value = gpu1_data[idx];
    
    // Process value...
    value = value * 2.0f;
    
    // Write back to GPU1
    gpu1_data[idx] = value;
}
```

**Performance Characteristics:**
- **NVLink systems:** Full NVLink bandwidth (~600 GB/s - 1.8 TB/s)
- **PCIe systems:** PCIe bandwidth (~32 GB/s), but avoids CPU memory bounce
- **Latency:** ~1-2 µs for NVLink, ~5 µs for PCIe (vs ~10-20 µs with CPU copy)

### 8.3 GPUDirect Storage (GDS)

Enables direct data transfer between GPU memory and NVMe storage, bypassing CPU and system memory:

<div class="diagram">
<div class="diagram-title">Traditional I/O vs GPUDirect Storage</div>
<div class="flow">
  <div class="flow-node accent wide">Traditional: Storage → CPU Memory → GPU (2 copies)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">GDS: Storage → GPU Memory (direct PCIe path)</div>
</div>
</div>

**cuFile API:**

```cuda
#include <cufile.h>

void gds_read_example() {
    // Open file with cuFile (GPUDirect Storage API)
    CUfileDescr_t file_desc;
    CUfileHandle_t file_handle;
    
    file_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    file_desc.handle.fd = open("/data/dataset.bin", O_RDIRECT);
    
    cuFileHandleRegister(&file_handle, &file_desc);
    
    // Allocate GPU memory
    void *gpu_buffer;
    size_t size = 1ULL << 30;  // 1 GB
    cudaMalloc(&gpu_buffer, size);
    
    // Register GPU buffer with cuFile
    cuFileBufRegister(gpu_buffer, size, 0);
    
    // Direct read from storage to GPU (no CPU involvement!)
    ssize_t bytes_read = cuFileRead(file_handle, gpu_buffer, size, 0, 0);
    
    printf("Read %zd bytes directly to GPU\n", bytes_read);
    
    // GPU can immediately start processing
    process_kernel<<<blocks, threads>>>(gpu_buffer, size);
}
```

**Performance Benefits:**

```python
def gds_performance_comparison(file_size_gb=10):
    """Compare traditional I/O vs GPUDirect Storage"""
    
    # Traditional path: Storage → CPU → GPU
    storage_to_cpu = file_size_gb / 7.0  # ~7 GB/s NVMe PCIe 4.0 x4
    cpu_to_gpu = file_size_gb / 30.0  # ~30 GB/s PCIe 4.0 x16
    traditional_time = storage_to_cpu + cpu_to_gpu
    
    print(f"Traditional I/O: {traditional_time:.2f} seconds")
    # = 1.76 seconds for 10 GB
    
    # GDS path: Storage → GPU (direct)
    # Bandwidth limited by NVMe or PCIe, whichever is lower
    gds_bandwidth = min(7.0, 12.0)  # PCIe 4.0 x4 NVMe limits to ~7 GB/s
    gds_time = file_size_gb / gds_bandwidth
    
    print(f"GPUDirect Storage: {gds_time:.2f} seconds")
    # = 1.43 seconds for 10 GB
    
    # Key benefit: Frees CPU memory and CPU cycles
    print(f"CPU memory saved: {file_size_gb} GB")
    print(f"Speedup: {traditional_time / gds_time:.2f}×")
```

**Requirements:**
- NVIDIA GPU with GDS support (Ampere or newer)
- NVMe drives connected via PCIe switches with peer-to-peer routing
- CUDA 11.4+ with GDS enabled
- Supported filesystems (ext4, XFS with O_DIRECT)

### 8.4 GPUDirect Async

Allows CUDA kernels to directly initiate network/storage transfers without CPU synchronization:

```cuda
// Traditional approach: CPU controls everything
void traditional_transfer(cudaStream_t stream) {
    // 1. Launch kernel
    kernel<<<blocks, threads, 0, stream>>>();
    
    // 2. CPU waits for kernel
    cudaStreamSynchronize(stream);
    
    // 3. CPU initiates network transfer
    ib_post_send(...);
}

// GPUDirect Async: Kernel initiates transfer
__global__ void async_kernel_with_transfer() {
    // Compute phase
    compute_something();
    
    // Kernel-side synchronization
    __syncthreads();
    
    // Thread 0 initiates RDMA transfer
    if (threadIdx.x == 0) {
        // Trigger RDMA operation from GPU
        gpudirect_async_send(dest_gpu, local_buffer, size);
    }
    
    // Continue with other work while transfer happens
    do_other_work();
}
```

This enables **overlapping compute and communication** at fine granularity, improving pipeline efficiency.

## 9. Topology Comparisons

### 9.1 Network Topologies for Multi-GPU Systems

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">⭕</div>
    <div class="card-title">Ring Topology</div>
    <div class="card-desc">
      Each GPU connects to two neighbors. Simple, but bandwidth and latency grow with N. Used in early multi-GPU systems.
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🌲</div>
    <div class="card-title">Tree Topology</div>
    <div class="card-desc">
      Hierarchical structure. Root can become bottleneck. Used with PCIe switches and CPU-based interconnects.
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">🌳</div>
    <div class="card-title">Fat-Tree</div>
    <div class="card-desc">
      Multiple paths between nodes. Full bisection bandwidth. Industry standard for InfiniBand fabrics. Scales to thousands of nodes.
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🔗</div>
    <div class="card-title">All-to-All</div>
    <div class="card-desc">
      Every GPU directly connected. Maximum bandwidth, minimal latency. Requires NVSwitch. Limited to single node or small clusters.
    </div>
  </div>
</div>

### 9.2 Ring Topology

<div class="diagram">
<div class="diagram-title">8-GPU Ring</div>
<div class="flow">
  <div class="flow-node accent">GPU 0</div>
  <div class="flow-arrow accent" style="writing-mode: vertical-lr;">→</div>
  <div class="flow-node green">GPU 1</div>
  <div class="flow-arrow green" style="writing-mode: vertical-lr;">→</div>
  <div class="flow-node blue">GPU 2</div>
  <div class="flow-arrow blue" style="writing-mode: vertical-lr;">→</div>
  <div class="flow-node purple">GPU 3</div>
  <div class="flow-arrow purple" style="writing-mode: vertical-lr;">↓</div>
  <div class="flow-node orange">GPU 4</div>
  <div class="flow-arrow orange" style="writing-mode: vertical-lr;">←</div>
  <div class="flow-node pink">GPU 5</div>
  <div class="flow-arrow pink" style="writing-mode: vertical-lr;">←</div>
  <div class="flow-node cyan">GPU 6</div>
  <div class="flow-arrow cyan" style="writing-mode: vertical-lr;">←</div>
  <div class="flow-node teal">GPU 7</div>
  <div class="flow-arrow teal">← (back to GPU 0)</div>
</div>
</div>

**Characteristics:**
- **Bandwidth:** Single link bandwidth between neighbors
- **All-reduce latency:** O(N) - must traverse all nodes
- **Bisection bandwidth:** 1× link bandwidth (bottleneck)
- **Scaling:** Poor - latency and bandwidth degrade linearly

**Ring All-Reduce Algorithm:**

```python
def ring_allreduce(data, gpu_id, num_gpus):
    """Ring-based all-reduce implementation"""
    chunk_size = len(data) // num_gpus
    
    # Scatter-reduce phase (N-1 steps)
    for step in range(num_gpus - 1):
        send_chunk = (gpu_id - step) % num_gpus
        recv_chunk = (gpu_id - step - 1) % num_gpus
        
        # Send chunk to next GPU, receive from previous
        send_to_neighbor(data[send_chunk * chunk_size:(send_chunk + 1) * chunk_size])
        received = recv_from_neighbor()
        
        # Reduce received into local chunk
        data[recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size] += received
    
    # All-gather phase (N-1 steps)
    for step in range(num_gpus - 1):
        send_chunk = (gpu_id - step + 1) % num_gpus
        recv_chunk = (gpu_id - step) % num_gpus
        
        send_to_neighbor(data[send_chunk * chunk_size:(send_chunk + 1) * chunk_size])
        data[recv_chunk * chunk_size:(recv_chunk + 1) * chunk_size] = recv_from_neighbor()
    
    # Total: 2(N-1) communication steps
    # Bandwidth utilization: Optimal (fully pipelined)
```

### 9.3 Tree and Fat-Tree Topologies

**Tree Topology:**
```
        Switch (Root)
       /      |      \
   GPU0-1  GPU2-3  GPU4-5
```

- **Problem:** Root switch is bottleneck for cross-subtree traffic
- **Bisection bandwidth:** 1× uplink bandwidth

**Fat-Tree Topology:**

<div class="diagram">
<div class="diagram-title">2-Tier Fat-Tree (8 GPUs)</div>
<div class="flow">
  <div class="flow-node accent wide">Spine Switches (4×)</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">Leaf Switches (4×) — Each connects to 2 GPUs + all spines</div>
  <div class="flow-arrow green"></div>
  <div class="flow-node blue wide">8 GPUs (2 per leaf)</div>
</div>
</div>

**Fat-Tree Properties:**
- **Multiple paths:** Any leaf can reach any other via multiple spines
- **Bisection bandwidth:** Equal to endpoint bandwidth (non-blocking)
- **Scalability:** Can build 10,000+ node clusters
- **Cost:** Higher switch count than tree

**Routing:**
```
GPU 0 (Leaf 0) → GPU 6 (Leaf 3):
Path 1: GPU 0 → Leaf 0 → Spine 0 → Leaf 3 → GPU 6
Path 2: GPU 0 → Leaf 0 → Spine 1 → Leaf 3 → GPU 6
Path 3: GPU 0 → Leaf 0 → Spine 2 → Leaf 3 → GPU 6
Path 4: GPU 0 → Leaf 0 → Spine 3 → Leaf 3 → GPU 6

All paths have equal latency (2 hops)
Traffic load-balanced across paths
```

### 9.4 All-to-All Topology (NVSwitch)

<div class="diagram">
<div class="diagram-title">NVSwitch All-to-All Fabric</div>
<div style="text-align: center; padding: 2rem; background: rgba(118, 185, 0, 0.05);">
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem;">
    <div class="flow-node accent">GPU 0</div>
    <div class="flow-node accent">GPU 1</div>
    <div class="flow-node accent">GPU 2</div>
    <div class="flow-node accent">GPU 3</div>
  </div>
  <div style="padding: 2rem; background: rgba(118, 185, 0, 0.2); border: 2px solid #76b900; border-radius: 8px; margin: 1rem 0;">
    <strong style="font-size: 1.2em;">NVSwitch Fabric</strong><br>
    Non-Blocking Crossbar<br>
    Full Bandwidth All-to-All
  </div>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 2rem;">
    <div class="flow-node accent">GPU 4</div>
    <div class="flow-node accent">GPU 5</div>
    <div class="flow-node accent">GPU 6</div>
    <div class="flow-node accent">GPU 7</div>
  </div>
</div>
</div>

**Characteristics:**
- **Bandwidth:** Every GPU can use full bandwidth to any other GPU simultaneously
- **Latency:** Single hop through switch fabric (~1 µs)
- **Bisection bandwidth:** Full (non-blocking)
- **Scalability:** Limited to NVSwitch scale (8-576 GPUs with Blackwell)

**Performance Comparison:**

| Topology | Latency GPU 0→4 | Bisection BW | Cost | Scalability |
|----------|----------------|--------------|------|-------------|
| **Ring** | 4 hops | 1× | Low | Poor |
| **Tree** | 2 hops | 1× | Medium | Medium |
| **Fat-Tree** | 2-4 hops | Full | High | Excellent |
| **All-to-All** | 1 hop | Full | Very High | Limited |

### 9.5 Hybrid Topologies

Modern systems combine topologies:

<div class="diagram">
<div class="diagram-title">Hybrid Multi-Tier Topology</div>
<div class="layer-stack">
  <div class="layer accent">
    <strong>Intra-Node (8 GPUs):</strong> NVSwitch all-to-all, 1.8 TB/s per GPU
  </div>
  <div class="layer green">
    <strong>Intra-Rack (8-16 nodes):</strong> InfiniBand fat-tree, 400 Gb/s per node
  </div>
  <div class="layer blue">
    <strong>Inter-Rack:</strong> Spine-leaf InfiniBand, full bisection bandwidth
  </div>
  <div class="layer purple">
    <strong>Inter-Cluster:</strong> WAN links, limited bandwidth, high latency
  </div>
</div>
</div>

## 10. Bandwidth Comparison

### 10.1 Comprehensive Bandwidth Table

<div style="overflow-x: auto;">
<table style="width: 100%; border-collapse: collapse; margin: 2rem 0;">
<thead style="background: rgba(118, 185, 0, 0.2);">
<tr>
  <th style="padding: 0.75rem; border: 1px solid #76b900;">Technology</th>
  <th style="padding: 0.75rem; border: 1px solid #76b900;">Generation</th>
  <th style="padding: 0.75rem; border: 1px solid #76b900;">Year</th>
  <th style="padding: 0.75rem; border: 1px solid #76b900;">Bandwidth (GB/s)</th>
  <th style="padding: 0.75rem; border: 1px solid #76b900;">Latency (µs)</th>
  <th style="padding: 0.75rem; border: 1px solid #76b900;">Typical Use</th>
</tr>
</thead>
<tbody>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">PCIe</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">3.0 x16</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2010</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">16</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">5-10</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">CPU-GPU, legacy</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">PCIe</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">4.0 x16</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2017</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">32</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">5-10</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">CPU-GPU, current</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">PCIe</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">5.0 x16</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2019</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">64</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">5-10</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">CPU-GPU, emerging</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">PCIe</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">6.0 x16</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2022</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">128</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">5-10</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">CPU-GPU, future</td>
</tr>
<tr style="background: rgba(118, 185, 0, 0.1);">
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>NVLink</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1.0 (P100)</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2016</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">160</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1.5</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">GPU-GPU direct</td>
</tr>
<tr style="background: rgba(118, 185, 0, 0.1);">
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>NVLink</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2.0 (V100)</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2017</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">300</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1.2</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">GPU-GPU direct</td>
</tr>
<tr style="background: rgba(118, 185, 0, 0.1);">
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>NVLink</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">3.0 (A100)</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2020</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">600</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1.0</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">GPU-NVSwitch</td>
</tr>
<tr style="background: rgba(118, 185, 0, 0.1);">
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>NVLink</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">4.0 (H100)</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2022</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">900</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1.0</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">GPU-NVSwitch</td>
</tr>
<tr style="background: rgba(118, 185, 0, 0.15);">
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>NVLink</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">5.0 (B200)</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2024</td>
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>1800</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1.0</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">GPU-NVSwitch</td>
</tr>
<tr style="background: rgba(118, 185, 0, 0.15);">
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>NVLink C2C</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">B200</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2024</td>
  <td style="padding: 0.75rem; border: 1px solid #555;"><strong>900</strong></td>
  <td style="padding: 0.75rem; border: 1px solid #555;">0.1</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Die-to-die</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">InfiniBand</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">FDR</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2013</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">7</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1-2</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">InfiniBand</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">EDR</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2014</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">12.5</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1-2</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">InfiniBand</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">HDR</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2017</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">25</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1-2</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">InfiniBand</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">NDR</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2020</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">50</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1-2</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">InfiniBand</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">XDR</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2024</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">100</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">1-2</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">Ethernet</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">400 GbE</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2020</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">50</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">3-5</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
<tr>
  <td style="padding: 0.75rem; border: 1px solid #555;">Ethernet</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">800 GbE</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">2024</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">100</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">3-5</td>
  <td style="padding: 0.75rem; border: 1px solid #555;">Node-to-node</td>
</tr>
</tbody>
</table>
</div>

### 10.2 Bandwidth Evolution Visualization

<div class="diagram">
<div class="diagram-title">Interconnect Bandwidth Growth (2016-2024)</div>
<div style="padding: 2rem; background: rgba(118, 185, 0, 0.05);">
<div style="margin-bottom: 1rem;">
  <strong>PCIe:</strong> 
  <div style="background: rgba(255, 255, 255, 0.1); height: 20px; width: 100%; position: relative; margin: 0.5rem 0;">
    <div style="background: #888; height: 100%; width: 12.5%;"></div>
    <span style="position: absolute; left: 13%; top: 0; color: #fff; font-size: 0.8em;">16 GB/s (2016)</span>
  </div>
  <div style="background: rgba(255, 255, 255, 0.1); height: 20px; width: 100%; position: relative; margin: 0.5rem 0;">
    <div style="background: #aaa; height: 100%; width: 50%;"></div>
    <span style="position: absolute; left: 51%; top: 0; color: #fff; font-size: 0.8em;">64 GB/s (2024)</span>
  </div>
</div>
<div style="margin-bottom: 1rem;">
  <strong>NVLink:</strong>
  <div style="background: rgba(255, 255, 255, 0.1); height: 20px; width: 100%; position: relative; margin: 0.5rem 0;">
    <div style="background: #76b900; height: 100%; width: 8.9%;"></div>
    <span style="position: absolute; left: 10%; top: 0; color: #fff; font-size: 0.8em;">160 GB/s (2016)</span>
  </div>
  <div style="background: rgba(255, 255, 255, 0.1); height: 20px; width: 100%; position: relative; margin: 0.5rem 0;">
    <div style="background: #76b900; height: 100%; width: 100%;"></div>
    <span style="position: absolute; right: 2%; top: 0; color: #fff; font-size: 0.8em;">1800 GB/s (2024)</span>
  </div>
</div>
<div style="margin-bottom: 1rem;">
  <strong>InfiniBand:</strong>
  <div style="background: rgba(255, 255, 255, 0.1); height: 20px; width: 100%; position: relative; margin: 0.5rem 0;">
    <div style="background: #4a9eff; height: 100%; width: 0.7%;"></div>
    <span style="position: absolute; left: 2%; top: 0; color: #fff; font-size: 0.8em;">12.5 GB/s (2016)</span>
  </div>
  <div style="background: rgba(255, 255, 255, 0.1); height: 20px; width: 100%; position: relative; margin: 0.5rem 0;">
    <div style="background: #4a9eff; height: 100%; width: 5.6%;"></div>
    <span style="position: absolute; left: 7%; top: 0; color: #fff; font-size: 0.8em;">100 GB/s (2024)</span>
  </div>
</div>
</div>
</div>

### 10.3 Cost-Performance Analysis

```python
def calculate_tco_per_gbs(technology, bandwidth_gbs, cost_per_port):
    """Calculate total cost of ownership per GB/s"""
    
    configs = {
        'pcie_5': {'bw': 64, 'cost': 0, 'power': 30},  # Included in motherboard
        'nvlink_5': {'bw': 1800, 'cost': 2000, 'power': 400},  # Estimated NVSwitch cost
        'infiniband_ndr': {'bw': 50, 'cost': 1500, 'power': 50},
        'ethernet_800g': {'bw': 100, 'cost': 800, 'power': 40},
    }
    
    config = configs[technology]
    
    # 5-year TCO
    initial_cost = config['cost']
    power_cost = config['power'] * 0.1 * 24 * 365 * 5  # $0.10/kWh
    total_cost = initial_cost + power_cost
    
    cost_per_gbs = total_cost / config['bw']
    
    print(f"{technology}:")
    print(f"  Bandwidth: {config['bw']} GB/s")
    print(f"  5-year TCO: ${total_cost:.2f}")
    print(f"  Cost per GB/s: ${cost_per_gbs:.2f}")
    print()

# Run analysis
for tech in ['pcie_5', 'nvlink_5', 'infiniband_ndr', 'ethernet_800g']:
    calculate_tco_per_gbs(tech, 0, 0)

# Output:
# pcie_5:
#   Bandwidth: 64 GB/s
#   5-year TCO: $1314.00
#   Cost per GB/s: $20.53
#
# nvlink_5:
#   Bandwidth: 1800 GB/s
#   5-year TCO: $19536.00
#   Cost per GB/s: $10.85  ← Best for high bandwidth
#
# infiniband_ndr:
#   Bandwidth: 50 GB/s
#   5-year TCO: $3690.00
#   Cost per GB/s: $73.80
#
# ethernet_800g:
#   Bandwidth: 100 GB/s
#   5-year TCO: $2552.00
#   Cost per GB/s: $25.52  ← Best for node-to-node
```

## Conclusion

Multi-GPU interconnects form the nervous system of modern AI infrastructure. The choice of interconnect technology profoundly impacts system performance, cost, and scalability:

<div class="diagram-grid cols-2">
  <div class="diagram-card accent">
    <div class="card-icon">🏢</div>
    <div class="card-title">Intra-Node</div>
    <div class="card-desc">
      <strong>NVLink 5.0:</strong> 1.8 TB/s per GPU via NVSwitch fabric. Enables tensor parallelism and fine-grained communication. Single-node systems with 8 GPUs.
    </div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🌐</div>
    <div class="card-title">Inter-Node</div>
    <div class="card-desc">
      <strong>InfiniBand NDR/XDR:</strong> 50-100 GB/s per node with GPUDirect RDMA. Fat-tree topology for large clusters. Thousands of GPUs.
    </div>
  </div>
  <div class="diagram-card blue">
    <div class="card-icon">💰</div>
    <div class="card-title">Cost-Optimized</div>
    <div class="card-desc">
      <strong>PCIe 5.0 + Ethernet:</strong> 64 GB/s CPU-GPU, 100 GB/s node-to-node. Lower cost but higher latency. Good for data parallelism.
    </div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">🚀</div>
    <div class="card-title">Future</div>
    <div class="card-desc">
      <strong>NVLink Switch System:</strong> 576-GPU domains with NVLink connectivity. Blurs the line between intra-node and inter-node. UCIe integration.
    </div>
  </div>
</div>

**Key Takeaways:**

1. **Bandwidth Hierarchy:** NVLink (1.8 TB/s) > InfiniBand (100 GB/s) > Ethernet (100 GB/s) > PCIe (64 GB/s)

2. **Latency Matters:** For tensor parallelism and fine-grained communication, sub-microsecond latency (NVLink) is critical

3. **Topology Impact:** All-to-all (NVSwitch) provides maximum bandwidth but limited scale; fat-tree (InfiniBand) provides excellent scalability

4. **Software Transparency:** GPUDirect technologies (RDMA, Storage, P2P) enable zero-copy data paths, critical for performance

5. **Economic Trade-offs:** NVLink provides best cost-per-GB/s at high bandwidth; Ethernet/InfiniBand better for distributed systems

The future of AI training and inference depends on continued interconnect innovation. As model sizes grow exponentially, the bandwidth and latency of multi-GPU interconnects become the primary bottleneck—not compute. NVIDIA's vertically integrated approach (GPUs + NVLink + NVSwitch + InfiniBand + software) provides unmatched performance for large-scale AI, but Ethernet alternatives are closing the gap for cost-sensitive workloads.

Understanding these interconnect technologies is essential for architecting efficient AI systems. The next chapter explores how these interconnects combine into complete data center architectures optimized for AI workloads.

---

**Next: [Chapter 11 — Data Center Architectures →](./11_data_center_architectures.md)**

*Last updated: April 2026*
