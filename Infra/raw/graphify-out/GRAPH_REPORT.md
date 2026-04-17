# Graph Report - Infra/raw/  (2026-04-17)

## Corpus Check
- Corpus is ~2,679 words - fits in a single context window. You may not need a graph.

## Summary
- 58 nodes · 80 edges · 9 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_FSDP & Sharding|FSDP & Sharding]]
- [[_COMMUNITY_Megatron Pipeline Parallelism|Megatron Pipeline Parallelism]]
- [[_COMMUNITY_NCCL Collectives|NCCL Collectives]]
- [[_COMMUNITY_Transformer Architecture|Transformer Architecture]]
- [[_COMMUNITY_Compute & Memory Math|Compute & Memory Math]]
- [[_COMMUNITY_Inference Optimization|Inference Optimization]]
- [[_COMMUNITY_Operational ML Engineering|Operational ML Engineering]]
- [[_COMMUNITY_JAX Roofline & Mesh|JAX Roofline & Mesh]]
- [[_COMMUNITY_KV Cache|KV Cache]]

## God Nodes (most connected - your core abstractions)
1. `The Ultra-Scale Playbook` - 12 edges
2. `ZeRO: Memory Optimizations Toward Training Trillion Parameter Models` - 10 edges
3. `Transformer Math 101` - 8 edges
4. `Large Transformer Model Inference Optimization` - 7 edges
5. `FSDP2 (fully_shard API with per-parameter DTensor sharding)` - 7 edges
6. `Megatron-LM: Efficient Large-Scale Language Model Training on GPU Clusters` - 7 edges
7. `ML Engineering — Stas Bekman (Open Book)` - 6 edges
8. `NCCL User Guide — Collective Operations` - 6 edges
9. `The Illustrated Transformer` - 6 edges
10. `Tensor Parallelism` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Speculative Decoding` --semantically_similar_to--> `How to Scale Your Model (JAX Scaling Book)`  [INFERRED] [semantically similar]
  Lilian Weng - Inference Optimization.md → JAX Scaling Book.md
- `Tensor Mesh Sharding (JAX shard_map)` --semantically_similar_to--> `Tensor Parallelism`  [INFERRED] [semantically similar]
  JAX Scaling Book.md → Ultra-Scale Playbook.md
- `Hardware Topology (GPU, NVLink, PCIe, InfiniBand)` --semantically_similar_to--> `NCCL User Guide — Collective Operations`  [INFERRED] [semantically similar]
  ML Engineering Book.md → NCCL User Guide.md
- `1F1B Pipeline Schedule (interleaved, virtual stages)` --semantically_similar_to--> `Pipeline Parallelism`  [INFERRED] [semantically similar]
  Megatron-LM Paper.md → Ultra-Scale Playbook.md
- `Multi-Head Attention` --semantically_similar_to--> `Tensor Parallelism Column-then-Row Matmul Splitting`  [INFERRED] [semantically similar]
  Illustrated Transformer.md → Megatron-LM Paper.md

## Hyperedges (group relationships)
- **PTD-P Parallelism Composition: Tensor Parallelism (intra-node NVLink), Pipeline Parallelism (inter-node), Data Parallelism / ZeRO (across all) compose to enable trillion-parameter training** — megatron_lm_ptdp_parallelism, ultra_scale_tensor_parallelism, ultra_scale_pipeline_parallelism, ultra_scale_data_parallelism, zero_paper, nccl_user_guide [EXTRACTED 1.00]
- **FSDP Training Flow: AllGather params on forward, ReduceScatter grads on backward, implementing ZeRO-3 gather-just-in-time pattern via NCCL** — pytorch_fsdp_fsdp2, zero_paper_zero_stage3, nccl_allgather, nccl_reducescatter [EXTRACTED 1.00]
- **Inference Optimization Stack: KV cache sizing (Transformer Math formula), quantization, FlashAttention, and speculative decoding together determine serving feasibility on a given GPU** — lilian_weng_kv_cache, transformer_math_kv_cache_formula, lilian_weng_quantization, lilian_weng_flashattention, lilian_weng_speculative_decoding [INFERRED 0.80]

## Communities

### Community 0 - "FSDP & Sharding"
Cohesion: 0.2
Nodes (12): Activation Checkpointing (via apply_activation_checkpointing), FSDP2 (fully_shard API with per-parameter DTensor sharding), Rationale: Read ZeRO paper before FSDP tutorial — API is trivial but concepts are not; ZeRO mental model clarifies FSDP wrap granularity and precision choices, Getting Started with Fully Sharded Data Parallel (FSDP / FSDP2), Pipeline Parallelism, Tensor Parallelism, ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, Rationale: ZeRO-3 1.5x communication overhead — why real clusters combine ZeRO with TP and PP rather than ZeRO-3 across all GPUs (+4 more)

### Community 1 - "Megatron Pipeline Parallelism"
Cohesion: 0.2
Nodes (11): 1F1B Pipeline Schedule (interleaved, virtual stages), Megatron-LM: Efficient Large-Scale Language Model Training on GPU Clusters, Pipeline Bubble Fraction Formula: (p-1)/(v*m), PTD-P Parallelism (Tensor + Pipeline + Data Parallelism composition), Rationale: TP run inside a node (NVLink), PP inter-node, DP on top — because TP requires two all_reduce per layer which is expensive across slower interconnects, TorchTitan, 4D/5D Parallelism Composition, Expert Parallelism (MoE) (+3 more)

### Community 2 - "NCCL Collectives"
Cohesion: 0.29
Nodes (8): Tensor Parallelism Column-then-Row Matmul Splitting, AllGather (NCCL collective), AllReduce (NCCL collective), Rationale: Mapping framework API calls to exact collectives enables performance reasoning, ReduceScatter (NCCL collective), Ring All-Reduce Algorithm, NCCL User Guide — Collective Operations, Data Parallelism (DDP)

### Community 3 - "Transformer Architecture"
Cohesion: 0.33
Nodes (6): The Illustrated Transformer, Scaled Dot-Product Attention (Q, K, V), Encoder/Decoder Stack with Cross-Attention, Multi-Head Attention, Rationale: Fuzzy attention shape mental model causes every TP/SP/FSDP concept to slide off — 30 minutes pays for itself many times over, FlashAttention

### Community 4 - "Compute & Memory Math"
Cohesion: 0.4
Nodes (5): Transformer Math 101, Chinchilla Scaling Laws, 6ND FLOPs Approximation for Transformer Training, Training Memory Formula (16-20 bytes/param rule), Rationale: Closed-form formulas enable predict peak memory to ~15% accuracy before launching a run

### Community 5 - "Inference Optimization"
Cohesion: 0.4
Nodes (5): Batching Strategies (static, dynamic, continuous/in-flight), Large Transformer Model Inference Optimization, Quantization (Post-Training: int8/int4, GPTQ, AWQ, SmoothQuant), Rationale: Inference memory math as warm-up for training math, Speculative Decoding

### Community 6 - "Operational ML Engineering"
Cohesion: 0.4
Nodes (5): ML Engineering — Stas Bekman (Open Book), Checkpointing at Scale, Debugging Distributed Training (NaN losses, hangs, NCCL timeouts), Hardware Topology (GPU, NVLink, PCIe, InfiniBand), Rationale: ML Engineering covers the 'everything's on fire' path complementing Ultra-Scale Playbook's happy path

### Community 7 - "JAX Roofline & Mesh"
Cohesion: 0.5
Nodes (4): How to Scale Your Model (JAX Scaling Book), Rationale: JAX Scaling Book gives math-flavored (mesh) perspective complementing Ultra-Scale Playbook's framework-flavored (PyTorch) perspective — same ideas in two idioms, Roofline Analysis (compute-bound vs. memory-bound vs. communication-bound), Tensor Mesh Sharding (JAX shard_map)

### Community 8 - "KV Cache"
Cohesion: 1.0
Nodes (2): KV Cache, KV Cache Sizing Formula (2 × d_model × bytes_per_element per token per layer)

## Knowledge Gaps
- **23 isolated node(s):** `Quantization (Post-Training: int8/int4, GPTQ, AWQ, SmoothQuant)`, `Batching Strategies (static, dynamic, continuous/in-flight)`, `Rationale: Inference memory math as warm-up for training math`, `Sequence / Context Parallelism (Ring Attention)`, `Expert Parallelism (MoE)` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `KV Cache`** (2 nodes): `KV Cache`, `KV Cache Sizing Formula (2 × d_model × bytes_per_element per token per layer)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `The Ultra-Scale Playbook` connect `Megatron Pipeline Parallelism` to `FSDP & Sharding`, `NCCL Collectives`, `Compute & Memory Math`, `Operational ML Engineering`, `JAX Roofline & Mesh`?**
  _High betweenness centrality (0.367) - this node is a cross-community bridge._
- **Why does `Transformer Math 101` connect `Compute & Memory Math` to `FSDP & Sharding`, `Megatron Pipeline Parallelism`, `Inference Optimization`, `JAX Roofline & Mesh`, `KV Cache`?**
  _High betweenness centrality (0.306) - this node is a cross-community bridge._
- **Why does `ZeRO: Memory Optimizations Toward Training Trillion Parameter Models` connect `FSDP & Sharding` to `Megatron Pipeline Parallelism`, `NCCL Collectives`, `Compute & Memory Math`?**
  _High betweenness centrality (0.296) - this node is a cross-community bridge._
- **What connects `Quantization (Post-Training: int8/int4, GPTQ, AWQ, SmoothQuant)`, `Batching Strategies (static, dynamic, continuous/in-flight)`, `Rationale: Inference memory math as warm-up for training math` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._