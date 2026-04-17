# Pipeline Parallelism (PP)

Inter-layer model parallelism: layers are split across GPUs (stages), and microbatches flow through the pipeline. Communication is `Send`/`Recv` between adjacent stages — cheap and topology-friendly.

## The pipeline bubble

With `p` stages and `m` microbatches, a naive schedule wastes `(p-1)/(m + p - 1)` of the time on "bubble" — stages idle while the pipeline fills and drains. Make `m >> p` to amortize.

## 1F1B and interleaved 1F1B

- **1F1B** (one-forward-one-backward): each stage alternates forward and backward passes on different microbatches once the pipeline is full. Reduces activation memory vs. "all-forward-then-all-backward."
- **Interleaved 1F1B**: each device owns *v* virtual stages (non-contiguous layers). Bubble shrinks to `(p-1)/(v·m + p - 1)`. The cost: more point-to-point traffic.

## When PP helps

- Cross-node parallelism: Send/Recv between stages is cheap over Ethernet/IB.
- Very large models that don't fit even with TP+FSDP inside a node.

## When PP hurts

- Microbatch sizes force you to choose between throughput and memory.
- Hard to compose with sequence-dependent reward signals in RL (one reason RL frameworks often avoid PP and lean on FSDP+TP).

## References

- [[Megatron-LM Paper]]
- [[Ultra-Scale Playbook]]
