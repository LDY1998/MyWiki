# MyWiki — Index

Top-level index for the personal knowledge base. Each subdomain maintains its own `raw/` (sources) and `wiki/` (curated pages).

## Subdomains

- **[[AI/wiki/index|AI]]** — LLM post-training, RL algorithms (GRPO, DAPO, PPO, DPO), related papers and concepts.
- **[[Infra/wiki/index|Infra]]** — AI infrastructure for large-scale RL, robotics, and world models. Distributed training, simulation, RL orchestration, world foundation models, VLAs, serving.

## Pattern

See `AI/raw/LLM Wiki.md` for the system description. In short:
- `raw/` holds one markdown file per external source (papers, blog posts, docs) with YAML frontmatter and a summary.
- `wiki/concepts/` and `wiki/papers/` hold curated, interlinked concept and paper pages.
- Each subdomain has an `index.md` cataloguing its contents.
