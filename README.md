# Cognitive-Core-Engine-Test

Multi-module architecture integrating fixed cognitive core with invention, governance, and **AGI-oriented capability layers**.

> **Scope Notice:** This is a research prototype demonstrating AGI-relevant cognitive mechanisms — autonomous goal generation, hierarchical abstraction, intrinsic motivation, and governance-gated self-modification — within a simulated environment. The system does **not** claim to achieve AGI. Capability scores are internal proxy metrics, not verified against real-world AGI benchmarks (e.g., ARC-AGI).

## Architecture

### Core Modules

Three-module system with strict separation of concerns:

1. **NON_RSI_AGI_CORE_v5.py** - Fixed cognitive orchestrator (main loop owner)
   - Hyperdimensional Computing (HDC) with 10,000-bit vectors
   - Position-bound encoding with multi-resolution bundling
   - World model: feature-based Q-value estimation with experience replay
   - Planner: beam search over world model (depth=3, width=6)
   - Skill DSL: data-level interpreted programs (call/if/foreach)
   - Multi-agent orchestrator with project graph
   - Intrinsic motivation blending (curiosity + novelty + learning progress)
   - Self-model task skipping and failure diagnosis

2. **omega_forge_two_stage_feedback.py** - Invention plugin (invoked on stagnation)
   - Structural-transition discovery via CFG analysis
   - Virtual machine: 8 registers, 64 memory cells, 21 opcodes
   - Detector: multi-stage control flow novelty (CFG edit distance, SCC analysis)
   - Curriculum: warmup phase with relaxed constraints
   - Crash-safe JSONL evidence logging

3. **unified_rsi_extended.py** - Governance/evaluation gate (critic-only adoption)
   - Pre-filtering and stress-checking of candidates
   - Expandable grammar for invention representation
   - Blackboard JSONL logging for multi-loop coordination

### Capability Modules (`agi_modules/`)

12 modules implementing autonomous goal generation, intrinsic motivation, hierarchical abstraction, cross-domain transfer, self-modeling, and open-ended learning. These modules implement **AGI-adjacent** mechanisms and serve as research scaffolding toward general intelligence — they do not constitute AGI.

| Module | Purpose |
|--------|---------|
| `competence_map.py` | Tracks (domain, difficulty) success rates; identifies zone-of-proximal-development |
| `goal_generator.py` | Autonomous goal creation via frontier expansion, gap remediation, creative exploration |
| `intrinsic_motivation.py` | Curiosity (prediction error), novelty (visit count), learning progress rewards |
| `concept_graph.py` | Hierarchical abstraction: L0 actions -> L1 skills -> L2 strategies -> L3+ meta |
| `hierarchical_planner.py` | Multi-level planning using concept graph, falls back to flat beam search |
| `transfer_engine.py` | Cross-domain knowledge transfer with negative-transfer detection and rollback |
| `self_model.py` | Capability prediction, task skip decisions, failure diagnosis (exploration/knowledge/planning) |
| `difficulty_scheduler.py` | Curriculum learning with chaos injection to escape local optima |
| `self_improvement.py` | Runtime parameter self-modification through governance gate |
| `agi_tracker.py` | 5-axis capability proxy scoring: generalization, autonomy, self-improvement, abstraction, open-endedness |
| `external_benchmark.py` | Held-out validation, overfitting detection, HDC retrieval precision benchmarks |

## Integration

### Original Call Chain

`Orchestrator -> Omega (on stagnation) -> Unified (critic) -> Orchestrator (register/reject)`

- **Contract A (GapSpec)**: Orchestrator -> Omega capability gap specification
- **Contract B (CandidatePacket)**: Omega -> Unified -> Orchestrator artifact + evidence + verdict

### Capability Extension Call Chain

```
Orchestrator.run_recursive_cycle()
  |-- GoalGenerator.generate()        [autonomous task creation]
  |-- Agent.act_on_project()           [intrinsic reward blending]
  |-- CompetenceMap.update()           [competence tracking]
  |-- ConceptGraph.promote()           [abstraction formation]
  |-- TransferEngine.transfer()        [cross-domain knowledge reuse]
  |-- SelfImprovementEngine.introspect() [parameter self-modification]
  |-- DifficultyScheduler.schedule()   [curriculum adjustment]
  |-- AGIProgressTracker.tick_round()  [5-axis proxy measurement]
  `-- ExternalBenchmark.run_adb_snapshot() [held-out validation]
```

No file merging. No self-adoption by invention module. All self-improvements go through governance gate.

## Technical Details

**HDC Memory**: Associative retrieval using bundled hypervectors with:
- Position-bound encoding: `hv = sum(permute(token_hv, position))`
- Multi-resolution bundling (character/bigram/trigram levels)
- Deterministic tie-breaking (no global random perturbation)
- Similarity threshold: 0.51 (random baseline: 0.50)
- Max items: 20,000

**Structural Discovery**: CFG-based novelty detection with:
- Edit distance K (warmup: 3, strict: 6)
- Active subsequence length L (warmup: 8, strict: 10)
- Minimum coverage: 0.55
- Reproducibility: N=4 trials, max CFG variants: 2

**World Model**: TD-learning with:
- Non-linear feature combinations
- Experience replay buffer (200 samples)
- Gamma: 0.9, LR: 0.08
- Combined reward: extrinsic (0.6) + intrinsic (0.4)

**AGI-Proxy Capability Scoring** (geometric mean of 5 axes):
- Generalization: cross-domain transfer success rate
- Autonomy: fraction of self-generated goals
- Self-Improvement: beneficial parameter modification rate
- Abstraction: concept graph depth / target depth
- Open-Endedness: domain growth + difficulty progression rate

> **Caveat:** These scores are internal proxy metrics only. They do not correspond to verified performance on standardized general intelligence benchmarks. Self-improvement uses empirical env rollouts (not arithmetic). HDC precision 0.80 measured without tag filtering. External benchmark connected to agent WorldModel. See RESULTS.md for detailed evidence and honest failure reporting.

## Usage

```bash
# Run selftest suite (core + contract + capability integration tests)
python NON_RSI_AGI_CORE_v5.py selftest

# Run fixed core
python NON_RSI_AGI_CORE_v5.py --rounds 40 --agents 8

# Reproduce baseline evidence logs
python scripts/run_results.py

# Run capability evidence with 3-way ablation comparison
python scripts/run_agi_evidence.py

# Run Omega Forge two-stage pipeline
python omega_forge_two_stage_feedback.py full --stage1_gens 200 --stage2_gens 200 --seed 42
```

## Evidence Summary

50-round capability evidence run (seed=42):

| Configuration | Composite Score | Domains |
|--------------|----------------|---------|
| **Full system (all modules)** | **0.494** | 47 |
| Ablation A (no capability modules) | 0.004 | 6 |
| Ablation B (no GoalGenerator) | 0.040 | 6 |
| Ablation C (no TransferEngine) | 0.192 | 47 |

See [RESULTS.md](RESULTS.md) for full evidence report including external validation, HDC precision benchmarks, and honest failure reporting.

> Note: The critic module is `unified_rsi_extended.py`. The core loader includes a fallback
> for the historical filename with a trailing space, but the canonical filename is the
> space-free version.

## Scope & Limitations

This project is a **research/engineering prototype**, not a claim of AGI achievement.

**What this system demonstrates:**
- Governance-gated architecture with rollback — safe self-modification within fixed source code
- Autonomous goal generation producing tasks beyond the hardcoded set
- Hierarchical concept formation from raw experience (L0 → L3+)
- Intrinsic motivation blending (curiosity, novelty, learning progress)
- Cross-domain transfer learning with negative-transfer detection
- Curriculum learning with chaos injection for local optima escape

**Known limitations & open problems:**
- Concept graph depth reaches 5 via promote_cascade() but depth is partially driven by threshold calibration
- HDC retrieval precision (0.80) passes threshold but relies on domain-specific vocabulary separation
- Transfer analogy uses name-similarity heuristics, not deep structural matching
- Open-endedness score (0.65) uses mastery-fraction scoring but domain creation is still string-label based
- Self-improvement acceptance rate (60%) is empirically gated but limited to 5-episode rollouts
- All environments are simulated; real-world grounding is absent

## Status

Research/engineering prototype. Governance-gated architecture with rollback. Capability modules provide autonomous goal generation, hierarchical abstraction, and self-improvement within fixed-source-code constraints.

## License

MIT License
