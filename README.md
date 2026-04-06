# Cognitive-Core-Engine-Test

Multi-module architecture integrating fixed cognitive core with invention, governance, and AGI capability layers.

## Architecture

### Core Modules

Three-module system with strict separation of concerns:

1. **NON_RSI_AGI_CORE_v5.py** - Fixed orchestrator (main loop owner)
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

### AGI Modules (`agi_modules/`)

12 modules implementing autonomous goal generation, intrinsic motivation, hierarchical abstraction, cross-domain transfer, self-modeling, and open-ended learning:

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
| `agi_tracker.py` | 5-axis AGI scoring: generalization, autonomy, self-improvement, abstraction, open-endedness |
| `external_benchmark.py` | Held-out validation, overfitting detection, HDC retrieval precision benchmarks |

## Integration

### Original Call Chain

`Orchestrator -> Omega (on stagnation) -> Unified (critic) -> Orchestrator (register/reject)`

- **Contract A (GapSpec)**: Orchestrator -> Omega capability gap specification
- **Contract B (CandidatePacket)**: Omega -> Unified -> Orchestrator artifact + evidence + verdict

### AGI Extension Call Chain

```
Orchestrator.run_recursive_cycle()
  |-- GoalGenerator.generate()        [autonomous task creation]
  |-- Agent.act_on_project()           [intrinsic reward blending]
  |-- CompetenceMap.update()           [competence tracking]
  |-- ConceptGraph.promote()           [abstraction formation]
  |-- TransferEngine.transfer()        [cross-domain knowledge reuse]
  |-- SelfImprovementEngine.introspect() [parameter self-modification]
  |-- DifficultyScheduler.schedule()   [curriculum adjustment]
  |-- AGIProgressTracker.tick_round()  [5-axis measurement]
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

**AGI Progress Scoring** (geometric mean of 5 axes):
- Generalization: cross-domain transfer success rate
- Autonomy: fraction of self-generated goals
- Self-Improvement: beneficial parameter modification rate
- Abstraction: concept graph depth / target depth
- Open-Endedness: domain growth + difficulty progression rate

## Usage

```bash
# Run selftest suite (core + contract + AGI integration tests)
python NON_RSI_AGI_CORE_v5.py selftest

# Run fixed core
python NON_RSI_AGI_CORE_v5.py --rounds 40 --agents 8

# Reproduce baseline evidence logs
python scripts/run_results.py

# Run AGI evidence with 3-way ablation comparison
python scripts/run_agi_evidence.py

# Run Omega Forge two-stage pipeline
python omega_forge_two_stage_feedback.py full --stage1_gens 200 --stage2_gens 200 --seed 42
```

## Evidence Summary

50-round AGI evidence run (seed=42):

| Configuration | Composite Score | Domains |
|--------------|----------------|---------|
| **Full AGI system** | **0.500** | 42 |
| Ablation A (no AGI modules) | 0.003 | 6 |
| Ablation B (no GoalGenerator) | 0.034 | 6 |
| Ablation C (no TransferEngine) | 0.169 | 42 |

See [RESULTS.md](RESULTS.md) for full evidence report including external validation, HDC precision benchmarks, and honest failure reporting.

> Note: The critic module is `unified_rsi_extended.py`. The core loader includes a fallback
> for the historical filename with a trailing space, but the canonical filename is the
> space-free version.

## Status

Research/engineering hybrid. Governance-gated architecture with rollback. AGI modules provide autonomous goal generation, hierarchical abstraction, and self-improvement within fixed-source-code constraints.

## License

MIT License
