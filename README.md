# Cognitive-Core-Engine-Test

Multi-module architecture integrating fixed cognitive core with invention, governance, and capability layers — featuring a self-referential meta-simulation model with anti-wireheading defense.

> **Scope Notice:** This is a research prototype demonstrating cognitive mechanisms — autonomous goal generation, hierarchical abstraction, intrinsic motivation, recursive self-modeling, and governance-gated self-modification — within a simulated environment.

## Project Structure

```
cognitive_core_engine/
  core/                     # Fixed cognitive orchestrator
    utils.py                  stable_hash, now_ms, tokenize
    hdc.py                    HyperVector (10,000-bit HDC)
    memory.py                 MemoryItem, SharedMemory
    tools.py                  ToolRegistry, tool factories
    skills.py                 SkillStep, Skill, SkillLibrary
    world_model.py            TransitionSummary, WorldModel (TD learning)
    planner.py                PlanCandidate, Planner (beam search)
    project_graph.py          ProjectNode, ProjectGraph
    environment.py            RuleProposal, TaskSpec, ResearchEnvironment
    agent.py                  AgentConfig, Agent (B-type architecture)
    orchestrator.py           OrchestratorConfig, Orchestrator (C-layer)
  omega_forge/              # Invention plugin (invoked on stagnation)
    instructions.py           OPS, Instruction, ProgramGenome, ExecutionState
    cfg.py                    ControlFlowGraph
    vm.py                     VirtualMachine, MacroLibrary
    concepts.py               Concept, ConceptLibrary, rand_inst
    benchmark.py              TaskBenchmark, DetectorParams, StrictStructuralDetector
    evidence.py               EvidenceWriter, EngineConfig
    engine.py                 OmegaForgeV13
    stage1.py                 Stage1Engine, TaskBenchmarkV4, ConceptDiscoveryBenchmark
    stage2.py                 Stage2Engine, feedback functions
    rsi_pipeline.py           RSISkillRegistrar — OmegaForge → SkillLibrary pipeline (BN-02)
    cli.py                    CLI commands and entry points
  governance/               # Evaluation gate and meta-control
    utils.py                  now_ms, sha256, clamp, safe_mkdir, etc.
    critic.py                 critic_evaluate_candidate_packet, RunLogger
    invention.py              Invention system (18 classes)
    sandbox.py                SAFE_BUILTINS, validators, safe_exec, safe_eval
    engine_types.py           EngineStrategy, TaskSpec, Genome, Batch, evaluate()
    evolution.py              Mutation operators, OPERATORS dict, mutate_learner
    meta.py                   SurrogateModel, MAPElitesArchive, Universe, GlobalState
    autopatch.py              AutoPatch functions, scoring, filtering
    loops.py                  run_duo_loop, run_rsi_loop
    cli.py                    build_parser, cmd_* functions, main()
agi_modules/                # Capability extension modules
  competence_map.py           Zone-of-proximal-development tracking
  goal_generator.py           Autonomous goal creation (frontier/gap/creative)
  intrinsic_motivation.py     Curiosity, novelty, learning progress rewards
  concept_graph.py            Hierarchical abstraction (L0-L5)
  hierarchical_planner.py     Multi-level planning via concept graph
  transfer_engine.py          Cross-domain transfer with HDC structural matching (BN-04)
  self_model.py               Legacy capability prediction (backward compat)
  self_referential_model.py   Advanced self-referential meta-simulation
  difficulty_scheduler.py     Curriculum learning with chaos injection
  self_improvement.py         Empirical env-rollout parameter tuning
  agi_tracker.py              5-axis capability proxy scoring
  external_benchmark.py       ARC-AGI + HumanEval held-out validation (BN-03)
data/
  arc_agi_sample.json         20 bundled ARC-AGI tasks (BN-03)
  humaneval_sample.json       10 bundled HumanEval problems (BN-03)
tests/
  test_selftest.py            Core selftest + contract negative tests
  test_benchmarks.py          ADB, ARC, program synthesis benchmarks
  test_agi_integration.py     11 integration tests + anti-cheat audit
scripts/
  run_results.py              Reproduce baseline evidence logs
  run_agi_evidence.py         50-round evidence with 3-way ablation
  verify_self_improvement.py  Self-improvement verification suite
main.py                     # Entry point
```

## Architecture

### Core Call Chain

`Orchestrator -> Omega (on stagnation) -> Governance (critic) -> RSISkillRegistrar -> SkillLibrary`

### Self-Referential Meta-Simulation Loop

```
Orchestrator.run_round()
  for each agent:
    |-- SelfReferentialModel.encode_self_referential_state()
    |     Binds env observation + ZPD frontier + concept graph + active skills
    |     into unified 10,000-bit hypervector via XOR binding
    |-- SelfReferentialModel.detect_architectural_drift()
    |     Cosine distance on HDC state history; critical drift -> governance rollback
    |-- Agent.act_on_project()
    |     Meta-rollout predicts BOTH next env state AND agent's own policy shift
    |     Depth ceiling adapts to calibration error (BN-06)
    |
  Orchestrator.run_recursive_cycle()
    |-- validate_metric_integrity()
    |     Anti-wireheading gate: rejects self-improvement claims that lack
    |     structural correlation or external benchmark confirmation
    |     Hash-fallback path REMOVED; holdout_rate is mandatory (BN-05)
    |-- Immutable objective anchor checked (read-only HDC vector)
```

### Capability Extension

```
Orchestrator.run_recursive_cycle()
  |-- GoalGenerator.generate()           [autonomous task creation]
  |-- CompetenceMap.update()             [competence tracking]
  |-- ConceptGraph.sweep_promote_all()   [hierarchical abstraction]
  |-- TransferEngine.transfer()          [HDC structural similarity — BN-04]
  |-- SelfImprovementEngine.introspect() [empirical parameter tuning]
  |-- DifficultyScheduler.schedule()     [curriculum adjustment]
  |-- AGIProgressTracker.tick_round()    [5-axis proxy measurement]
  `-- ExternalBenchmark.run_full_benchmark() [ARC-AGI + HumanEval — BN-03]
```

## Technical Details

**Self-Referential Model** (`self_referential_model.py`):
- HDC state encoding: env hash XOR competence profile XOR concept structure XOR skill set
- Recursive meta-rollout: dual simulation predicting env state AND internal policy shift
- Architectural drift detection: cosine distance on unified state history (threshold 0.35)
- Anti-wireheading: immutable objective anchor + metric integrity validation + decoupled evaluation

**HDC Memory**: Title-weighted position-bound encoding, deterministic tie-breaking, similarity threshold 0.51, max 20,000 items.

**World Model**: TD-learning with non-linear features, experience replay (200 samples), gamma 0.9, combined reward: extrinsic (0.6) + intrinsic (0.4).

**Self-Improvement**: Empirical env.step() rollouts (5 baseline + 5 modified episodes). Anti-wireheading gate rejects modifications exceeding MAX_CREDIBLE_LEAP (0.25) or lacking external benchmark correlation. Hash-fallback scoring path fully removed (BN-05).

**Adaptive Meta-Depth** (BN-06): `_meta_depth_ceiling()` computes allowable rollout depth based on calibration error over the last 8 transitions. Calibration error < 0.05 → depth 4; ≥ 0.30 → depth 1. Theorist/strategist roles receive +1 bonus.

**RSI Skill Registration** (BN-02): `RSISkillRegistrar` compiles approved OmegaForge candidates into `ProgramGenome`, quarantines via 5-input smoke-test (min 3 clean halts), wraps in `_VMSkillCallable`, and registers into `SkillLibrary`.

**Transfer Engine** (BN-04): Replaced name-similarity heuristic with ConceptGraph HDC structural vector similarity. Cross-domain transfer now operates on 10,000-bit binding vectors rather than string edit distance.

**External Benchmarks** (BN-03): `run_full_benchmark()` loads `data/arc_agi_sample.json` (20 tasks, ARC canonical format) and `data/humaneval_sample.json` (10 problems, OpenAI HumanEval format). Weighted combined score: ARC-AGI × 0.60 + HumanEval × 0.40. Replaces legacy trivial list-reversal ADB tasks.

**Capability Scoring** (geometric mean of 5 axes):
- Generalization, Autonomy, Self-Improvement, Abstraction, Open-Endedness

> **Caveat:** These are internal proxy metrics. See RESULTS.md for honest failure reporting.

## Usage

```bash
# Run all tests (core + contract + 11 integration tests)
python main.py selftest

# Run anti-cheat audit
python main.py audit

# Run cognitive engine
python main.py --rounds 40 --agents 8

# Run capability evidence with 3-way ablation
python scripts/run_agi_evidence.py

# Self-improvement verification suite
python scripts/verify_self_improvement.py

# Benchmarks
python main.py benchmark --suite ADB_v1 --seed 0 --trials 20
```

## Evidence Summary

50-round evidence run (seed=42, with anti-wireheading active, post BN-01~06 fixes):

| Configuration | Composite | Domains |
|--------------|----------|---------|
| **Full system** | **0.488** | 41 |
| Ablation A (no capability modules) | 0.004 | 6 |
| Ablation B (no GoalGenerator) | 0.011 | 6 |
| Ablation C (no TransferEngine) | 0.192 | 41 |

### External Benchmark Baseline (BN-03)

| Benchmark | Tasks | Solved | Accuracy | Notes |
|-----------|-------|--------|----------|-------|
| ARC-AGI sample | 20 | 0 | 0.000 | No ARC solver plugged in |
| HumanEval sample | 10 | 0 | 0.000 | No code-gen solver plugged in |
| **Combined (60/40)** | 30 | 0 | **0.000** | Honest baseline; legacy ADB score 1.000 was trivial list-reversal |

Self-improvement score reflects governance-gated scoring with mandatory holdout metrics (hash-fallback removed). See [RESULTS.md](RESULTS.md) for full report.

## Bottleneck Fixes (BN-01 ~ BN-06)

| ID | Fix | Status |
|----|-----|--------|
| BN-01 | WorldModel tiny-transformer rewrite | ✅ Complete |
| BN-02 | OmegaForge → SkillLibrary RSI pipeline | ✅ Complete |
| BN-03 | ARC-AGI + HumanEval real benchmark datasets | ✅ Complete |
| BN-04 | TransferEngine HDC structural similarity | ✅ Complete |
| BN-05 | Governance hash-fallback removed | ✅ Complete |
| BN-06 | Adaptive meta-depth ceiling (calibration-based) | ✅ Complete |

## Scope & Limitations

- External benchmark score is 0.000 at baseline — no ARC or code-gen solver is connected yet; this is the correct honest baseline replacing the legacy trivial ADB score of 1.000
- Concept graph depth (5) partially driven by threshold calibration
- Meta-rollout confidence decays with depth; predictions beyond 3 steps are low-confidence
- All environments simulated; no real-world grounding
- TransferEngine HDC similarity improves over name-heuristics but cross-domain analogy remains limited by ConceptGraph depth

## License

MIT License
