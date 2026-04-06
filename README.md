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
  transfer_engine.py          Cross-domain transfer with rollback
  self_model.py               Legacy capability prediction (backward compat)
  self_referential_model.py   Advanced self-referential meta-simulation
  difficulty_scheduler.py     Curriculum learning with chaos injection
  self_improvement.py         Empirical env-rollout parameter tuning
  agi_tracker.py              5-axis capability proxy scoring
  external_benchmark.py       Held-out validation, overfitting detection
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

`Orchestrator -> Omega (on stagnation) -> Governance (critic) -> Orchestrator (register/reject)`

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
    |
  Orchestrator.run_recursive_cycle()
    |-- validate_metric_integrity()
    |     Anti-wireheading gate: rejects self-improvement claims that lack
    |     structural correlation or external benchmark confirmation
    |-- Immutable objective anchor checked (read-only HDC vector)
```

### Capability Extension

```
Orchestrator.run_recursive_cycle()
  |-- GoalGenerator.generate()           [autonomous task creation]
  |-- CompetenceMap.update()             [competence tracking]
  |-- ConceptGraph.sweep_promote_all()   [hierarchical abstraction]
  |-- TransferEngine.transfer()          [cross-domain knowledge reuse]
  |-- SelfImprovementEngine.introspect() [empirical parameter tuning]
  |-- DifficultyScheduler.schedule()     [curriculum adjustment]
  |-- AGIProgressTracker.tick_round()    [5-axis proxy measurement]
  `-- ExternalBenchmark.run_adb_snapshot() [held-out validation]
```

## Technical Details

**Self-Referential Model** (`self_referential_model.py`):
- HDC state encoding: env hash XOR competence profile XOR concept structure XOR skill set
- Recursive meta-rollout: dual simulation predicting env state AND internal policy shift
- Architectural drift detection: cosine distance on unified state history (threshold 0.35)
- Anti-wireheading: immutable objective anchor + metric integrity validation + decoupled evaluation

**HDC Memory**: Title-weighted position-bound encoding, deterministic tie-breaking, similarity threshold 0.51, max 20,000 items.

**World Model**: TD-learning with non-linear features, experience replay (200 samples), gamma 0.9, combined reward: extrinsic (0.6) + intrinsic (0.4).

**Self-Improvement**: Empirical env.step() rollouts (5 baseline + 5 modified episodes). Anti-wireheading gate rejects modifications exceeding MAX_CREDIBLE_LEAP (0.25) or lacking external benchmark correlation.

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

50-round evidence run (seed=42, with anti-wireheading active):

| Configuration | Composite | Domains |
|--------------|----------|---------|
| **Full system** | **0.088** | 46 |
| Ablation A (no capability modules) | 0.004 | 6 |
| Ablation B (no GoalGenerator) | 0.010 | 6 |
| Ablation C (no TransferEngine) | 0.042 | 46 |

Self-improvement score = 0.0 reflects the anti-wireheading gate correctly rejecting modifications that lack external benchmark correlation. The lower composite vs earlier runs reflects honest, integrity-gated scoring.

See [RESULTS.md](RESULTS.md) for full report.

## Scope & Limitations

- Anti-wireheading gate may be overly conservative (self-improvement = 0.0)
- Concept graph depth (5) partially driven by threshold calibration
- Transfer analogy uses name-similarity heuristics, not structural matching
- Meta-rollout confidence decays with depth; predictions beyond 3 steps are low-confidence
- All environments simulated; no real-world grounding

## License

MIT License
