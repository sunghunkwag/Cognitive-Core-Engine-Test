# AGI Evidence Report

> Last updated: 2026-04-07 — reflects BN-01 ~ BN-06 bottleneck fixes.

## 0. Bottleneck Fix Summary

| ID | Description | Impact |
|----|-------------|--------|
| BN-01 | WorldModel tiny-transformer (2-layer) replacing linear feature hashing | Core prediction quality |
| BN-02 | RSI pipeline: OmegaForge → quarantine → SkillLibrary registration | Real code-gen RSI loop |
| BN-03 | External benchmarks: ARC-AGI (20 tasks) + HumanEval (10 problems) replacing trivial ADB list-reversal | Honest external validation |
| BN-04 | TransferEngine: HDC structural vector similarity replacing name-edit-distance | Cross-domain analogy quality |
| BN-05 | Governance critic: hash-fallback scoring path fully removed; holdout_rate mandatory | Anti-gaming hardening |
| BN-06 | Adaptive meta-depth ceiling based on calibration error history (depth 1–4) | Rollout reliability |

---

## 1. AGI Progress Curves

| Round | Generalization | Autonomy | Self-Improvement | Abstraction | Open-Endedness | Composite |
|-------|---------------|----------|-----------------|-------------|---------------|-----------|
|   0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
|   5 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.003 |
|  10 | 0.012 | 0.650 | 0.000 | 0.200 | 0.000 | 0.017 |
|  15 | 0.025 | 0.700 | 0.000 | 0.200 | 0.200 | 0.059 |
|  20 | 0.033 | 0.725 | 0.000 | 0.200 | 0.262 | 0.066 |
|  25 | 0.046 | 0.720 | 0.000 | 0.200 | 0.350 | 0.075 |
|  30 | 0.060 | 0.717 | 0.000 | 0.600 | 0.420 | 0.102 |
|  35 | 0.074 | 0.729 | 0.000 | 0.600 | 0.490 | 0.110 |
|  40 | 0.085 | 0.713 | 0.500 | 1.000 | 0.583 | 0.446 |
|  45 | 0.107 | 0.678 | 0.667 | 1.000 | 0.583 | 0.490 |

---

## 2. Autonomous Goal Generation Evidence

- Rounds with autonomous goals: 49/50
- Final autonomy score: 0.663

---

## 3. Concept Formation Evidence

- Final concept count: 292
- Final concept depth: 5
- Depth over time: [(0, 0), (10, 1), (20, 1), (30, 3), (40, 5)]
- Promoted concepts: 133
- Multi-domain concepts (A7): 87

---

## 4. Transfer Learning Evidence (BN-04)

**Pre-fix (name-similarity heuristic):**
- Transfer attempts: 10
- Analogy quality: 0.067 – 0.523 (string edit distance; domain-name collision artifacts)

**Post-fix (HDC structural vector similarity):**
- TransferEngine now computes cosine similarity on 10,000-bit ConceptGraph binding vectors
- Name-collision artifacts eliminated; cross-domain transfer governed by structural overlap
- Transfer analogy scores now reflect genuine concept structure, not lexical proximity

---

## 5. Self-Improvement Evidence (BN-02 + BN-05)

- Modifications proposed: 3
- Modifications applied: 2
- **BN-05:** hash-fallback path removed from governance critic; all accepted proposals required valid holdout_rate
- **BN-02:** accepted OmegaForge candidates now compiled → quarantined (5-input smoke-test, ≥3 clean halts) → registered into SkillLibrary via RSISkillRegistrar

---

## 6. Open-Ended Learning Evidence

- Total domains (start=6): 41
- Open-endedness score: 0.583

---

## 7. External Benchmark Results (BN-03)

### ⚠️ Legacy ADB Scores Were Invalid

Previous RESULTS.md reported `External Benchmark Scores: First 1.000, Last 1.000`. These scores came from `run_adb_snapshot()` which tested trivial list-reversal on random integers — not a meaningful external benchmark. BN-03 replaces this with real datasets.

### Real External Benchmark Baseline

| Benchmark | Dataset | Tasks | Solved | Accuracy |
|-----------|---------|-------|--------|----------|
| ARC-AGI | `data/arc_agi_sample.json` | 20 tasks | 0 | **0.000** |
| HumanEval | `data/humaneval_sample.json` | 10 problems | 0 | **0.000** |
| **Combined (ARC×0.6 + HE×0.4)** | — | 30 | 0 | **0.000** |

**Interpretation:** Score of 0.000 is the correct honest baseline. The system currently has no ARC grid solver or code-generation solver connected. Plugging in a real solver (e.g., a fine-tuned LLM or symbolic search) is the next step to measure genuine external generalization. A score of 0.000 here is more informative than 1.000 on list-reversal.

### HDC Retrieval Precision (A6)
- Mean precision: 0.800
- Passes threshold (0.6): True
  - algorithm: 0.600
  - systems: 1.000
  - theory: 0.800

### SelfModel Novel Task Calibration (A9)
- High confidence on novel tasks: 0
- Miscalibrated: False
- Passes: True

### Overfitting Check (A2)
- Is overfitting: True (internal composite improves; external benchmark remains 0.000 — expected at this stage)

---

## 8. Adaptive Meta-Depth Evidence (BN-06)

| Calibration Error | Depth Ceiling | Notes |
|-------------------|---------------|-------|
| < 0.05 | 4 | High-confidence rollout |
| 0.05 – 0.14 | 3 | Normal operation |
| 0.15 – 0.29 | 2 | Degraded confidence |
| ≥ 0.30 | 1 | Conservative; theorist/strategist roles get +1 |

- Calibration error tracked over last 8 transitions per agent
- `calibration_error()` exposed as public diagnostic method
- Episode memory records `calibration_error` and `planner_depth_used` fields

---

## 9. Ablation Comparison (A10)

| Configuration | Final Composite | Mean Reward (last 10) | Concept Depth | Domains |
|--------------|----------------|----------------------|---------------|---------|
| **Full AGI system** | **0.4877** | 0.4132 | 5 | 41 |
| No new modules (legacy only) | 0.0040 | 0.0482 | 5 | 6 |
| All modules, GoalGenerator disabled | 0.0110 | 0.0191 | 4 | 6 |
| All modules, TransferEngine disabled | 0.1916 | 0.4132 | 5 | 41 |

---

## What This Proves

- The AGI modules produce measurable progress across 5 capability axes
- Autonomous goal generation produces diverse tasks beyond hardcoded set
- Concept formation creates hierarchical abstractions from experience
- Self-improvement engine proposes and applies parameter modifications under mandatory holdout gating (BN-05)
- RSI pipeline now registers approved OmegaForge candidates into SkillLibrary (BN-02)
- Ablation comparison confirms new modules contribute beyond baseline
- HDC retrieval precision validated against domain-specific benchmark
- SelfModel correctly reports low confidence on novel unseen tasks

## What This Does NOT Prove

- These results do not demonstrate general intelligence
- External benchmark score is 0.000 — no ARC grid solver or code-gen solver is integrated yet
- The legacy ADB score of 1.000 reflected trivial list-reversal, not meaningful external validation
- The system operates in a simplified simulation environment
- Internal AGI axis scores may overestimate true capability (A4 caveat)
- ConceptGraph depth is partially driven by threshold calibration
- TransferEngine HDC similarity improves over name-heuristics but analogy quality remains limited

---
Seed: 42 | Rounds: 50 | Time: 37.3s | Branch: feat/agi-bottleneck-fixes
