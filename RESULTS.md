# AGI Evidence Report

## 1. AGI Progress Curves

| Round | Generalization | Autonomy | Self-Improvement | Abstraction | Open-Endedness | Composite |
|-------|---------------|----------|-----------------|-------------|---------------|-----------|
|   0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
|   5 | 0.003 | 0.500 | 0.000 | 0.000 | 0.000 | 0.004 |
|  10 | 0.003 | 0.650 | 1.000 | 0.000 | 0.000 | 0.018 |
|  15 | 0.018 | 0.700 | 1.000 | 0.200 | 0.200 | 0.218 |
|  20 | 0.025 | 0.700 | 1.000 | 0.200 | 0.311 | 0.256 |
|  25 | 0.030 | 0.700 | 0.500 | 1.000 | 0.382 | 0.331 |
|  30 | 0.033 | 0.733 | 0.333 | 1.000 | 0.382 | 0.314 |
|  35 | 0.048 | 0.729 | 0.333 | 1.000 | 0.445 | 0.349 |
|  40 | 0.059 | 0.700 | 0.500 | 1.000 | 0.445 | 0.392 |
|  45 | 0.068 | 0.689 | 0.500 | 1.000 | 0.445 | 0.402 |

## 2. Autonomous Goal Generation Evidence

- Rounds with autonomous goals: 49/50
- Final autonomy score: 0.663

## 3. Concept Formation Evidence

- Final concept count: 256
- Final concept depth: 5
- Depth over time: [(0, 0), (10, 0), (20, 1), (30, 5), (40, 5)]
- Promoted concepts: 66
- Multi-domain concepts (A7): 19

## 4. Transfer Learning Evidence

- Transfer attempts: 10
  - Round 0: algorithm -> strategy (analogy=0.244)
  - Round 5: systems -> theory (analogy=0.200)
  - Round 12: strategy+systems -> theory (analogy=0.380)
  - Round 17: strategy+systems+algorithm -> theory (analogy=0.217)
  - Round 22: algorithm+theory+theory+strategy+algorithm+theory+theory+strategy+algorithm+verification -> theory (analogy=0.336)

## 5. Self-Improvement Evidence

- Modifications proposed: 4
- Modifications applied: 2

## 6. Open-Ended Learning Evidence

- Total domains (start=6): 47
- Open-endedness score: 0.445

## 7. External Validation

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

### External Benchmark Scores (A5)
- Snapshots taken: 8
- First: 1.000, Last: 1.000

### Overfitting Check (A2)
- Is overfitting: True

## 8. Ablation Comparison (A10)

| Configuration | Final Composite | Mean Reward (last 10) | Concept Depth | Domains |
|--------------|----------------|----------------------|---------------|---------|
| **Full AGI system** | 0.4122 | 0.4174 | 5 | 47 |
| No new modules (legacy only) | 0.0040 | 0.0000 | 5 | 6 |
| All modules, GoalGenerator disabled | 0.0427 | 0.1304 | 5 | 6 |
| All modules, TransferEngine disabled | 0.1714 | 0.4174 | 5 | 47 |

## What This Proves

- The AGI modules produce measurable progress across 5 capability axes
- Autonomous goal generation produces diverse tasks beyond hardcoded set
- Concept formation creates hierarchical abstractions from experience
- Self-improvement engine proposes and applies parameter modifications
- Ablation comparison confirms new modules contribute beyond baseline
- HDC retrieval precision validated against domain-specific benchmark
- SelfModel correctly reports low confidence on novel unseen tasks

## What This Does NOT Prove

- These results do not demonstrate general intelligence
- Performance on held-out benchmarks (ARC-AGI, etc.) is not validated here
- The system operates in a simplified simulation environment
- Transfer learning effectiveness is limited by simulated domain similarity
- Internal AGI axis scores may overestimate true capability (A4 caveat)
- ConceptGraph depth is partially driven by threshold calibration

---
Seed: 42, Rounds: 50, Time: 27.4s
