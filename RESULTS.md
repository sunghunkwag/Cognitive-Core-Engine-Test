# AGI Evidence Report

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

## 2. Autonomous Goal Generation Evidence

- Rounds with autonomous goals: 49/50
- Final autonomy score: 0.663

## 3. Concept Formation Evidence

- Final concept count: 292
- Final concept depth: 5
- Depth over time: [(0, 0), (10, 1), (20, 1), (30, 3), (40, 5)]
- Promoted concepts: 133
- Multi-domain concepts (A7): 87

## 4. Transfer Learning Evidence

- Transfer attempts: 10
  - Round 0: algorithm -> strategy (analogy=0.244)
  - Round 5: systems+algorithm+strategy -> theory (analogy=0.067)
  - Round 10: algorithm+strategy+systems+algorithm+strategy -> theory (analogy=0.294)
  - Round 15: engineering -> theory (analogy=0.523)
  - Round 20: algorithm+systems+algorithm+strategy+algorithm+strategy+systems+algorithm+strategy -> theory (analogy=0.370)

## 5. Self-Improvement Evidence

- Modifications proposed: 3
- Modifications applied: 2

## 6. Open-Ended Learning Evidence

- Total domains (start=6): 41
- Open-endedness score: 0.583

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
| **Full AGI system** | 0.4877 | 0.4132 | 5 | 41 |
| No new modules (legacy only) | 0.0040 | 0.0482 | 5 | 6 |
| All modules, GoalGenerator disabled | 0.0110 | 0.0191 | 4 | 6 |
| All modules, TransferEngine disabled | 0.1916 | 0.4132 | 5 | 41 |

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
Seed: 42, Rounds: 50, Time: 37.3s
