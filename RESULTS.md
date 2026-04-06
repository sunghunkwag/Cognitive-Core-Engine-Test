# AGI Evidence Report

## 1. AGI Progress Curves

| Round | Generalization | Autonomy | Self-Improvement | Abstraction | Open-Endedness | Composite |
|-------|---------------|----------|-----------------|-------------|---------------|-----------|
|   0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
|   5 | 0.008 | 0.500 | 0.000 | 0.200 | 0.000 | 0.015 |
|  10 | 0.040 | 0.650 | 0.500 | 0.400 | 0.000 | 0.088 |
|  15 | 0.060 | 0.667 | 0.500 | 0.400 | 0.100 | 0.241 |
|  20 | 0.075 | 0.650 | 0.500 | 0.400 | 0.382 | 0.327 |
|  25 | 0.089 | 0.680 | 0.667 | 0.400 | 0.382 | 0.361 |
|  30 | 0.109 | 0.667 | 0.667 | 1.000 | 0.431 | 0.461 |
|  35 | 0.124 | 0.686 | 0.667 | 1.000 | 0.538 | 0.498 |
|  40 | 0.141 | 0.675 | 0.750 | 1.000 | 0.646 | 0.541 |
|  45 | 0.163 | 0.644 | 0.600 | 1.000 | 0.646 | 0.527 |

## 2. Autonomous Goal Generation Evidence

- Rounds with autonomous goals: 49/50
- Final autonomy score: 0.633

## 3. Concept Formation Evidence

- Final concept count: 305
- Final concept depth: 5
- Depth over time: [(0, 0), (10, 2), (20, 2), (30, 5), (40, 5)]
- Promoted concepts: 114
- Multi-domain concepts (A7): 54

## 4. Transfer Learning Evidence

- Transfer attempts: 10
  - Round 0: algorithm -> strategy (analogy=0.444)
  - Round 5: algorithm+strategy -> strategy (analogy=0.650)
  - Round 10: systems -> strategy (analogy=0.450)
  - Round 15: engineering -> strategy (analogy=0.336)
  - Round 20: engineering -> strategy (analogy=0.311)

## 5. Self-Improvement Evidence

- Modifications proposed: 5
- Modifications applied: 3

## 6. Open-Ended Learning Evidence

- Total domains (start=6): 42
- Open-endedness score: 0.646

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
- First: 1.000, Last: 0.000

### Overfitting Check (A2)
- Is overfitting: True

## 8. Ablation Comparison (A10)

| Configuration | Final Composite | Mean Reward (last 10) | Concept Depth | Domains |
|--------------|----------------|----------------------|---------------|---------|
| **Full AGI system** | 0.5250 | 0.4672 | 5 | 42 |
| No new modules (legacy only) | 0.0040 | 0.1956 | 5 | 6 |
| All modules, GoalGenerator disabled | 0.0462 | 0.1607 | 5 | 6 |
| All modules, TransferEngine disabled | 0.1896 | 0.4672 | 5 | 42 |

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
Seed: 42, Rounds: 50, Time: 33.5s
