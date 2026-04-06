# AGI Evidence Report

## 1. AGI Progress Curves

| Round | Generalization | Autonomy | Self-Improvement | Abstraction | Open-Endedness | Composite |
|-------|---------------|----------|-----------------|-------------|---------------|-----------|
|   0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
|   5 | 0.004 | 0.500 | 1.000 | 0.600 | 0.000 | 0.064 |
|  10 | 0.011 | 0.650 | 1.000 | 1.000 | 0.000 | 0.094 |
|  15 | 0.015 | 0.700 | 1.000 | 1.000 | 0.233 | 0.301 |
|  20 | 0.033 | 0.700 | 1.000 | 1.000 | 0.400 | 0.391 |
|  25 | 0.045 | 0.700 | 1.000 | 1.000 | 0.525 | 0.439 |
|  30 | 0.061 | 0.733 | 1.000 | 1.000 | 0.612 | 0.487 |
|  35 | 0.082 | 0.743 | 1.000 | 1.000 | 0.700 | 0.532 |
|  40 | 0.104 | 0.713 | 1.000 | 1.000 | 0.700 | 0.554 |
|  45 | 0.123 | 0.689 | 1.000 | 1.000 | 0.700 | 0.568 |

## 2. Autonomous Goal Generation Evidence

- Rounds with autonomous goals: 49/50
- Final autonomy score: 0.704

## 3. Concept Formation Evidence

- Final concept count: 515
- Final concept depth: 5
- Depth over time: [(0, 0), (10, 5), (20, 5), (30, 5), (40, 5)]
- Promoted concepts: 252
- Multi-domain concepts (A7): 172

## 4. Transfer Learning Evidence

- Transfer attempts: 10
  - Round 0: algorithm -> strategy (analogy=0.244)
  - Round 5: theory+algorithm -> strategy (analogy=0.285)
  - Round 10: theory+algorithm+theory+algorithm+theory+theory -> strategy (analogy=0.253)
  - Round 15: theory+algorithm+theory+theory+verification -> strategy (analogy=0.266)
  - Round 20: strategy+engineering -> strategy (analogy=0.489)

## 5. Self-Improvement Evidence

- Modifications proposed: 7
- Modifications applied: 7

## 6. Open-Ended Learning Evidence

- Total domains (start=6): 48
- Open-endedness score: 0.700

## 7. External Validation

### HDC Retrieval Precision (A6)
- Mean precision: 1.000
- Passes threshold (0.6): True
  - algorithm: 1.000
  - systems: 1.000
  - theory: 1.000

### SelfModel Novel Task Calibration (A9)
- High confidence on novel tasks: 0
- Miscalibrated: False
- Passes: True

### External Benchmark Scores (A5)
- Snapshots taken: 8
- First: 0.000, Last: 0.000

### Overfitting Check (A2)
- Is overfitting: False

## 8. Ablation Comparison (A10)

| Configuration | Final Composite | Mean Reward (last 10) | Concept Depth | Domains |
|--------------|----------------|----------------------|---------------|---------|
| **Full AGI system** | 0.5705 | 0.4858 | 5 | 48 |
| No new modules (legacy only) | 0.0040 | 0.0612 | 5 | 6 |
| All modules, GoalGenerator disabled | 0.0451 | 0.0978 | 4 | 6 |
| All modules, TransferEngine disabled | 0.2180 | 0.4858 | 5 | 48 |

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
Seed: 42, Rounds: 50, Time: 36.9s
