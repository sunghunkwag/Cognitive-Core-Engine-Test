# Cognitive-Core-Engine

Production-style integration of a fixed orchestrator (NON-RSI core) with two plugins:
- Invention plugin (Omega Forge) invoked ONLY on stagnation
- Governance/Evaluation gate (Unified RSI Extended) that decides adoption (Critic-only)

## Non-Negotiable Architecture (Source of Truth)

1) **NON_RSI_AGI_CORE_v5.py** is the ONLY orchestrator and main loop owner.
2) **omega_forge_two_stage_feedback.py** is an invention plugin. It must be called ONLY when stagnation is declared by the orchestrator.
3) **UNIFIED_RSI_EXTENDED.py** is a governance/evaluation gate. It validates candidates and returns a verdict.  
   - Creator proposes. **Critic adopts.** No self-adoption by the invention module.

**Integration goal:** Do NOT merge files. Implement a deterministic call chain:
`B (Orchestrator) -> Omega (Invention on stagnation) -> Unified (Critic gate) -> B (register/reject)`

## Contracts (Data Interfaces)

We integrate via two JSON-serializable contracts:

### Contract A: GapSpec (Orchestrator -> Omega)
A structured request describing the current capability gap and constraints.

### Contract B: CandidatePacket (Omega -> Unified -> Orchestrator)
A structured artifact packet containing:
- candidate code/artifact
- evidence: train/hold/stress/transfer cases
- metrics and constraints
- critic verdict

## Work Rules for Codex (Strict)

- Do NOT treat the largest file as the main architecture. The main loop owner is NON_RSI_AGI_CORE_v5.py.
- Do NOT merge these three files into a single monolithic file.
- Do NOT create new folders or “adapter/bridge/helper” modules.
- Do NOT add external dependencies.
- Keep changes minimal and localized. Prefer adding glue functions to the orchestrator.

### Allowed Changes (Scope)
- Modify `NON_RSI_AGI_CORE_v5.py` to add:
  1) stagnation detection
  2) GapSpec builder
  3) Omega invocation (import or subprocess; choose one)
  4) Unified evaluation call and verdict handling
- Optional: add ONE small helper function in Omega/Unified to export/ingest the packets (no new files).

### Output Requirements
- Provide a short integration plan (<= 30 lines).
- Provide patch/diff-style edits only.
- Add a minimal self-test in `NON_RSI_AGI_CORE_v5.py` that:
  - simulates stagnation,
  - invokes Omega once,
  - sends one candidate to Unified for evaluation,
  - prints the final verdict.

If any constraint cannot be satisfied, stop and explain why rather than inventing a new architecture.

## Repository Status
Research/engineering hybrid. Safety via governance gate and rollback mindset.
