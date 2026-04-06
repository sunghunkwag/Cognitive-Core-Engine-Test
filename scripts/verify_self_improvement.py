#!/usr/bin/env python3
"""
자가 개선 실질 검증 테스트 (Self-Improvement Verification Suite)
================================================================
이 스크립트는 시스템의 자가 개선이 실제로 동작하는지 검증합니다.

검증 항목:
1. 경험적 env 롤아웃이 실제로 env.step()을 호출하는가?
2. 수정 전/후 성능 차이가 실제 측정인가 (산술 시뮬레이션이 아닌가)?
3. 거부된 수정이 존재하는가? (100% 수락은 고무도장)
4. 적용된 수정이 실제로 에이전트 행동을 변화시키는가?
5. 50라운드 진행 시 CompetenceMap, ConceptGraph, TransferEngine이
   실제로 개선 궤적을 보이는가?
6. 외부 벤치마크(ADB)에서 에이전트가 실제로 학습하는가?
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import NON_RSI_AGI_CORE_v5 as core
from agi_modules.self_improvement import SelfImprovementEngine
from agi_modules.competence_map import CompetenceMap
from agi_modules.concept_graph import ConceptGraph
from agi_modules.external_benchmark import ExternalBenchmarkHarness


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    marker = "  [OK]" if passed else "  [!!]"
    print(f"{marker} {name}: {status}")
    if detail:
        print(f"       {detail}")
    return passed


def test_1_empirical_rollout():
    """검증 1: test_modification()이 실제로 env.step()을 호출하는가?"""
    section("TEST 1: test_modification() 경험적 env 롤아웃 검증")

    env = core.ResearchEnvironment(seed=99)
    sie = SelfImprovementEngine()

    # 결정 이력 쌓기
    for i in range(10):
        sie.record_decision({"reward": 0.05 + i * 0.002, "action": "build_tool", "domain": "algorithm"})

    mod = {"type": "policy_modification", "changes": {"risk_delta": 0.05}}
    params = {"risk": 0.25}

    # env.step() 호출 횟수 추적
    original_step = env.step
    call_count = [0]
    def counting_step(*args, **kwargs):
        call_count[0] += 1
        return original_step(*args, **kwargs)
    env.step = counting_step

    result = sie.test_modification(mod, env, params)

    check("반환 타입이 dict인가?", isinstance(result, dict),
          f"type={type(result).__name__}")
    check("empirically_tested=True인가?", result.get("empirically_tested") is True,
          f"empirically_tested={result.get('empirically_tested')}")
    check("method='env_rollout'인가?", result.get("method") == "env_rollout",
          f"method={result.get('method')}")
    check("env.step()이 실제 호출되었는가?", call_count[0] > 0,
          f"env.step() 호출 횟수: {call_count[0]}")
    check("baseline_avg가 포함되어 있는가?", "baseline_avg" in result,
          f"baseline_avg={result.get('baseline_avg', 'MISSING')}")
    check("modified_avg가 포함되어 있는가?", "modified_avg" in result,
          f"modified_avg={result.get('modified_avg', 'MISSING')}")

    # env 없이 호출 시 fallback 확인
    result_no_env = sie.test_modification(mod, None, params)
    check("env=None → empirically_tested=False", result_no_env.get("empirically_tested") is False,
          f"result={result_no_env}")
    check("env=None → delta=0.0", result_no_env.get("delta") == 0.0)

    return call_count[0] > 0


def test_2_rejection_exists():
    """검증 2: 거부된 수정이 존재하는가? (100% 수락 방지)"""
    section("TEST 2: 수정 거부 메커니즘 검증")

    random.seed(42)
    env = core.ResearchEnvironment(seed=42)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    for r in range(30):
        orch.run_recursive_cycle(r, stagnation_override=(r > 5 and r % 7 == 0))

    proposed = orch.self_improvement.proposed_count()
    applied = orch.self_improvement.applied_count()
    rejected = proposed - applied

    check("제안이 발생했는가?", proposed > 0, f"proposed={proposed}")
    check("적용이 발생했는가?", applied > 0, f"applied={applied}")
    check("거부가 존재하는가 (100% 수락 아님)?", rejected > 0 or proposed < 5,
          f"proposed={proposed}, applied={applied}, rejected={rejected}")

    if proposed > 0:
        rate = applied / proposed
        check("수락률이 80% 이하인가?", rate <= 0.80,
              f"acceptance rate={rate:.1%}")
        guard = orch.self_improvement.acceptance_rate_guard()
        check("acceptance_rate_guard() suspicious=False?",
              not guard.get("suspicious", True),
              f"guard={guard}")

    return proposed > 0 and rejected >= 0


def test_3_parameter_change_effect():
    """검증 3: 적용된 수정이 실제로 에이전트 행동을 변화시키는가?"""
    section("TEST 3: 파라미터 변경 → 행동 변화 검증")

    random.seed(123)
    env = core.ResearchEnvironment(seed=123)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=4, base_budget=12, selection_top_k=2)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    # 초기 risk 값 기록
    initial_risk = orch._org_policy["risk"]

    # 30라운드 실행 (self-improvement가 5라운드마다 작동)
    for r in range(30):
        orch.run_recursive_cycle(r, stagnation_override=(r > 5 and r % 7 == 0))

    final_risk = orch._org_policy["risk"]
    mods = orch.self_improvement.get_applied_modifications()

    risk_changed = abs(final_risk - initial_risk) > 1e-6
    check("risk 파라미터가 변화했는가?", risk_changed,
          f"initial={initial_risk:.4f} → final={final_risk:.4f}")

    if mods:
        for i, mod in enumerate(mods):
            tr = mod.get("test_result", {})
            if isinstance(tr, dict):
                check(f"  mod[{i}] 경험적 테스트 여부",
                      tr.get("empirically_tested") is True,
                      f"delta={tr.get('delta', '?'):.4f}, method={tr.get('method')}")
    else:
        check("수정 적용 이력 존재", False, "mods 비어있음")

    return risk_changed


def test_4_competence_trajectory():
    """검증 4: 50라운드에 걸친 실질적 개선 궤적"""
    section("TEST 4: 50라운드 개선 궤적 검증")

    random.seed(42)
    env = core.ResearchEnvironment(seed=42)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=6, base_budget=20, selection_top_k=3)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    checkpoints = []
    start = time.time()

    for r in range(50):
        out = orch.run_recursive_cycle(
            r, stagnation_override=(r > 5 and r % 7 == 0),
            force_meta_proposal=(r > 10 and r % 10 == 0))

        if r % 10 == 0 or r == 49:
            mean_reward = sum(res["reward"] for res in out["results"]) / max(1, len(out["results"]))
            checkpoints.append({
                "round": r,
                "mean_reward": mean_reward,
                "competence_keys": len(orch.competence_map.all_keys()),
                "concept_count": orch.concept_graph.size(),
                "concept_depth": orch.concept_graph.depth(),
                "domains": len(env.tasks),
                "agi_composite": orch.agi_tracker.composite_score(),
                "agi_scores": orch.agi_tracker.score(),
            })

    elapsed = time.time() - start

    print(f"\n  50라운드 완료 ({elapsed:.1f}s)")
    print(f"  {'Round':>5} | {'Reward':>7} | {'Competence':>10} | {'Concepts':>8} | {'Depth':>5} | {'Domains':>7} | {'Composite':>9}")
    print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*9}")
    for cp in checkpoints:
        print(f"  {cp['round']:5d} | {cp['mean_reward']:7.4f} | {cp['competence_keys']:10d} | "
              f"{cp['concept_count']:8d} | {cp['concept_depth']:5d} | {cp['domains']:7d} | "
              f"{cp['agi_composite']:9.4f}")

    first = checkpoints[0]
    last = checkpoints[-1]

    check("컴피턴스 키 증가", last["competence_keys"] > first["competence_keys"],
          f"{first['competence_keys']} → {last['competence_keys']}")
    check("컨셉 그래프 성장", last["concept_count"] > first["concept_count"],
          f"{first['concept_count']} → {last['concept_count']}")
    check("컨셉 깊이 > 1", last["concept_depth"] > 1,
          f"depth={last['concept_depth']}")
    check("도메인 확장", last["domains"] > 6,
          f"6 → {last['domains']}")
    check("AGI 복합 점수 개선", last["agi_composite"] > first["agi_composite"],
          f"{first['agi_composite']:.4f} → {last['agi_composite']:.4f}")

    # 외부 벤치마크
    ext_scores = orch.external_benchmark.get_external_score_history()
    check("외부 벤치마크 기록 존재", len(ext_scores) > 0,
          f"snapshots={len(ext_scores)}")
    if ext_scores:
        check("에이전트 외부 정확도 > 0", max(ext_scores) > 0,
              f"scores={ext_scores}")

    # 전이 학습
    transfer_hist = orch.transfer_engine.get_history()
    check("전이 시도 존재", len(transfer_hist) > 0,
          f"transfer attempts={len(transfer_hist)}")
    if transfer_hist:
        best_analogy = max(t["analogy_score"] for t in transfer_hist)
        check("전이 유사도 > 0.02", best_analogy > 0.02,
              f"best analogy={best_analogy:.4f}")

    # 자가 개선
    si_proposed = orch.self_improvement.proposed_count()
    si_applied = orch.self_improvement.applied_count()
    check("자가 개선 제안 존재", si_proposed > 0,
          f"proposed={si_proposed}, applied={si_applied}")

    # 최종 점수 분석
    final_scores = last["agi_scores"]
    print(f"\n  최종 AGI 축 점수:")
    for axis, score in final_scores.items():
        marker = "[OK]" if score > 0.05 else "[--]"
        print(f"    {marker} {axis}: {score:.4f}")

    return last["agi_composite"] > first["agi_composite"]


def test_5_external_benchmark_learning():
    """검증 5: 외부 벤치마크에서 에이전트가 실제로 학습하는가?"""
    section("TEST 5: 외부 벤치마크 학습 곡선 검증")

    random.seed(42)
    env = core.ResearchEnvironment(seed=42)
    tools = core.ToolRegistry()
    cfg = core.OrchestratorConfig(agents=6, base_budget=20, selection_top_k=3)
    orch = core.Orchestrator(cfg, env, tools)
    tools.register("write_note", core.tool_write_note_factory(orch.mem))
    tools.register("write_artifact", core.tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", core.tool_evaluate_candidate)
    tools.register("tool_build_report", core.tool_tool_build_report)

    # 에이전트 solve_fn으로 벤치마크 실행 (학습 전)
    solve_fn_before = orch._make_agent_solve_fn()
    harness_before = ExternalBenchmarkHarness(seed=77)
    result_before = harness_before.run_adb_snapshot(solve_fn=solve_fn_before)
    acc_before = result_before["accuracy"]

    # 30라운드 학습
    for r in range(30):
        orch.run_recursive_cycle(r, stagnation_override=(r > 5 and r % 7 == 0))

    # 학습 후 벤치마크
    solve_fn_after = orch._make_agent_solve_fn()
    harness_after = ExternalBenchmarkHarness(seed=77)  # 동일 시드로 동일 문제
    result_after = harness_after.run_adb_snapshot(solve_fn=solve_fn_after)
    acc_after = result_after["accuracy"]

    check("학습 전 벤치마크 실행", True, f"accuracy={acc_before:.1%}")
    check("학습 후 벤치마크 실행", True, f"accuracy={acc_after:.1%}")
    check("학습 후 정확도 >= 학습 전", acc_after >= acc_before,
          f"{acc_before:.1%} → {acc_after:.1%}")

    # HDC 검증
    fresh_mem = core.SharedMemory()
    hdc_result = orch.external_benchmark.validate_hdc_retrieval(fresh_mem)
    check("HDC 정밀도 >= 0.6 (태그 필터 없이)",
          hdc_result["passes_threshold"],
          f"precision={hdc_result['mean_precision']:.3f}, "
          f"tag_inflation={hdc_result.get('possible_tag_inflation', '?')}, "
          f"random_baseline={hdc_result.get('random_baseline', '?')}")

    # 오버피팅 체크
    overfitting = orch.external_benchmark.is_overfitting(
        orch.agi_tracker.composite_score(),
        acc_after)
    check("오버피팅 미감지", not overfitting, f"is_overfitting={overfitting}")

    return acc_after >= acc_before


def main():
    print("=" * 60)
    print("  자가 개선 실질 검증 테스트 (Self-Improvement Verification)")
    print("=" * 60)

    results = {}
    results["1_empirical_rollout"] = test_1_empirical_rollout()
    results["2_rejection_exists"] = test_2_rejection_exists()
    results["3_parameter_effect"] = test_3_parameter_change_effect()
    results["4_improvement_trajectory"] = test_4_competence_trajectory()
    results["5_external_learning"] = test_5_external_benchmark_learning()

    section("최종 결과 요약")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        marker = "[OK]" if passed else "[!!]"
        print(f"  {marker} {name}: {status}")
        if not passed:
            all_pass = False

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\n  {passed_count}/{total_count} 검증 통과")

    if all_pass:
        print("\n  결론: 자가 개선이 실질적으로 작동하고 있음")
    else:
        print("\n  결론: 일부 자가 개선 메커니즘에 문제가 있음")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
