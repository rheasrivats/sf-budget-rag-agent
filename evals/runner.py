from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable
from urllib.parse import urlparse

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import Base
from app.models import PlanVersion
from app.services.chat import answer_question
from evals.cases import load_suite
from evals.costs import estimate_trial_cost, sum_cost_estimates
from evals.graders import run_deterministic_graders
from evals.judges import RUBRIC_NAMES, grade_with_llm
from evals.trace import TraceCollector, redact_secrets
from evals.types import CaseAggregateResult, CaseTrialResult, EvalRunResult, QACase, RubricGrade

JudgeFn = Callable[[QACase, dict, str, float, TraceCollector], dict[str, RubricGrade]]
TargetFn = Callable[[Session, PlanVersion, str, QACase, TraceCollector], dict]
ReporterFn = Callable[[str], None]


def _blocking_deterministic_passed(grades) -> bool:
    return all(grade.passed for grade in grades if grade.blocking)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Q&A evals against the current RAG database.")
    parser.add_argument("--cases", type=Path, default=Path("evals/qa_cases.yml"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--critical-floor", type=float, default=0.5)
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    return parser.parse_args(argv)


def _sqlite_path_from_url(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Q&A evals currently support sqlite:/// DATABASE_URL values only.")
    parsed = urlparse(database_url)
    raw_path = parsed.path
    if database_url.startswith("sqlite:////"):
        return Path(raw_path)
    return Path(raw_path.lstrip("/") or database_url.removeprefix("sqlite:///")).resolve()


def _copy_db_to_temp() -> tuple[tempfile.TemporaryDirectory, Path]:
    source = _sqlite_path_from_url(settings.database_url)
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")
    temp_dir = tempfile.TemporaryDirectory(prefix="qa-eval-db-")
    target = Path(temp_dir.name) / source.name
    shutil.copy2(source, target)
    return temp_dir, target


def _session_for_db(path: Path) -> tuple[Session, object]:
    engine = create_engine(f"sqlite:///{path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return local(), engine


def resolve_plan(db: Session, plan_id: int | str) -> PlanVersion:
    if plan_id == "latest":
        plan = db.scalar(select(PlanVersion).order_by(PlanVersion.created_at.desc()))
    else:
        plan = db.scalar(select(PlanVersion).where(PlanVersion.id == int(plan_id)))
    if plan is None:
        raise ValueError(f"Plan not found for eval plan_id={plan_id!r}")
    return plan


def default_target(db: Session, plan: PlanVersion, thread_id: str, case: QACase, trace: TraceCollector) -> dict:
    return answer_question(
        db,
        plan=plan,
        thread_id=thread_id,
        query=case.query,
        escalate=case.escalate,
        include_retrieved_context=True,
        trace_collector=trace,
        crag_enabled=bool(case.crag_enabled),
        force_crag=bool(case.force_crag),
        crag_rerank_threshold=case.crag_rerank_threshold,
    )


def default_judge(
    case: QACase,
    prediction: dict,
    model_name: str,
    threshold: float,
    trace: TraceCollector,
) -> dict[str, RubricGrade]:
    return grade_with_llm(case, prediction, model_name=model_name, threshold=threshold, trace_collector=trace)


def _aggregate_case(
    case_id: str,
    trials: list[CaseTrialResult],
    *,
    threshold: float,
    critical_floor: float,
) -> CaseAggregateResult:
    failures: list[str] = []
    deterministic_passed = all(
        _blocking_deterministic_passed(trial.deterministic_grades) for trial in trials if trial.error is None
    ) and all(trial.error is None for trial in trials)

    rubric_summary: dict[str, dict[str, float | bool]] = {}
    for rubric_name in RUBRIC_NAMES:
        scores = [trial.rubric_grades[rubric_name].score for trial in trials if rubric_name in trial.rubric_grades]
        if not scores:
            rubric_summary[rubric_name] = {"min": 0.0, "mean": 0.0, "max": 0.0, "passed": False}
            failures.append(f"{rubric_name}: no scores")
            continue
        mean_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        passed = mean_score >= threshold and min_score >= critical_floor
        rubric_summary[rubric_name] = {
            "min": round(min_score, 4),
            "mean": round(mean_score, 4),
            "max": round(max_score, 4),
            "passed": passed,
        }
        if not passed:
            failures.append(f"{rubric_name}: mean={mean_score:.3f}, min={min_score:.3f}")

    for trial in trials:
        if trial.error:
            failures.append(f"trial {trial.trial}: {trial.error}")
        for grade in trial.deterministic_grades:
            if not grade.passed and grade.blocking:
                failures.append(f"trial {trial.trial} {grade.name}: {grade.explanation}")

    trial_count = len(trials)
    passed_trial_count = sum(1 for trial in trials if trial.passed)
    pass_at_k = passed_trial_count > 0
    pass_caret_k = trial_count > 0 and passed_trial_count == trial_count
    trial_pass_rate = passed_trial_count / trial_count if trial_count else 0.0
    passed = deterministic_passed and all(bool(item["passed"]) for item in rubric_summary.values())
    cost_estimate = sum_cost_estimates(
        [trial.cost_estimate for trial in trials if trial.cost_estimate is not None]
    )
    return CaseAggregateResult(
        case_id=case_id,
        passed=passed,
        trials=trials,
        pass_at_k=pass_at_k,
        pass_caret_k=pass_caret_k,
        trial_pass_rate=round(trial_pass_rate, 4),
        deterministic_passed=deterministic_passed,
        rubric_summary=rubric_summary,
        cost_estimate=cost_estimate,
        failures=failures,
    )


def _aggregate_run(
    *,
    suite_name: str,
    run_id: str,
    limited: bool,
    limit: int | None,
    trials: int,
    threshold: float,
    critical_floor: float,
    cases: list[CaseAggregateResult],
) -> EvalRunResult:
    failures = [
        {"case_id": case.case_id, "reason": "; ".join(case.failures)}
        for case in cases
        if not case.passed
    ]
    rubric_means: dict[str, float] = {}
    for rubric_name in RUBRIC_NAMES:
        scores = [
            trial.rubric_grades[rubric_name].score
            for case in cases
            for trial in case.trials
            if rubric_name in trial.rubric_grades
        ]
        rubric_means[rubric_name] = round(sum(scores) / len(scores), 4) if scores else 0.0

    total_cases = len(cases)
    total_trials = sum(len(case.trials) for case in cases)
    passed_trials = sum(1 for case in cases for trial in case.trials if trial.passed)
    suite_passed = not failures
    cost_estimate = sum_cost_estimates(
        [case.cost_estimate for case in cases if case.cost_estimate is not None]
    )
    return EvalRunResult(
        suite=suite_name,
        run_id=run_id,
        limited=limited,
        limit=limit,
        trials=trials,
        threshold=threshold,
        critical_floor=critical_floor,
        suite_passed=suite_passed,
        cases_passed=sum(1 for case in cases if case.passed),
        cases_failed=sum(1 for case in cases if not case.passed),
        pass_at_k=round(sum(1 for case in cases if case.pass_at_k) / total_cases, 4) if total_cases else 0.0,
        pass_caret_k=round(sum(1 for case in cases if case.pass_caret_k) / total_cases, 4) if total_cases else 0.0,
        trial_pass_rate=round(passed_trials / total_trials, 4) if total_trials else 0.0,
        rubric_means=rubric_means,
        cost_estimate=cost_estimate,
        failures=failures,
        cases=cases,
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(redact_secrets(payload), indent=2), encoding="utf-8")


def _write_markdown(path: Path, result: EvalRunResult) -> None:
    lines = [
        f"# Eval Run {result.run_id}",
        "",
        f"- Suite: `{result.suite}`",
        f"- Passed: `{result.suite_passed}`",
        f"- Limited: `{result.limited}`",
        f"- Trials: `{result.trials}`",
        f"- Threshold: `{result.threshold}`",
        f"- Critical floor: `{result.critical_floor}`",
        f"- pass@{result.trials}: `{result.pass_at_k}`",
        f"- pass^{result.trials}: `{result.pass_caret_k}`",
        f"- Trial pass rate: `{result.trial_pass_rate}`",
        f"- Estimated tokens: `{result.cost_estimate.total_tokens if result.cost_estimate else 0}`",
        f"- Estimated cost: `${result.cost_estimate.estimated_cost_usd if result.cost_estimate else 0.0:.6f}`",
        "",
        "## Rubric Means",
        "",
    ]
    for name, score in result.rubric_means.items():
        lines.append(f"- `{name}`: `{score}`")
    lines.extend(["", "## Cases", ""])
    for case in result.cases:
        lines.append(
            f"- `{case.case_id}`: `{'pass' if case.passed else 'fail'}` "
            f"(pass@{result.trials}=`{case.pass_at_k}`, pass^{result.trials}=`{case.pass_caret_k}`, "
            f"trial_pass_rate=`{case.trial_pass_rate}`, "
            f"est_cost=`${case.cost_estimate.estimated_cost_usd if case.cost_estimate else 0.0:.6f}`)"
        )
        for failure in case.failures[:5]:
            lines.append(f"  - {failure}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trace(path: Path, trace: TraceCollector) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")


def _format_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _report_trial_result(report: ReporterFn | None, trial_result: CaseTrialResult) -> None:
    if report is None:
        return
    if trial_result.error:
        report(f"    trial {trial_result.trial}: FAIL error={trial_result.error}")
        return

    deterministic_passed = _blocking_deterministic_passed(trial_result.deterministic_grades)
    deterministic_by_name = {grade.name: grade for grade in trial_result.deterministic_grades}
    retrieval_bits = []
    for name, label in [
        ("citation_urls", "citation_urls"),
        ("retrieval_recall_at_k", "recall"),
        ("retrieval_precision_at_k", "precision"),
        ("retrieval_mrr", "mrr"),
        ("citation_validity", "citation_validity"),
    ]:
        grade = deterministic_by_name.get(name)
        if grade is not None and grade.score is not None:
            status = _format_bool(grade.passed) if grade.blocking else ("OK" if grade.passed else "WARN")
            retrieval_bits.append(f"{label}={grade.score:.2f}/{status}")
    rubric_bits = []
    for name in RUBRIC_NAMES:
        grade = trial_result.rubric_grades.get(name)
        if grade is None:
            rubric_bits.append(f"{name}=missing")
        else:
            rubric_bits.append(f"{name}={grade.score:.2f}/{_format_bool(grade.passed)}")
    cost = trial_result.cost_estimate
    cost_bit = (
        f" cost≈${cost.estimated_cost_usd:.4f} tokens≈{cost.total_tokens}"
        if cost is not None
        else ""
    )
    report(
        "    "
        f"trial {trial_result.trial}: {_format_bool(trial_result.passed)} "
        f"deterministic={_format_bool(deterministic_passed)} "
        + " ".join(retrieval_bits)
        + (" " if retrieval_bits else "")
        + " ".join(rubric_bits)
        + cost_bit
    )


def _report_case_result(report: ReporterFn | None, case_result: CaseAggregateResult) -> None:
    if report is None:
        return
    stats = (
        f"pass@{len(case_result.trials)}={_format_bool(case_result.pass_at_k)} "
        f"pass^{len(case_result.trials)}={_format_bool(case_result.pass_caret_k)} "
        f"trial_pass_rate={case_result.trial_pass_rate:.2f} "
        f"cost≈${case_result.cost_estimate.estimated_cost_usd if case_result.cost_estimate else 0.0:.4f}"
    )
    if case_result.passed:
        report(f"  case result: PASS {stats}")
        return
    report(f"  case result: FAIL {stats}")
    for failure in case_result.failures[:3]:
        report(f"    - {failure}")


def run_qa_eval(
    cases_path: Path,
    output_dir: Path = Path("eval_results"),
    trials: int = 1,
    limit: int | None = None,
    threshold: float = 0.7,
    critical_floor: float = 0.5,
    judge_model: str = "gpt-5.4-mini",
    *,
    target_fn: TargetFn = default_target,
    judge_fn: JudgeFn = default_judge,
    report: ReporterFn | None = None,
) -> EvalRunResult:
    if trials < 1:
        raise ValueError("--trials must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be >= 1 when provided")

    suite = load_suite(cases_path)
    cases_with_suite_defaults = [
        case.model_copy(
            update={
                "crag_enabled": suite.crag_enabled if case.crag_enabled is None else case.crag_enabled,
                "crag_rerank_threshold": (
                    suite.crag_rerank_threshold
                    if case.crag_rerank_threshold is None
                    else case.crag_rerank_threshold
                ),
            }
        )
        for case in suite.cases
    ]
    selected_cases = cases_with_suite_defaults[:limit] if limit is not None else cases_with_suite_defaults
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    total_trials = len(selected_cases) * trials
    if report is not None:
        report(f"Starting eval suite '{suite.suite}'")
        report(f"Run ID: {run_id}")
        report(f"Cases: {len(selected_cases)} of {len(suite.cases)} | Trials per case: {trials} | Total trials: {total_trials}")
        if limit is not None:
            report("Limited run: true (subset pass is not a full-suite pass)")
    temp_dir, db_path = _copy_db_to_temp()

    case_results: list[CaseAggregateResult] = []
    try:
        db, engine = _session_for_db(db_path)
        try:
            for case_idx, case in enumerate(selected_cases, start=1):
                trial_results: list[CaseTrialResult] = []
                resolved_plan_id = case.plan_id if case.plan_id is not None else suite.default_plan_id
                plan = resolve_plan(db, resolved_plan_id)
                if report is not None:
                    report(f"\n[{case_idx}/{len(selected_cases)}] {case.id}")
                    report(f"  query: {case.query}")
                for trial_idx in range(1, trials + 1):
                    thread_prefix = case.thread_id or f"eval-{run_id}-{case.id}"
                    thread_id = f"{thread_prefix}-trial-{trial_idx}"
                    trace = TraceCollector(run_id=run_id, suite=suite.suite, case_id=case.id, trial=trial_idx)
                    trace.record(
                        "run_metadata",
                        {
                            "resolved_db_path": str(db_path),
                            "resolved_plan_id": plan.id,
                            "limited": limit is not None,
                        },
                    )
                    trace.record("case", case.model_dump(mode="json"))
                    started = perf_counter()
                    trace_path = run_dir / "traces" / case.id / f"trial_{trial_idx}.json"
                    try:
                        prediction = target_fn(db, plan, thread_id, case, trace)
                        deterministic = run_deterministic_graders(case, prediction)
                        trace.record("deterministic_grades", [g.model_dump(mode="json") for g in deterministic])
                        rubric = judge_fn(case, prediction, judge_model, threshold, trace)
                        cost_estimate = estimate_trial_cost(
                            case=case,
                            prediction=prediction,
                            judge_model=judge_model,
                            trace_data=trace.to_dict(),
                        )
                        trace.record("cost_estimate", cost_estimate.model_dump(mode="json"))
                        trace.record("elapsed_ms", round((perf_counter() - started) * 1000, 3))
                        trial_passed = _blocking_deterministic_passed(deterministic) and all(
                            g.passed for g in rubric.values()
                        )
                        trial_result = CaseTrialResult(
                            case_id=case.id,
                            trial=trial_idx,
                            passed=trial_passed,
                            answer=prediction.get("answer", ""),
                            model_used=prediction.get("model_used", ""),
                            citations=prediction.get("citations", []),
                            deterministic_grades=deterministic,
                            rubric_grades=rubric,
                            cost_estimate=cost_estimate,
                            trace_path=str(trace_path),
                        )
                        db.commit()
                    except Exception as exc:
                        db.rollback()
                        trace.record("error", str(exc))
                        trace.record("elapsed_ms", round((perf_counter() - started) * 1000, 3))
                        trial_result = CaseTrialResult(
                            case_id=case.id,
                            trial=trial_idx,
                            passed=False,
                            answer="",
                            model_used="",
                            citations=[],
                            deterministic_grades=[],
                            rubric_grades={},
                            trace_path=str(trace_path),
                            error=str(exc),
                        )
                    _write_trace(trace_path, trace)
                    _report_trial_result(report, trial_result)
                    trial_results.append(trial_result)
                case_result = _aggregate_case(
                    case.id,
                    trial_results,
                    threshold=threshold,
                    critical_floor=critical_floor,
                )
                _report_case_result(report, case_result)
                case_results.append(case_result)
        finally:
            db.close()
            engine.dispose()
    finally:
        temp_dir.cleanup()

    result = _aggregate_run(
        suite_name=suite.suite,
        run_id=run_id,
        limited=limit is not None,
        limit=limit,
        trials=trials,
        threshold=threshold,
        critical_floor=critical_floor,
        cases=case_results,
    )
    _write_json(run_dir / "summary.json", result)
    _write_markdown(run_dir / "summary.md", result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_run.txt").write_text(run_id + "\n", encoding="utf-8")
    if report is not None:
        report("")
        report(f"Suite result: {_format_bool(result.suite_passed)}")
        report(f"Cases passed: {result.cases_passed}; failed: {result.cases_failed}")
        report(
            f"Multi-trial stats: pass@{result.trials}={result.pass_at_k:.2%} "
            f"pass^{result.trials}={result.pass_caret_k:.2%} "
            f"trial_pass_rate={result.trial_pass_rate:.2%}"
        )
        if result.cost_estimate is not None:
            report(
                f"Estimated usage: tokens≈{result.cost_estimate.total_tokens} "
                f"(input≈{result.cost_estimate.input_tokens}, output≈{result.cost_estimate.output_tokens}) "
                f"cost≈${result.cost_estimate.estimated_cost_usd:.4f}"
            )
        rubric_summary = " ".join(f"{name}={score:.2f}" for name, score in result.rubric_means.items())
        report(f"Rubric means: {rubric_summary}")
        report(f"Results: {run_dir}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_qa_eval(
            cases_path=args.cases,
            output_dir=args.output_dir,
            trials=args.trials,
            limit=args.limit,
            threshold=args.threshold,
            critical_floor=args.critical_floor,
            judge_model=args.judge_model,
            report=print,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Eval setup error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected eval error: {exc}", file=sys.stderr)
        return 3

    return 0 if result.suite_passed else 1
