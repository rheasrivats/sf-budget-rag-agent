from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import PlanVersion
from evals.cases import load_suite
from evals.graders import extract_numeric_date_claims, run_deterministic_graders
from evals.judges import _rubric_prompt
from evals.runner import parse_args, run_qa_eval
from evals.trace import redact_secrets
from evals.types import QACase, RubricGrade


def test_load_initial_qa_suite():
    suite = load_suite(Path("evals/qa_cases.yml"))
    assert suite.suite == "qa_official_v1_current_db"
    assert len(suite.cases) == 8
    assert all(case.expected_source_kinds == ["official"] for case in suite.cases)
    assert all("plan://" in case.must_not_include for case in suite.cases)
    assert suite.cases[0].relevant_doc_ids == [11]
    assert suite.cases[1].relevant_doc_ids == [9]


def test_deterministic_graders_pass_official_prediction():
    case = QACase(
        id="timeline",
        query="timeline?",
        reference_answer="December and March.",
        expected_source_kinds=["official"],
        expected_citation_urls=["https://sf.gov/topics/budget"],
        must_include=["December"],
        must_not_include=["plan://"],
    )
    prediction = {
        "answer": "Departments get instructions in December.",
        "citations": [{"url": "https://sf.gov/topics/budget", "source_kind": "official"}],
        "retrieved_context": [
            {
                "content": "Budget timeline",
                "metadata": {"source_kind": "official", "url": "https://sf.gov/topics/budget"},
            }
        ],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = run_deterministic_graders(case, prediction)
    assert all(grade.passed for grade in grades)


def test_deterministic_grades_include_human_readable_descriptions():
    case = QACase(id="x", query="x", reference_answer="x")
    prediction = {
        "answer": "x",
        "citations": [],
        "retrieved_context": [{"content": "x", "metadata": {"source_kind": "official", "url": "https://sf.gov/x"}}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = run_deterministic_graders(case, prediction)

    assert all(grade.description for grade in grades)
    assert "non-empty answer" in grades[0].description


def test_no_plan_source_enforcement_fails_on_plan_source():
    case = QACase(id="x", query="x", reference_answer="x")
    prediction = {
        "answer": "x",
        "citations": [{"url": "plan://1", "source_kind": "plan"}],
        "retrieved_context": [{"content": "x", "metadata": {"source_kind": "plan", "url": "plan://1"}}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert not grades["no_plan_sources"].passed


def test_retrieval_metrics_use_relevant_doc_ids():
    case = QACase(id="x", query="x", reference_answer="x", relevant_doc_ids=[7], retrieval_k=3, min_mrr=0.5)
    prediction = {
        "answer": "x",
        "retrieved_context": [
            {"source_id": 1, "content": "wrong", "metadata": {"url": "https://sf.gov/1"}},
            {"source_id": 7, "content": "right", "metadata": {"url": "https://sf.gov/7"}},
            {"source_id": 9, "content": "other", "metadata": {"url": "https://sf.gov/9"}},
        ],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert grades["retrieval_recall_at_k"].passed
    assert grades["retrieval_recall_at_k"].score == 1.0
    assert grades["retrieval_precision_at_k"].observed["precision"] == 1 / 3
    assert grades["retrieval_precision_at_k"].score == 1 / 3
    assert grades["retrieval_mrr"].passed
    assert grades["retrieval_mrr"].observed["mrr"] == 0.5
    assert grades["retrieval_mrr"].score == 0.5


def test_retrieval_mrr_fails_when_gold_doc_is_too_low():
    case = QACase(id="x", query="x", reference_answer="x", relevant_doc_ids=[7], retrieval_k=3, min_mrr=1.0)
    prediction = {
        "answer": "x",
        "retrieved_context": [
            {"source_id": 1, "content": "wrong", "metadata": {"url": "https://sf.gov/1"}},
            {"source_id": 7, "content": "right", "metadata": {"url": "https://sf.gov/7"}},
        ],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert not grades["retrieval_mrr"].passed


def test_citation_validity_requires_cited_docs_to_be_retrieved():
    case = QACase(id="x", query="x", reference_answer="x")
    prediction = {
        "answer": "x",
        "citations": [{"url": "https://sf.gov/not-retrieved"}],
        "retrieved_context": [{"source_id": 1, "content": "x", "metadata": {"url": "https://sf.gov/retrieved"}}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert not grades["citation_validity"].passed
    assert grades["citation_validity"].score == 0.0


def test_citation_url_recall_scores_partial_matches():
    case = QACase(
        id="x",
        query="x",
        reference_answer="x",
        expected_citation_urls=["https://sf.gov/a", "https://sf.gov/b"],
    )
    prediction = {
        "answer": "x",
        "citations": [{"url": "https://sf.gov/a"}],
        "retrieved_context": [{"content": "x", "metadata": {"url": "https://sf.gov/a"}}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert not grades["citation_urls"].passed
    assert grades["citation_urls"].score == 0.5


def test_citation_coverage_scores_numeric_date_claim_sentences():
    case = QACase(id="x", query="x", reference_answer="x", min_citation_coverage=1.0)
    prediction = {
        "answer": "The shortfall was $817.5 million [1]. The prior estimate was $875.9 million.",
        "retrieved_context": [{"content": "$817.5 million $875.9 million", "metadata": {}}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert not grades["citation_coverage"].passed
    assert grades["citation_coverage"].observed["coverage"] == 0.5


def test_extract_numeric_date_claims():
    claims = extract_numeric_date_claims(
        "The March 2025 update projected $817.5 million, 15%, and FY 2025-26."
    )
    assert "march 2025" in claims
    assert "817.5 million" in claims
    assert "15%" in claims
    assert "fy 2025-26" in claims


def test_numeric_date_claims_supported_by_retrieved_context_with_format_variation():
    case = QACase(id="x", query="x", reference_answer="x")
    prediction = {
        "answer": "The shortfall was $817.5 million in FY2025-26.",
        "retrieved_context": [{"content": "The update projected an 817.5 million shortfall in FY 2025-26."}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert grades["numeric_date_claims_supported"].passed
    assert grades["numeric_date_claims_supported"].observed["score"] == 1.0


def test_numeric_date_claims_use_threshold_when_not_all_claims_are_supported():
    case = QACase(id="x", query="x", reference_answer="x", min_numeric_date_support=0.8)
    prediction = {
        "answer": "The shortfall was $900 million in the March 2025 update.",
        "retrieved_context": [{"content": "The March 2025 update projected an $817.5 million shortfall."}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert not grades["numeric_date_claims_supported"].passed
    assert grades["numeric_date_claims_supported"].blocking is False
    assert "900 million" in grades["numeric_date_claims_supported"].observed["unsupported"]
    assert grades["numeric_date_claims_supported"].observed["score"] == 0.5


def test_numeric_date_claims_can_pass_with_partial_support_threshold():
    case = QACase(id="x", query="x", reference_answer="x", min_numeric_date_support=0.5)
    prediction = {
        "answer": "The shortfall was $900 million in the March 2025 update.",
        "retrieved_context": [{"content": "The March 2025 update projected an $817.5 million shortfall."}],
        "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
    }

    grades = {grade.name: grade for grade in run_deterministic_graders(case, prediction)}
    assert grades["numeric_date_claims_supported"].passed


def test_parse_args_supports_trials_limit_and_thresholds():
    args = parse_args(
        [
            "--cases",
            "evals/qa_cases.yml",
            "--trials",
            "3",
            "--limit",
            "2",
            "--threshold",
            "0.8",
            "--critical-floor",
            "0.4",
        ]
    )
    assert args.trials == 3
    assert args.limit == 2
    assert args.threshold == 0.8
    assert args.critical_floor == 0.4


def test_rubric_prompt_defines_score_endpoints():
    case = QACase(id="x", query="question", reference_answer="reference")
    prompt = _rubric_prompt("correctness", case, {"answer": "answer", "retrieved_context": []})
    assert "0.0 means the model output is the worst possible output" in prompt
    assert "1.0 means the model output is the best possible output" in prompt


def test_rubric_prompt_uses_general_correctness_and_groundedness_guidance():
    case = QACase(id="x", query="question", reference_answer="reference")
    correctness = _rubric_prompt("correctness", case, {"answer": "answer", "retrieved_context": []})
    groundedness = _rubric_prompt("groundedness", case, {"answer": "answer", "retrieved_context": []})

    assert "expected core facts" in correctness
    assert "not as the only acceptable wording or structure" in correctness
    assert "Judge only whether the actual answer's factual claims are supported by the retrieved context" in groundedness
    assert "Do not compare groundedness to the reference answer" in groundedness


def test_redact_secrets_nested_values():
    payload = {"OPENAI_API_KEY": "sk-test", "nested": {"token": "abc", "safe": "ok"}}
    assert redact_secrets(payload) == {
        "OPENAI_API_KEY": "[REDACTED]",
        "nested": {"token": "[REDACTED]", "safe": "ok"},
    }


class _NoCleanup:
    def cleanup(self):
        return None


def _make_db(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    local = sessionmaker(bind=engine, future=True)
    with local() as db:
        db.add(PlanVersion(memo_markdown="memo", model_used="test"))
        db.commit()
    engine.dispose()


def test_run_qa_eval_writes_artifacts_with_stubbed_target_and_judge(tmp_path, monkeypatch):
    db_path = tmp_path / "eval.db"
    _make_db(db_path)
    cases_path = tmp_path / "cases.yml"
    cases_path.write_text(
        """
version: 1
suite: stub_suite
default_plan_id: latest
cases:
  - id: stub_case
    query: "What is the timeline?"
    reference_answer: "December"
    expected_source_kinds: ["official"]
    expected_citation_urls: ["https://sf.gov/topics/budget"]
    must_include: ["December"]
    must_not_include: ["plan://"]
""",
        encoding="utf-8",
    )

    monkeypatch.setattr("evals.runner._copy_db_to_temp", lambda: (_NoCleanup(), db_path))

    def target_fn(_db, _plan, _thread_id, _case, trace):
        trace.record("target_call", {"selected_model": "stub"})
        return {
            "answer": "The timeline includes December.",
            "model_used": "stub",
            "citations": [{"url": "https://sf.gov/topics/budget", "source_kind": "official"}],
            "retrieved_context": [
                {
                    "content": "December timeline",
                    "metadata": {"source_kind": "official", "url": "https://sf.gov/topics/budget"},
                }
            ],
            "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
        }

    def judge_fn(_case, _prediction, _model_name, _threshold, trace):
        trace.append("judge_calls", {"rubric": "stub"})
        return {
            name: RubricGrade(score=0.9, passed=True, rationale="stub")
            for name in ("correctness", "answer_relevance", "groundedness", "retrieval_relevance")
        }

    result = run_qa_eval(
        cases_path,
        output_dir=tmp_path / "results",
        trials=2,
        limit=1,
        target_fn=target_fn,
        judge_fn=judge_fn,
    )

    assert result.suite_passed
    assert result.limited
    assert result.pass_at_k == 1.0
    assert result.pass_caret_k == 1.0
    assert result.trial_pass_rate == 1.0
    assert result.cases[0].pass_at_k is True
    assert result.cases[0].pass_caret_k is True
    assert result.cases[0].trial_pass_rate == 1.0
    run_dir = tmp_path / "results" / result.run_id
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "traces" / "stub_case" / "trial_1.json").exists()
    assert (tmp_path / "results" / "latest_run.txt").read_text(encoding="utf-8").strip() == result.run_id
    trace = json.loads((run_dir / "traces" / "stub_case" / "trial_1.json").read_text(encoding="utf-8"))
    assert trace["deterministic_grades"][0]["description"]


def test_run_qa_eval_reports_progress(tmp_path, monkeypatch):
    db_path = tmp_path / "eval.db"
    _make_db(db_path)
    cases_path = tmp_path / "cases.yml"
    cases_path.write_text(
        """
version: 1
suite: progress_suite
default_plan_id: latest
cases:
  - id: progress_case
    query: "What is the timeline?"
    reference_answer: "December"
    expected_source_kinds: ["official"]
    relevant_doc_ids: [1]
    must_include: ["December"]
""",
        encoding="utf-8",
    )

    monkeypatch.setattr("evals.runner._copy_db_to_temp", lambda: (_NoCleanup(), db_path))

    def target_fn(_db, _plan, _thread_id, _case, _trace):
        return {
            "answer": "December",
            "model_used": "stub",
            "citations": [],
            "retrieved_context": [{"source_id": 1, "content": "December", "metadata": {"source_kind": "official"}}],
            "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
        }

    def judge_fn(_case, _prediction, _model_name, _threshold, _trace):
        return {
            name: RubricGrade(score=0.9, passed=True, rationale="stub")
            for name in ("correctness", "answer_relevance", "groundedness", "retrieval_relevance")
        }

    lines: list[str] = []
    run_qa_eval(
        cases_path,
        output_dir=tmp_path / "results",
        trials=1,
        target_fn=target_fn,
        judge_fn=judge_fn,
        report=lines.append,
    )

    output = "\n".join(lines)
    assert "Starting eval suite 'progress_suite'" in output
    assert "Total trials: 1" in output
    assert "[1/1] progress_case" in output
    assert "trial 1: PASS" in output
    assert "citation_urls=" in output
    assert "recall=" in output
    assert "Suite result: PASS" in output
    assert "Multi-trial stats: pass@1=100.00% pass^1=100.00% trial_pass_rate=100.00%" in output


def test_non_blocking_numeric_date_failure_does_not_fail_trial_or_case(tmp_path, monkeypatch):
    db_path = tmp_path / "eval.db"
    _make_db(db_path)
    cases_path = tmp_path / "cases.yml"
    cases_path.write_text(
        """
version: 1
suite: warning_suite
default_plan_id: latest
cases:
  - id: warning_case
    query: "What is unavailable?"
    reference_answer: "Unavailable"
    expected_source_kinds: ["official"]
""",
        encoding="utf-8",
    )

    monkeypatch.setattr("evals.runner._copy_db_to_temp", lambda: (_NoCleanup(), db_path))

    def target_fn(_db, _plan, _thread_id, _case, _trace):
        return {
            "answer": "I cannot verify FY 2032-33, but FY 2025-26 is shown.",
            "model_used": "stub",
            "citations": [],
            "retrieved_context": [{"content": "FY 2025-26", "metadata": {"source_kind": "official"}}],
            "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
        }

    def judge_fn(_case, _prediction, _model_name, _threshold, _trace):
        return {
            name: RubricGrade(score=0.9, passed=True, rationale="stub")
            for name in ("correctness", "answer_relevance", "groundedness", "retrieval_relevance")
        }

    result = run_qa_eval(
        cases_path,
        output_dir=tmp_path / "results",
        target_fn=target_fn,
        judge_fn=judge_fn,
    )

    assert result.suite_passed
    assert result.cases[0].passed
    assert result.cases[0].deterministic_passed
    trial = result.cases[0].trials[0]
    assert trial.passed
    grades = {grade.name: grade for grade in trial.deterministic_grades}
    assert not grades["numeric_date_claims_supported"].passed
    assert grades["numeric_date_claims_supported"].blocking is False


def test_multi_trial_stats_capture_at_least_one_vs_all_success(tmp_path, monkeypatch):
    db_path = tmp_path / "eval.db"
    _make_db(db_path)
    cases_path = tmp_path / "cases.yml"
    cases_path.write_text(
        """
version: 1
suite: mixed_suite
default_plan_id: latest
cases:
  - id: mixed_case
    query: "What is mixed?"
    reference_answer: "Mixed"
    expected_source_kinds: ["official"]
""",
        encoding="utf-8",
    )

    monkeypatch.setattr("evals.runner._copy_db_to_temp", lambda: (_NoCleanup(), db_path))

    def target_fn(_db, _plan, _thread_id, _case, _trace):
        return {
            "answer": "Mixed",
            "model_used": "stub",
            "citations": [],
            "retrieved_context": [{"content": "Mixed", "metadata": {"source_kind": "official"}}],
            "debug": {"retrieval_policy": {"allowed_source_kinds": ["official"]}},
        }

    judge_calls = {"count": 0}

    def judge_fn(_case, _prediction, _model_name, _threshold, _trace):
        judge_calls["count"] += 1
        score = 0.9 if judge_calls["count"] == 1 else 0.2
        passed = score >= 0.7
        return {
            name: RubricGrade(score=score, passed=passed, rationale="stub")
            for name in ("correctness", "answer_relevance", "groundedness", "retrieval_relevance")
        }

    result = run_qa_eval(
        cases_path,
        output_dir=tmp_path / "results",
        trials=2,
        target_fn=target_fn,
        judge_fn=judge_fn,
    )

    case = result.cases[0]
    assert not result.suite_passed
    assert result.pass_at_k == 1.0
    assert result.pass_caret_k == 0.0
    assert result.trial_pass_rate == 0.5
    assert case.pass_at_k is True
    assert case.pass_caret_k is False
    assert case.trial_pass_rate == 0.5
