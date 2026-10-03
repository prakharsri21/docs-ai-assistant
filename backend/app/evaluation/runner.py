"""
Course AI batch evaluator.

Important:
- Uses the exact same LangGraph as production/chat:
      from app.graph.workflow import graph
      graph.invoke({"query": question})
- Does NOT create a second RAG pipeline.
- Resumable: reuse the same --run-id after interruption.
- Saves each question immediately to JSONL.
- Writes a live summary/status so the frontend can show progress.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.graph.workflow import graph


TRACKS = ("basic_rag", "hybrid_rag", "okf_hybrid_rag")
ProgressCallback = Callable[[dict], None]


class BenchmarkQuestion(BaseModel):
    question: str
    context: str = ""
    ground_truth: str


class TrackJudgment(BaseModel):
    score: Literal[0, 1] = Field(
        description=(
            "1 if the answer is substantively correct relative to the reference "
            "answer; otherwise 0."
        )
    )
    reason: str = Field(description="Brief reason for the score.")


class EvaluationJudgment(BaseModel):
    basic_rag: TrackJudgment
    hybrid_rag: TrackJudgment
    okf_hybrid_rag: TrackJudgment


JUDGE_PROMPT = """You are the evaluator for a retrieval-augmented generation benchmark.

Evaluate three independently generated answers to the SAME question.

Use the reference answer as the primary correctness standard.

Scoring:
- score=1: the answer is substantively correct and addresses the question. Minor wording
  differences, reordering, or harmless omissions do not make it incorrect.
- score=0: the answer is materially incorrect, contradicts the reference, answers a
  different question, or is too incomplete to count as correct.

Do not reward an answer merely because it sounds plausible.
Do not use outside knowledge to replace missing information in the reference answer.
Judge each track independently. Do not rank the tracks.

Question:
{question}

Reference answer:
{ground_truth}

Basic RAG answer:
{basic_answer}

Hybrid RAG answer:
{hybrid_answer}

OKF + Hybrid RAG answer:
{okf_hybrid_answer}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_questions(path: Path) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(
                    BenchmarkQuestion.model_validate(json.loads(line))
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluation record at line {line_no}: {exc}"
                ) from exc

    if not questions:
        raise ValueError(f"No evaluation questions found in {path}")

    return questions


def load_existing_results(output_path: Path) -> dict[int, dict]:
    if not output_path.exists():
        return {}

    records: dict[int, dict] = {}

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records[int(record["question_id"])] = record
            except Exception:
                # Ignore malformed/truncated final lines so a rerun can recover.
                continue

    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_answer(track_payload: dict | None) -> str:
    return str((track_payload or {}).get("answer") or "")


def citation_valid(track_payload: dict | None) -> bool:
    validation = (track_payload or {}).get("citation_validation") or {}
    return bool(validation.get("valid", False))


def judge_answers(
    llm: ChatOpenAI,
    question: BenchmarkQuestion,
    final_response: dict,
) -> EvaluationJudgment:
    structured_llm = llm.with_structured_output(EvaluationJudgment)

    basic = final_response.get("basic_rag") or {}
    hybrid = final_response.get("hybrid_rag") or {}
    okf = final_response.get("okf_hybrid_rag") or {}

    prompt = JUDGE_PROMPT.format(
        question=question.question,
        ground_truth=question.ground_truth,
        basic_answer=safe_answer(basic),
        hybrid_answer=safe_answer(hybrid),
        okf_hybrid_answer=safe_answer(okf),
    )
    return structured_llm.invoke(prompt)


def build_summary(
    *,
    run_id: str,
    questions_total: int,
    records: list[dict],
) -> dict:
    valid = [
        record
        for record in records
        if all(record.get("tracks", {}).get(track) is not None for track in TRACKS)
    ]

    summary = {
        "run_id": run_id,
        "total_questions": questions_total,
        "completed_questions": len(valid),
        "failed_questions": max(questions_total - len(valid), 0),
        "completed_at": utc_now(),
        "tracks": {},
    }

    for track in TRACKS:
        evaluated = [
            record
            for record in valid
            if record.get("tracks", {}).get(track) is not None
        ]
        score = sum(
            int(record["tracks"][track]["score"]) for record in evaluated
        )
        citation_ok = sum(
            bool(record["tracks"][track]["citation_valid"])
            for record in evaluated
        )
        elapsed = [
            float(record["tracks"][track]["latency_ms"])
            for record in evaluated
            if record["tracks"][track].get("latency_ms") is not None
        ]

        summary["tracks"][track] = {
            "score": score,
            "total": questions_total,
            "evaluated": len(evaluated),
            "percentage": round((score / questions_total) * 100, 2)
            if questions_total
            else 0.0,
            "citation_valid": citation_ok,
            "citation_valid_percentage": round(
                (citation_ok / len(evaluated)) * 100, 2
            )
            if evaluated
            else 0.0,
            "average_latency_ms": round(sum(elapsed) / len(elapsed), 2)
            if elapsed
            else None,
        }

    return summary


def run(
    dataset_path: Path,
    output_dir: Path,
    *,
    run_id: str = "baseline",
    limit: int | None = None,
    resume: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    all_questions = load_questions(dataset_path)
    questions = all_questions[:limit] if limit is not None else all_questions

    run_dir = output_dir / run_id
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    latest_summary_path = output_dir / "latest_summary.json"
    status_path = run_dir / "status.json"

    existing = load_existing_results(results_path) if resume else {}

    # Successful result only: errors are intentionally retried on rerun.
    successful_existing = {
        qid: record
        for qid, record in existing.items()
        if all(record.get("tracks", {}).get(track) is not None for track in TRACKS)
    }

    judge_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    def emit(payload: dict) -> None:
        if progress_callback:
            progress_callback(payload)

    status_base = {
        "run_id": run_id,
        "status": "running",
        "total_questions": len(questions),
        "completed_questions": len(successful_existing),
        "failed_questions": 0,
        "current_question_id": None,
        "current_question": None,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "error": None,
    }
    write_json(status_path, status_base)

    emit(status_base)

    records = list(existing.values())
    records.sort(key=lambda item: int(item["question_id"]))

    for index, question in enumerate(questions, start=1):
        question_id = index

        if question_id in successful_existing:
            continue

        started = time.perf_counter()

        running_status = {
            **status_base,
            "status": "running",
            "completed_questions": len(successful_existing),
            "failed_questions": len(
                [
                    record
                    for record in records
                    if record.get("tracks")
                    and any(record["tracks"].get(track) is None for track in TRACKS)
                ]
            ),
            "current_question_id": question_id,
            "current_question": question.question,
            "updated_at": utc_now(),
            "error": None,
        }
        write_json(status_path, running_status)
        emit(running_status)

        try:
            # THIS IS THE SAME BACKEND USED BY THE APP.
            graph_result = graph.invoke({"query": question.question})
            final_response = graph_result.get("final_response")

            if not final_response:
                raise RuntimeError("Graph completed without final_response")

            judged = judge_answers(
                judge_llm,
                question,
                final_response,
            )

            total_elapsed_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            record = {
                "question_id": question_id,
                "question": question.question,
                "ground_truth": question.ground_truth,
                "total_elapsed_ms": total_elapsed_ms,
                "tracks": {},
            }

            for track in TRACKS:
                payload = final_response.get(track) or {}
                judgment = getattr(judged, track)

                record["tracks"][track] = {
                    "score": int(judgment.score),
                    "reason": judgment.reason,
                    "answer": safe_answer(payload),
                    "citations": payload.get("citations") or [],
                    "citation_valid": citation_valid(payload),
                    "citation_validation": payload.get("citation_validation") or {},
                    # graph.invoke runs the 3 branches as one parallel graph.
                    # This is the question-level end-to-end graph time.
                    "latency_ms": total_elapsed_ms,
                }

            append_jsonl(results_path, record)
            successful_existing[question_id] = record

        except Exception as exc:
            # Persist the failed attempt so it is inspectable, but do NOT count
            # it as completed; rerunning the same run_id will retry it.
            error_record = {
                "question_id": question_id,
                "question": question.question,
                "ground_truth": question.ground_truth,
                "total_elapsed_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
                "error": str(exc),
                "tracks": {track: None for track in TRACKS},
            }
            append_jsonl(results_path, error_record)

        records = list(load_existing_results(results_path).values())
        records.sort(key=lambda item: int(item["question_id"]))

        live_summary = build_summary(
            run_id=run_id,
            questions_total=len(questions),
            records=records,
        )
        write_json(summary_path, live_summary)
        write_json(latest_summary_path, live_summary)

        failed_count = sum(
            1
            for record in records
            if record.get("tracks")
            and any(record["tracks"].get(track) is None for track in TRACKS)
        )

        completed_count = len(
            [
                record
                for record in records
                if all(record.get("tracks", {}).get(track) is not None for track in TRACKS)
            ]
        )

        status_base = {
            **status_base,
            "completed_questions": completed_count,
            "failed_questions": failed_count,
            "current_question_id": question_id,
            "current_question": question.question,
            "updated_at": utc_now(),
            "error": None,
        }
        write_json(status_path, status_base)
        emit(status_base)

    final_records = list(load_existing_results(results_path).values())
    final_records.sort(key=lambda item: int(item["question_id"]))

    final_summary = build_summary(
        run_id=run_id,
        questions_total=len(questions),
        records=final_records,
    )
    write_json(summary_path, final_summary)
    write_json(latest_summary_path, final_summary)

    final_status = {
        **status_base,
        "status": "completed",
        "completed_questions": final_summary["completed_questions"],
        "failed_questions": final_summary["failed_questions"],
        "current_question_id": None,
        "current_question": None,
        "updated_at": utc_now(),
        "error": None,
    }
    write_json(status_path, final_status)
    emit(final_status)

    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/ml_dl_evaluation_200_clean.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
    )
    parser.add_argument("--run-id", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run(
        args.dataset,
        args.output_dir,
        run_id=args.run_id,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
