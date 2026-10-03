"""
In-process evaluation job manager.

A single job is allowed at a time per backend process. Results and progress
are persisted under data/evaluation/, so the UI can survive page refreshes.
If the server restarts mid-run, starting the same run_id resumes unfinished
questions because the runner skips successful records and retries failures.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.evaluation.runner import run


OUTPUT_DIR = Path("data/evaluation")
DATASET_PATH = Path("data/evaluation/ml_dl_evaluation_200_clean.jsonl")

_job_lock = threading.Lock()
_job_thread: threading.Thread | None = None
_current_run_id: str | None = None


def _status_path(run_id: str) -> Path:
    return OUTPUT_DIR / run_id / "status.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _set_progress(payload: dict[str, Any]) -> None:
    # runner.py already persists status; this hook exists so the API can remain
    # decoupled from the runner's implementation.
    global _current_run_id
    _current_run_id = payload.get("run_id") or _current_run_id


def _run_in_background(run_id: str, limit: int | None) -> None:
    global _job_thread, _current_run_id

    try:
        run(
            DATASET_PATH,
            OUTPUT_DIR,
            run_id=run_id,
            limit=limit,
            resume=True,
            progress_callback=_set_progress,
        )
    except Exception as exc:
        path = _status_path(run_id)
        payload = _read_json(path)
        payload.update(
            {
                "run_id": run_id,
                "status": "failed",
                "updated_at": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(),
                "error": str(exc),
            }
        )
        _write_json(path, payload)
    finally:
        with _job_lock:
            _job_thread = None
            _current_run_id = None


def start_evaluation(
    *,
    run_id: str = "baseline",
    limit: int | None = None,
) -> dict:
    global _job_thread, _current_run_id

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with _job_lock:
        if _job_thread is not None and _job_thread.is_alive():
            existing_status = _read_json(
                _status_path(_current_run_id or run_id)
            )
            return {
                "started": False,
                "already_running": True,
                "run_id": _current_run_id or run_id,
                "status": existing_status,
            }

        thread = threading.Thread(
            target=_run_in_background,
            args=(run_id, limit),
            name=f"evaluation-{run_id}",
            daemon=True,
        )
        _job_thread = thread
        _current_run_id = run_id
        thread.start()

    return {
        "started": True,
        "already_running": False,
        "run_id": run_id,
        "status": _read_json(_status_path(run_id)),
    }


def get_status(run_id: str = "baseline") -> dict:
    status = _read_json(_status_path(run_id))

    if not status:
        return {
            "run_id": run_id,
            "status": "not_started",
            "total_questions": 0,
            "completed_questions": 0,
            "failed_questions": 0,
            "current_question_id": None,
            "current_question": None,
            "error": None,
        }

    with _job_lock:
        alive = _job_thread is not None and _job_thread.is_alive()

    if status.get("status") == "running" and not alive:
        # The process may have restarted during a previous run. The persisted
        # results remain resumable.
        status = {
            **status,
            "status": "interrupted",
            "error": (
                status.get("error")
                or "The evaluation worker is not running. Re-run to resume."
            ),
        }

    return status


def get_summary(run_id: str = "baseline") -> dict:
    summary = _read_json(OUTPUT_DIR / run_id / "summary.json")
    if summary:
        return summary

    # Preserve the frontend's stable response shape before the first run.
    return {
        "run_id": run_id,
        "total_questions": 200,
        "completed_questions": 0,
        "failed_questions": 0,
        "completed_at": None,
        "tracks": {
            "basic_rag": {
                "score": 0,
                "total": 200,
                "evaluated": 0,
                "percentage": 0.0,
            },
            "hybrid_rag": {
                "score": 0,
                "total": 200,
                "evaluated": 0,
                "percentage": 0.0,
            },
            "okf_hybrid_rag": {
                "score": 0,
                "total": 200,
                "evaluated": 0,
                "percentage": 0.0,
            },
        },
    }
