from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.evaluation.service import get_status, get_summary, start_evaluation


router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


class EvaluationRunRequest(BaseModel):
    run_id: str = Field(
        default="baseline",
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    limit: int | None = Field(default=None, ge=1, le=200)


@router.post("/run")
def start_run(request: EvaluationRunRequest) -> dict:
    try:
        return start_evaluation(
            run_id=request.run_id,
            limit=request.limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
def status(run_id: str = "baseline") -> dict:
    return get_status(run_id)


@router.get("/summary")
def summary(run_id: str = "baseline") -> dict:
    return get_summary(run_id)
