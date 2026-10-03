"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import styles from "./evaluation.module.css";

const TOTAL_QUESTIONS = 200;
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

type TrackKey = "basic_rag" | "hybrid_rag" | "okf_hybrid_rag";

type TrackData = {
  score: number;
  total: number;
  evaluated?: number;
  percentage?: number;
  citation_valid?: number;
  citation_valid_percentage?: number;
};

type EvaluationSummary = {
  run_id: string;
  total_questions: number;
  completed_questions: number;
  failed_questions: number;
  completed_at?: string | null;
  tracks: Record<TrackKey, TrackData>;
};

type EvaluationStatus = {
  run_id: string;
  status:
    | "not_started"
    | "running"
    | "completed"
    | "failed"
    | "interrupted";
  total_questions: number;
  completed_questions: number;
  failed_questions: number;
  current_question_id?: number | null;
  current_question?: string | null;
  started_at?: string | null;
  updated_at?: string | null;
  error?: string | null;
};

const TRACKS: Array<{
  key: TrackKey;
  title: string;
  method: string;
  className: string;
}> = [
  {
    key: "basic_rag",
    title: "Basic RAG",
    method: "Dense retrieval",
    className: "blue",
  },
  {
    key: "hybrid_rag",
    title: "Hybrid RAG",
    method: "Dense + BM25 + reranking",
    className: "green",
  },
  {
    key: "okf_hybrid_rag",
    title: "OKF + Hybrid",
    method: "OKF + hybrid + reranking",
    className: "purple",
  },
];

function clampScore(score: number, total: number) {
  return Math.max(0, Math.min(total, score));
}

function percentage(score: number, total: number) {
  return total ? Math.round((score / total) * 1000) / 10 : 0;
}

function buttonLabel(status: EvaluationStatus["status"]) {
  if (status === "running") return "Evaluation running…";
  if (status === "interrupted" || status === "failed") return "Resume baseline";
  if (status === "completed") return "Run baseline again";
  return "Run baseline evaluation";
}

export default function EvaluationPage() {
  const [status, setStatus] = useState<EvaluationStatus>({
    run_id: "baseline",
    status: "not_started",
    total_questions: TOTAL_QUESTIONS,
    completed_questions: 0,
    failed_questions: 0,
  });
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const loadEvaluation = useCallback(async () => {
    const [statusResponse, summaryResponse] = await Promise.all([
      fetch(`${API_BASE}/api/v1/evaluation/status?run_id=baseline`, {
        cache: "no-store",
      }),
      fetch(`${API_BASE}/api/v1/evaluation/summary?run_id=baseline`, {
        cache: "no-store",
      }),
    ]);

    if (statusResponse.ok) {
      setStatus((await statusResponse.json()) as EvaluationStatus);
    }

    if (summaryResponse.ok) {
      setSummary((await summaryResponse.json()) as EvaluationSummary);
    }
  }, []);

  useEffect(() => {
    loadEvaluation().catch(() => {
      setMessage("FastAPI is not reachable. Start the backend on port 8000.");
    });
  }, [loadEvaluation]);

  useEffect(() => {
    if (status.status !== "running") return;

    const timer = window.setInterval(() => {
      loadEvaluation().catch(() => {});
    }, 2000);

    return () => window.clearInterval(timer);
  }, [status.status, loadEvaluation]);

  const total = status.total_questions || summary?.total_questions || TOTAL_QUESTIONS;
  const completed = status.completed_questions || summary?.completed_questions || 0;
  const progress = useMemo(
    () => Math.min(100, percentage(completed, total)),
    [completed, total]
  );

  const hasScore = Boolean(
    summary && summary.completed_questions > 0
  );

  async function runBaseline() {
    setBusy(true);
    setMessage(null);

    try {
      const response = await fetch(`${API_BASE}/api/v1/evaluation/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          run_id: "baseline",
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Could not start the evaluation.");
      }

      setMessage(
        data.already_running
          ? "The baseline evaluation is already running."
          : "Baseline evaluation started."
      );

      await loadEvaluation();
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Could not start the evaluation."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className={styles.page}>
      <div className={styles.shell}>
        <header className={styles.topbar}>
          <Link href="/" className={styles.back}>
            <span aria-hidden="true">←</span>
            Chat
          </Link>

          <div className={styles.runState}>
            <span
              className={`${styles.statusDot} ${
                status.status === "running" ? styles.live : ""
              }`}
              aria-hidden="true"
            />
            <span>
              {status.status === "running"
                ? "Running"
                : status.status === "completed"
                  ? "Complete"
                  : status.status === "not_started"
                    ? "Ready"
                    : status.status}
            </span>
          </div>
        </header>

        <section className={styles.hero}>
          <div>
            <div className={styles.eyebrow}>BENCHMARK · V1</div>
            <h1 className={styles.title}>RAG Evaluation</h1>
            <p className={styles.subtitle}>
              Compare the three retrieval pipelines on the same 200-question benchmark.
            </p>
          </div>

          <button
            className={styles.runButton}
            type="button"
            onClick={runBaseline}
            disabled={busy || status.status === "running"}
          >
            <span aria-hidden="true">▶</span>
            {busy ? "Starting…" : buttonLabel(status.status)}
          </button>
        </section>

        <section className={styles.runnerCard}>
          <div className={styles.runnerTop}>
            <div>
              <div className={styles.runnerLabel}>BASELINE RUN</div>
              <div className={styles.runnerTitle}>
                {completed} <span>/ {total}</span> questions evaluated
              </div>
            </div>
            <div className={styles.runnerPercent}>{progress}%</div>
          </div>

          <div className={styles.progressTrack}>
            <div
              className={styles.progressFill}
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className={styles.runnerBottom}>
            <span>
              {status.current_question_id
                ? `Q${status.current_question_id} · ${status.current_question ?? ""}`
                : status.status === "not_started"
                  ? "Ready to start"
                  : status.status === "completed"
                    ? "All benchmark questions completed"
                    : "Waiting for worker"}
            </span>

            {status.failed_questions > 0 && (
              <span className={styles.failureText}>
                {status.failed_questions} failed
              </span>
            )}
          </div>

          {(message || status.error) && (
            <div className={status.error ? styles.error : styles.message}>
              {status.error || message}
            </div>
          )}
        </section>

        <section className={styles.scoreSection}>
          <div className={styles.sectionHeader}>
            <div>
              <div className={styles.sectionLabel}>RESULTS</div>
              <h2>Track scores</h2>
            </div>

            <span className={styles.questionPill}>{total} questions</span>
          </div>

          <div className={styles.scoreGrid}>
            {TRACKS.map((track) => {
              const data = summary?.tracks?.[track.key];
              const score = clampScore(data?.score ?? 0, total);
              const pct = hasScore ? percentage(score, total) : null;

              return (
                <article
                  key={track.key}
                  className={`${styles.scoreCard} ${styles[track.className]}`}
                >
                  <div className={styles.cardHeader}>
                    <div>
                      <div className={styles.cardLabel}>{track.method}</div>
                      <h3>{track.title}</h3>
                    </div>
                    <span className={styles.cardDot} aria-hidden="true" />
                  </div>

                  <div className={styles.scoreRow}>
                    <span className={styles.score}>{hasScore ? score : "—"}</span>
                    <span className={styles.total}>/ {total}</span>
                  </div>

                  <div className={styles.percent}>
                    {pct == null ? "Awaiting run" : `${pct}%`}
                  </div>

                  <div className={styles.scoreTrack} aria-hidden="true">
                    <div
                      className={styles.scoreFill}
                      style={{ width: `${pct ?? 0}%` }}
                    />
                  </div>

                  <div className={styles.cardFooter}>
                    <span>
                      {data?.evaluated ?? 0} / {total} evaluated
                    </span>
                    <span>
                      {data?.citation_valid_percentage != null
                        ? `${data.citation_valid_percentage}% citations`
                        : "—"}
                    </span>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <footer className={styles.footer}>
          <div className={styles.footerIcon}>i</div>
          <p>
            A question earns 1 point when the generated answer is judged
            substantively correct against the reference answer. Scores are
            displayed as <code>correct / 200</code>.
          </p>
        </footer>
      </div>
    </main>
  );
}
