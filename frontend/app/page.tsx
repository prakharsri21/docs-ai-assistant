"use client";

import { FormEvent, useState } from "react";

type Citation = {
  chunk_id: string;
  source_file: string;
  page: number;
};

type CitationValidation = {
  valid: boolean;
  errors: string[];
  validated_citations: Citation[];
};

type RAGResponse = {
  answer: string;
  citations: Citation[];
  citation_validation: CitationValidation;
};

type ChatResponse = {
  query: string;
  basic_rag: RAGResponse;
  hybrid_rag: RAGResponse;
  okf_hybrid_rag: RAGResponse;
};

type TrackName =
  | "basic_rag"
  | "hybrid_rag"
  | "okf_hybrid_rag";

type TrackState = {
  answer: string;
  ttftMs: number | null;
  totalMs: number | null;
  citations: Citation[];
  citationValid: boolean | null;
  status: "waiting" | "streaming" | "complete" | "error";
};

type AssistantMessage = {
  role: "assistant";
  tracks: Record<TrackName, TrackState>;
};

type ChatMessage =
  | {
      role: "user";
      content: string;
    }
  | AssistantMessage;

const EMPTY_TRACK = (): TrackState => ({
  answer: "",
  ttftMs: null,
  totalMs: null,
  citations: [],
  citationValid: null,
  status: "waiting",
});

const EMPTY_TRACKS = (): Record<TrackName, TrackState> => ({
  basic_rag: EMPTY_TRACK(),
  hybrid_rag: EMPTY_TRACK(),
  okf_hybrid_rag: EMPTY_TRACK(),
});

const TRACK_LABELS: Record<TrackName, string> = {
  basic_rag: "Basic RAG",
  hybrid_rag: "Hybrid RAG",
  okf_hybrid_rag: "OKF + Hybrid RAG",
};

export default function Home() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    const query = message.trim();

    if (!query || loading) {
      return;
    }

    setError("");
    setMessage("");

    const assistantMessage: AssistantMessage = {
      role: "assistant",
      tracks: EMPTY_TRACKS(),
    };

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: query,
      },
      assistantMessage,
    ]);

    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/v1/chat/stream",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: query,
          }),
        },
      );

      if (!response.ok) {
        const errorText = await response.text();

        throw new Error(
          `API request failed (${response.status}): ${errorText}`,
        );
      }

      if (!response.body) {
        throw new Error(
          "The server did not provide a streaming response.",
        );
      }

      await consumeStream(
        response.body,
        (event, data) => {
          handleStreamEvent(
            event,
            data,
            setMessages,
          );
        },
      );
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "Something went wrong.";

      setError(errorMessage);

      setMessages((previous) => {
        const updated = [...previous];
        const last = updated[updated.length - 1];

        if (
          last?.role === "assistant"
        ) {
          last.tracks.basic_rag.status = "error";
          last.tracks.hybrid_rag.status = "error";
          last.tracks.okf_hybrid_rag.status = "error";
        }

        return updated;
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-container">
        <header className="header">
          <div>
            <p className="eyebrow">COURSE AI</p>

            <h1>AI Study Assistant</h1>

            <p className="subtitle">
              Compare three retrieval pipelines against the same
              course question.
            </p>
          </div>
        </header>

        <section className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">
                AI
              </div>

              <h2>Ask a question</h2>

              <p>
                Your question is answered through Basic RAG,
                Hybrid RAG, and OKF + Hybrid RAG in parallel.
              </p>

              <div className="example">
                “What happens when the batch size is one in
                stochastic gradient descent?”
              </div>
            </div>
          )}

          {messages.map((item, index) => {
            if (item.role === "user") {
              return (
                <div
                  className="message-row user-row"
                  key={index}
                >
                  <div className="user-message">
                    {item.content}
                  </div>
                </div>
              );
            }

            return (
              <div
               className="assistant-block"
                key={index}
              >
                <div className="assistant-label">
                  AI RESPONSE
                </div>

                <div className="answer-grid">
                  <AnswerCard
                    title={TRACK_LABELS.basic_rag}
                    track={item.tracks.basic_rag}
                  />

                  <AnswerCard
                    title={TRACK_LABELS.hybrid_rag}
                    track={item.tracks.hybrid_rag}
                  />

                  <AnswerCard
                    title={TRACK_LABELS.okf_hybrid_rag}
                    track={item.tracks.okf_hybrid_rag}
                  />
                </div>
              </div>
            );
          })}

          {error && (
            <div className="error-card">
              <strong>Request failed</strong>
              <span>{error}</span>
            </div>
          )}
        </section>

        <form
          className="composer"
          onSubmit={handleSubmit}
        >
          <textarea
            value={message}
            onChange={(event) =>
              setMessage(event.target.value)
            }
            placeholder="Ask a question about the course material..."
            rows={2}
            disabled={loading}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey
              ) {
                event.preventDefault();

                event.currentTarget.form?.requestSubmit();
              }
            }}
          />

          <button
            type="submit"
            disabled={
              loading ||
              !message.trim()
            }
          >
            {loading ? "Running..." : "Ask"}
          </button>
        </form>
      </section>
    </main>
  );
}


function AnswerCard({
  title,
  track,
}: {
  title: string;
  track: TrackState;
}) {
  const isVerified =
    track.citationValid === true;

  const isStreaming =
    track.status === "streaming";

  const isComplete =
    track.status === "complete";

  return (
    <article className="answer-card">
      <div className="card-header">
        <div>
          <h3>{title}</h3>

          <span
            className={
              isVerified
                ? "verified"
                : "track-status"
            }
          >
            {isVerified
              ? "✓ Citation verified"
              : isComplete
                ? "No verified citation"
                : isStreaming
                  ? "● Generating"
                  : "Waiting"}
          </span>
        </div>
      </div>

      <p className="answer-text">
        {track.answer ||
          (isStreaming
            ? "Generating answer..."
            : "Waiting for this pipeline...")}
      </p>

      {(track.ttftMs !== null ||
        track.totalMs !== null) && (
        <div className="latency">
          <div className="latency-item">
            <span>TTFT</span>

            <strong>
          {formatLatency(track.ttftMs)}
            </strong>
          </div>

          <div className="latency-item">
            <span>Total</span>

            <strong>
              {track.totalMs !== null
                ? formatLatency(track.totalMs)
                : "—"}
            </strong>
          </div>
        </div>
      )}

      <div className="citations">
        <span className="citation-label">
          SOURCES
        </span>

        {track.citations.length === 0 ? (
          <span className="no-citations">
            {isComplete
              ? "No citations returned"
              : "Waiting for citations..."}
          </span>
        ) : (
          track.citations.map((citation) => (
            <div
              className="citation"
              key={citation.chunk_id}
            >
              <span>
                {citation.source_file}
              </span>

              <span>
                Page {citation.page}
              </span>
            </div>
          ))
      )}
      </div>
    </article>
  );
}


function formatLatency(
  value: number | null,
): string {
  if (value === null) {
    return "—";
  }

  return `${(value / 1000).toFixed(2)}s`;
}


async function consumeStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (
    event: string,
    data: Record<string, unknown>,
  ) => void,
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  while (true) {
    const { value, done } =
      await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(
      value,
      {
        stream: true,
      },
    );

    const events =
      buffer.split("\n\n");

    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      parseSSEEvent(
        rawEvent,
        onEvent,
      );
    }
  }

  if (buffer.trim()) {
    parseSSEEvent(
      buffer,
      onEvent,
    );
  }
}


function parseSSEEvent(
  rawEvent: string,
  onEvent: (
    event: string,
    data: Record<string, unknown>,
  ) => void,
) {
  let eventName = "";
  let data = "";

  const lines =
    rawEvent.split("\n");

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName =
        line.slice(6).trim();
    }

    if (line.startsWith("data:")) {
      data += line
        .slice(5)
        .trim();
    }
  }

  if (!eventName || !data) {
    return;
  }

  try {
    const parsed =
      JSON.parse(data);

    onEvent(
      eventName,
      parsed,
    );
  } catch {
    console.error(
      "Failed to parse SSE data:",
      data,
    );
  }
}


function handleStreamEvent(
  event: string,
  data: Record<string, unknown>,
  setMessages: React.Dispatch<
    React.SetStateAction<ChatMessage[]>
  >,
) {
  if (
    event === "track_started"
  ) {
    const track =
      data.track as TrackName;

    const ttftMs =
      typeof data.ttft_ms === "number"
        ? data.ttft_ms
        : null;

    updateLatestAssistant(
      setMessages,
      (assistant) => {
        assistant.tracks[track] = {
          ...assistant.tracks[track],
          ttftMs,
          status: "streaming",
        };
      },
    );

    return;
  }

  if (event === "token") {
    const track =
      data.track as TrackName;

    const text =
      typeof data.text === "string"
        ? data.text
        : "";

    updateLatestAssistant(
      setMessages,
      (assistant) => {
        assistant.tracks[track] = {
          ...assistant.tracks[track],
          answer:
            assistant.tracks[track].answer +
            text,
          status: "streaming",
        };
      },
    );

    return;
  }

  if (
    event === "track_completed"
  ) {
    const track =
      data.track as TrackName;

    const totalMs =
      typeof data.total_ms === "number"
        ? data.total_ms
        : null;

    const ttftMs =
      typeof data.ttft_ms === "number"
        ? data.ttft_ms
        : null;

    updateLatestAssistant(
      setMessages,
      (assistant) => {
        assistant.tracks[track] = {
          ...assistant.tracks[track],
          totalMs,
          ttftMs,
          status: "complete",
        };
      },
    );

    return;
  }

  if (event === "final") {
    const response =
      data.response as ChatResponse;

    updateLatestAssistant(
      setMessages,
      (assistant) => {
        const map: Array<
          [TrackName, RAGResponse]
        > = [
          [
            "basic_rag",
            response.basic_rag,
          ],
          [
            "hybrid_rag",
            response.hybrid_rag,
          ],
          [
            "okf_hybrid_rag",
            response.okf_hybrid_rag,
          ],
        ];

        for (const [
          track,
          result,
        ] of map) {
          assistant.tracks[track] = {
            ...assistant.tracks[track],
            answer: result.answer,
            citations:
              result.citations,
            citationValid:
              result
                .citation_validation
                .valid,
            status: "complete",
          };
        }
      },
    );

    return;
  }

  if (event === "error") {
    const message =
      typeof data.message === "string"
        ? data.message
        : "Streaming failed.";

    console.error(
      "Stream error:",
      message,
    );

    return;
  }
}


function updateLatestAssistant(
  setMessages: React.Dispatch<
    React.SetStateAction<ChatMessage[]>
  >,
  updater: (
    assistant: AssistantMessage,
  ) => void,
) {
  setMessages((previous) => {
    const updated = [...previous];

    for (
      let index =
        updated.length - 1;
      index >= 0;
      index--
    ) {
      const item =
        updated[index];

      if (
        item.role ===
        "assistant"
      ) {
        const assistant = {
          ...item,
          tracks: {
            basic_rag: {
              ...item.tracks.basic_rag,
            },
            hybrid_rag: {
              ...item.tracks.hybrid_rag,
            },
            okf_hybrid_rag: {
              ...item.tracks.okf_hybrid_rag,
            },
          },
        };

        updater(assistant);

        updated[index] =
          assistant;

        break;
      }
    }

    return updated;
  });
}
