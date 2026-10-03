from __future__ import annotations

import json
import time
from collections.abc import Iterator

from app.schemas.chat import ChatResponse


TRACKS = {
    "basic_answer": "basic_rag",
    "hybrid_answer": "hybrid_rag",
    "okf_answer": "okf_hybrid_rag",
}


def generate_chat_response(message: str) -> ChatResponse:
    """
    Execute the normal LangGraph workflow and return
    the complete validated response.
    """

    from app.graph.workflow import graph

    result = graph.invoke(
        {
            "query": message,
        }
    )

    final_response = result.get("final_response")

    if not final_response:
        raise RuntimeError(
            "RAG graph completed without a final response."
        )

    return ChatResponse.model_validate(
        final_response
    )


def _sse_event(
    event: str,
    data: dict,
) -> str:
    """
    Format one Server-Sent Event.
    """

    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


class AnswerExtractor:
    """
    Extract the `answer` value from streamed JSON fragments
    produced by structured output.

    The LLM streams something like:

        {"answer":"Deep learning is ...","citations":[...]}

    We expose only the answer text to the frontend.
    """

    MARKER = '"answer":"'
    END_MARKER = '","citations"'

    def __init__(self) -> None:
        self.buffer = ""
        self.started = False
        self.finished = False

    @staticmethod
    def _decode(fragment: str) -> str:
        """
        Decode common JSON string escapes while preserving
        normal Unicode text.
        """

        try:
            return json.loads(
                '"' + fragment + '"'
            )
        except json.JSONDecodeError:
            return (
                fragment
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )

    def feed(self, chunk: str) -> str:
        if not chunk or self.finished:
            return ""

        self.buffer += chunk

        # --------------------------------------------------------------
        # Find the beginning of the answer field.
        # --------------------------------------------------------------

        if not self.started:
            start = self.buffer.find(self.MARKER)

            if start == -1:
                # Preserve enough tail data in case the marker is split
                # across two incoming chunks.
                keep = len(self.MARKER) - 1

                if len(self.buffer) > keep:
                    self.buffer = self.buffer[-keep:]

                return ""

            self.started = True
            self.buffer = self.buffer[
                start + len(self.MARKER) :
            ]

        # --------------------------------------------------------------
        # Find the end of the answer field.
        # --------------------------------------------------------------

        end = self.buffer.find(
            self.END_MARKER
        )

        if end != -1:
            raw_fragment = self.buffer[:end]

            self.finished = True
            self.buffer = ""

            return self._decode(
                raw_fragment
            )

        # --------------------------------------------------------------
        # Emit a safe prefix while retaining enough characters to
        # detect the END_MARKER if it is split across chunks.
        # --------------------------------------------------------------

        keep = len(self.END_MARKER) - 1

        if len(self.buffer) <= keep:
            return ""

        raw_fragment = self.buffer[:-keep]

        self.buffer = self.buffer[-keep:]

        return self._decode(
            raw_fragment
        )


def stream_chat_response(
    message: str,
) -> Iterator[str]:
    """
    Stream the three RAG answers and per-track latency.

    Events:
        token
        track_completed
        final
        error
        done
    """

    from app.graph.workflow import graph

    request_start = time.perf_counter()

    extractors = {
        track: AnswerExtractor()
        for track in TRACKS.values()
    }

    ttft_ms: dict[str, float | None] = {
        track: None
        for track in TRACKS.values()
    }

    completed_tracks: set[str] = set()

    try:
        for part in graph.stream(
            {"query": message},
            stream_mode=[
                "messages",
                "updates",
            ],
            version="v2",
        ):
            elapsed_ms = (
                time.perf_counter()
                - request_start
            ) * 1000

            # ----------------------------------------------------------
            # LLM message chunks
            # ----------------------------------------------------------

            if part["type"] == "messages":
                msg, metadata = part["data"]

                node_name = metadata.get(
                    "langgraph_node"
                )

                track = TRACKS.get(
                    node_name
                )

                if not track:
                    continue

                content = getattr(
                    msg,
                    "content",
                    "",
                )

                if not isinstance(
                    content,
                    str,
                ):
                    continue

                text = extractors[
                    track
                ].feed(content)

                if not text:
                    continue

                # First visible answer token = TTFT.
                if ttft_ms[track] is None:
                    ttft_ms[track] = elapsed_ms

                    yield _sse_event(
                        "track_started",
                        {
                            "track": track,
                            "ttft_ms": round(
                                elapsed_ms,
                                2,
                            ),
                        },
                    )

                yield _sse_event(
                    "token",
                    {
                        "track": track,
                        "text": text,
                    },
                )

            # ----------------------------------------------------------
            # Graph node updates
            # ----------------------------------------------------------

            elif part["type"] == "updates":
                update_data = part["data"]

                # Detect completion of answer nodes.
                for node_name in update_data:
                    track = TRACKS.get(
                        node_name
                    )

                    if (
                        track
                        and track
                        not in completed_tracks
                    ):
                        completed_tracks.add(
                            track
                        )

                        yield _sse_event(
                            "track_completed",
                            {
                                "track": track,
                                "ttft_ms": (
                                    round(
                                        ttft_ms[track],
                                        2,
                                    )
                                    if ttft_ms[track]
                                    is not None
                                    else None
                                ),
                                "total_ms": round(
                                    elapsed_ms,
                                    2,
                                ),
                            },
                        )

                # ------------------------------------------------------
                # Finalize node
                # ------------------------------------------------------

                if "finalize" in update_data:
                    finalize_update = (
                        update_data["finalize"]
                    )

                    final_response = (
                        finalize_update.get(
                            "final_response"
                        )
                    )

                    if final_response:
                        yield _sse_event(
                            "final",
                            {
                                "response": final_response,
                            },
                        )

        total_ms = (
            time.perf_counter()
            - request_start
        ) * 1000

        yield _sse_event(
            "done",
            {
                "total_ms": round(
                    total_ms,
                    2,
                ),
            },
        )

    except Exception as exc:
        yield _sse_event(
            "error",
            {
                "message": str(exc),
            },
        )