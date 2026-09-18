from __future__ import annotations

from typing import Any

from Stage1.runtime import (
    LlamaCppChatClient,
    assert_llama_server_ready,
    get_experiment_runtime_profile,
)


STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION = "stage4-context-runtime-128k-total-2slot-v3"
STAGE4_SERVER_CONTEXT_SIZE = 131072
STAGE4_PARALLEL_SLOTS = 2
STAGE4_CONTEXT_SIZE = STAGE4_SERVER_CONTEXT_SIZE // STAGE4_PARALLEL_SLOTS
STAGE4_CLIENT_CONCURRENCY = 2


def get_stage4_context_runtime_profile() -> dict[str, Any]:
    """Return the exact runtime identity for contextual Stage 4 inference."""

    profile = get_experiment_runtime_profile(
        expected_context_size=STAGE4_CONTEXT_SIZE,
    )
    profile["expected_parallel_slots"] = STAGE4_PARALLEL_SLOTS
    profile["server_total_context_size"] = STAGE4_SERVER_CONTEXT_SIZE
    profile["client_request_concurrency"] = STAGE4_CLIENT_CONCURRENCY
    profile["stage4_context_runtime_profile_version"] = (
        STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION
    )
    return profile


def get_context_llm() -> LlamaCppChatClient:
    """Return the contextual Stage 4 client after enforcing the compact 128K server profile."""

    assert_llama_server_ready(
        expected_context_size=STAGE4_CONTEXT_SIZE,
        expected_parallel_slots=STAGE4_PARALLEL_SLOTS,
        profile_name="Stage 4 contextual 128K-total / 2-slot",
    )
    return LlamaCppChatClient(request_concurrency=STAGE4_CLIENT_CONCURRENCY)


__all__ = [
    "STAGE4_CONTEXT_RUNTIME_PROFILE_VERSION",
    "STAGE4_SERVER_CONTEXT_SIZE",
    "STAGE4_PARALLEL_SLOTS",
    "STAGE4_CONTEXT_SIZE",
    "STAGE4_CLIENT_CONCURRENCY",
    "get_stage4_context_runtime_profile",
    "get_context_llm",
]
