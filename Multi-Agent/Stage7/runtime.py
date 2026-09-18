from __future__ import annotations

from typing import Any

from Stage1.runtime import LlamaCppChatClient, assert_llama_server_ready, get_experiment_runtime_profile
from Stage6.runtime import (
    STAGE6_CONTEXT_SIZE,
    STAGE6_PARALLEL_SLOTS,
    STAGE6_SERVER_CONTEXT_SIZE,
)


STAGE7_RUNTIME_PROFILE_VERSION = "stage7-validation-runtime-128k-total-2slot-v1"
STAGE7_SERVER_CONTEXT_SIZE = STAGE6_SERVER_CONTEXT_SIZE
STAGE7_PARALLEL_SLOTS = STAGE6_PARALLEL_SLOTS
STAGE7_CONTEXT_SIZE = STAGE6_CONTEXT_SIZE
STAGE7_CLIENT_CONCURRENCY = 2


def get_stage7_runtime_profile() -> dict[str, Any]:
    profile = get_experiment_runtime_profile(expected_context_size=STAGE7_CONTEXT_SIZE)
    profile["expected_parallel_slots"] = STAGE7_PARALLEL_SLOTS
    profile["server_total_context_size"] = STAGE7_SERVER_CONTEXT_SIZE
    profile["client_request_concurrency"] = STAGE7_CLIENT_CONCURRENCY
    profile["stage7_runtime_profile_version"] = STAGE7_RUNTIME_PROFILE_VERSION
    return profile


def get_stage7_llm() -> LlamaCppChatClient:
    """Return the verifier client using exactly the existing two inference slots."""

    assert_llama_server_ready(
        expected_context_size=STAGE7_CONTEXT_SIZE,
        expected_parallel_slots=STAGE7_PARALLEL_SLOTS,
        profile_name="Stage 7 validation 128K-total / 2-slot",
    )
    return LlamaCppChatClient(request_concurrency=STAGE7_CLIENT_CONCURRENCY)
