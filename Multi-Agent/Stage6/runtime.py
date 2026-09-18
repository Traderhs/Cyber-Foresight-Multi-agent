from __future__ import annotations

from typing import Any

from Stage1.runtime import LlamaCppChatClient, assert_llama_server_ready, get_experiment_runtime_profile
from Stage4.context_runtime import (
    STAGE4_CONTEXT_SIZE,
    STAGE4_PARALLEL_SLOTS,
    STAGE4_SERVER_CONTEXT_SIZE,
)


STAGE6_RUNTIME_PROFILE_VERSION = "stage6-synthesis-runtime-128k-total-2slot-v1"
STAGE6_SERVER_CONTEXT_SIZE = STAGE4_SERVER_CONTEXT_SIZE
STAGE6_PARALLEL_SLOTS = STAGE4_PARALLEL_SLOTS
STAGE6_CONTEXT_SIZE = STAGE4_CONTEXT_SIZE
STAGE6_CLIENT_CONCURRENCY = 2


def get_stage6_runtime_profile() -> dict[str, Any]:
    profile = get_experiment_runtime_profile(expected_context_size=STAGE6_CONTEXT_SIZE)
    profile["expected_parallel_slots"] = STAGE6_PARALLEL_SLOTS
    profile["server_total_context_size"] = STAGE6_SERVER_CONTEXT_SIZE
    profile["client_request_concurrency"] = STAGE6_CLIENT_CONCURRENCY
    profile["stage6_runtime_profile_version"] = STAGE6_RUNTIME_PROFILE_VERSION
    return profile


def get_synthesis_llm() -> LlamaCppChatClient:
    assert_llama_server_ready(
        expected_context_size=STAGE6_CONTEXT_SIZE,
        expected_parallel_slots=STAGE6_PARALLEL_SLOTS,
        profile_name="Stage 6 synthesis 128K-total / 2-slot",
    )
    return LlamaCppChatClient(request_concurrency=STAGE6_CLIENT_CONCURRENCY)

