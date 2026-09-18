from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel


TModel = TypeVar("TModel", bound=BaseModel)
_REQUEST_GATE_LOCK = threading.Lock()
_REQUEST_GATES: dict[int, threading.BoundedSemaphore] = {}


def _request_gate(concurrency: int) -> threading.BoundedSemaphore:
    with _REQUEST_GATE_LOCK:
        gate = _REQUEST_GATES.get(concurrency)
        if gate is None:
            gate = threading.BoundedSemaphore(concurrency)
            _REQUEST_GATES[concurrency] = gate
        return gate


STRUCTURED_REPAIR_INSTRUCTION = (
    "Your previous JSON failed deterministic schema/semantic validation. Correct only the reported "
    "validation errors and return the complete corrected JSON object. Preserve the original case and substantive "
    "judgment unless the validation error shows that a cited claim is unsupported. Never infer or repair an evidence "
    "ID from its digest, prefix, spelling, or similarity. If the validation error lists allowed evidence IDs, choose "
    "only a complete exact string from that list; never combine the prefix of one ID with the digest of another. "
    "Re-read the supplied Stage 0 evidence records and use only an exact evidence_id that actually supports the "
    "literal claim. If no supplied record supports a claim, remove or rewrite that claim instead of inventing a "
    "citation. Validation error: {error}"
)


@dataclass(frozen=True)
class QwenRuntimeConfig:
    """Fixed main-experiment settings for local Qwen3.8-27B inference.

    Sampling values follow the Qwen3.8 thinking-mode recommendations. The
    llama.cpp server owns model loading, MTP, GPU offload and context settings;
    these per-request settings are repeated here so experiment behavior does
    not silently depend on server UI/default values.
    """

    api_base: str
    model: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    reasoning_effort: str = "xhigh"
    seed: int = 42
    timeout_seconds: int = 1800
    expected_llama_build: str = "b10919"
    expected_llama_commit: str = "d3146f2b5"
    expected_model_filename: str = "Qwen3.8-27B-Q6_K_L.gguf"
    expected_model_sha256: str = "e8750bb81ba49f90eb68df99776b250dbc14666043098952de6422cbecd77a21"
    expected_model_size_bytes: int = 24958908128
    model_source_repo: str = "bartowski/Qwen3.8-27B-GGUF"
    model_source_revision: str = "e4cc3ff3b37f5aabf253d8583b0b0247e8b25b70"
    quantization: str = "Q6_K_L"
    expected_context_size: int = 65536
    expected_parallel_slots: int = 2
    expected_gpu_layers: str = "all"
    batch_size: int = 2048
    ubatch_size: int = 512
    flash_attention: bool = True
    reasoning_budget: int = -1
    structured_validation_retries: int = 1
    mtp_enabled: bool = True
    mtp_draft_n_max: int = 4
    mtp_draft_p_min: float = 0.05
    target_kv_type: str = "q8_0"
    target_v_type: str = "q8_0"
    draft_kv_type: str = "q8_0"
    draft_v_type: str = "q8_0"


def get_runtime_config() -> QwenRuntimeConfig:
    return QwenRuntimeConfig(
        api_base=os.getenv("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080"),
        model=os.getenv("LLM_MODEL", "Qwen3.8-27B-Q6_K_L"),
        seed=int(os.getenv("LLM_SEED", "42")),
        timeout_seconds=int(os.getenv("LLAMA_SERVER_TIMEOUT_SECONDS", "1800")),
    )


def get_experiment_runtime_profile(*, expected_context_size: int | None = None) -> dict[str, Any]:
    """Return a paper-reportable inference profile.

    The active profile uses two 65,536-token inference slots backed by the
    same 131,072-token aggregate llama.cpp context allocation. Later stages
    may override only the required per-slot context capacity while preserving
    every other model/runtime setting in the identity.
    """
    raw = asdict(get_runtime_config())
    raw.pop("api_base", None)
    raw.pop("timeout_seconds", None)
    if expected_context_size is not None:
        raw["expected_context_size"] = int(expected_context_size)
    raw["structured_repair_prompt_sha256"] = hashlib.sha256(
        STRUCTURED_REPAIR_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    return raw


def get_legacy_one_slot_runtime_profile() -> dict[str, Any]:
    """Reconstruct the exact historical one-slot Stage 1/2 runtime identity.

    This is only for reusing already-frozen upstream artifacts created before
    the throughput-only two-slot migration. New inference uses the active
    two-slot profile.
    """

    raw = asdict(get_runtime_config())
    raw.pop("api_base", None)
    raw.pop("timeout_seconds", None)
    raw["expected_context_size"] = 131072
    raw["expected_parallel_slots"] = 1
    raw["structured_repair_prompt_sha256"] = hashlib.sha256(
        STRUCTURED_REPAIR_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    return raw


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    role_by_type = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }
    role = role_by_type.get(message.type)
    if role is None:
        raise ValueError(f"Unsupported message type for llama.cpp chat endpoint: {message.type!r}")
    return {"role": role, "content": message.content}


class _StructuredRunner:
    def __init__(
        self,
        client: "LlamaCppChatClient",
        schema: type[TModel],
        progress_label: str | None = None,
        semantic_validator: Callable[[TModel], TModel] | None = None,
        response_schema: dict[str, Any] | None = None,
    ):
        self.client = client
        self.schema = schema
        self.progress_label = progress_label
        self.semantic_validator = semantic_validator
        self.response_schema = response_schema

    def invoke(self, messages: list[BaseMessage]) -> TModel:
        return self.client._invoke_structured(
            messages,
            self.schema,
            progress_label=self.progress_label,
            semantic_validator=self.semantic_validator,
            response_schema=self.response_schema,
        )

    async def ainvoke(self, messages: list[BaseMessage]) -> TModel:
        return await asyncio.to_thread(self.invoke, messages)


class LlamaCppChatClient:
    """Minimal OpenAI-compatible llama.cpp client with fixed Qwen settings."""

    def __init__(
        self,
        config: QwenRuntimeConfig | None = None,
        *,
        request_concurrency: int = 2,
    ):
        self.config = config or get_runtime_config()
        if request_concurrency < 1:
            raise ValueError("request_concurrency must be >= 1")
        self.request_concurrency = int(request_concurrency)

    @property
    def endpoint(self) -> str:
        return f"{self.config.api_base.rstrip('/')}/v1/chat/completions"

    def _base_payload(self, messages: list[BaseMessage]) -> dict[str, Any]:
        cfg = self.config
        return {
            "model": cfg.model,
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "min_p": cfg.min_p,
            "presence_penalty": cfg.presence_penalty,
            "frequency_penalty": cfg.frequency_penalty,
            "repeat_penalty": cfg.repeat_penalty,
            "seed": cfg.seed,
            "reasoning_effort": cfg.reasoning_effort,
            "reasoning_format": "deepseek",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": cfg.reasoning_effort,
            },
            "stream": False,
        }

    def _progress_interval_seconds(self) -> float:
        try:
            return max(2.0, float(os.getenv("STAGE1_PROGRESS_INTERVAL_SECONDS", "10")))
        except ValueError:
            return 10.0

    def _read_slot_progress(self) -> dict[str, Any] | None:
        try:
            response = requests.get(
                f"{self.config.api_base.rstrip('/')}/slots",
                timeout=2,
            )
            response.raise_for_status()
            slots = response.json()
        except (requests.RequestException, ValueError):
            return None
        if not isinstance(slots, list):
            return None
        active = next(
            (
                slot
                for slot in slots
                if isinstance(slot, dict)
                and (slot.get("state") not in (None, "idle") or slot.get("is_processing") is True)
            ),
            slots[0] if slots and isinstance(slots[0], dict) else None,
        )
        if not active:
            return None

        def first_value(*keys: str) -> Any:
            for key in keys:
                value = active.get(key)
                if value is not None:
                    return value
            return None

        next_token = active.get("next_token")
        next_token_entry = (
            next_token[0]
            if isinstance(next_token, list) and next_token and isinstance(next_token[0], dict)
            else {}
        )

        # llama.cpp b10919 splits the request prompt into tokens reused from the
        # prompt cache plus tokens newly evaluated for the request. Either counter
        # may legitimately be zero, so neither is the input size by itself. Their
        # sum is the request prompt size and matches usage.prompt_tokens on the
        # completed response. n_prompt_tokens is the growing active slot/context
        # counter, while decode progress is next_token[0].n_decoded.
        context_tokens = first_value("n_prompt_tokens")
        prompt_cache_tokens = first_value("n_prompt_tokens_cache")
        prompt_processed_tokens = first_value("n_prompt_tokens_processed")
        generated_tokens = next_token_entry.get("n_decoded")
        if generated_tokens is None:
            generated_tokens = first_value("n_decoded", "tokens_predicted")

        input_tokens = None
        if (
            isinstance(prompt_cache_tokens, int)
            and isinstance(prompt_processed_tokens, int)
        ):
            input_tokens = prompt_cache_tokens + prompt_processed_tokens
        if input_tokens is None:
            input_tokens = first_value(
                "tokens_evaluated",
                "prompt_tokens",
                "n_prompt_tokens_input",
            )

        return {
            "state": first_value("state", "is_processing"),
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "context_tokens": context_tokens,
            "prompt_cache_tokens": prompt_cache_tokens,
            "prompt_processed_tokens": prompt_processed_tokens,
            "remaining_tokens": next_token_entry.get("n_remain", first_value("n_remaining")),
            "task_id": first_value("id_task"),
        }

    def _progress_monitor(
        self,
        *,
        stop_event: threading.Event,
        label: str,
        request_started: float,
    ) -> None:
        interval = self._progress_interval_seconds()
        while not stop_event.wait(interval):
            progress = self._read_slot_progress()
            elapsed = time.monotonic() - request_started
            if progress is None:
                print(f"[QWEN][{label}][+{elapsed:.0f}s] request active; slot metrics temporarily unavailable.", flush=True)
                continue
            fields = [f"state={progress['state']}"]
            if progress["input_tokens"] is not None:
                fields.append(f"input_tokens={progress['input_tokens']}")
            if progress["generated_tokens"] is not None:
                fields.append(f"generated_tokens={progress['generated_tokens']}")
            if progress["context_tokens"] is not None:
                fields.append(f"context_tokens={progress['context_tokens']}")
            if progress["remaining_tokens"] is not None:
                fields.append(f"remaining={progress['remaining_tokens']}")
            if progress["task_id"] is not None:
                fields.append(f"task={progress['task_id']}")
            print(f"[QWEN][{label}][+{elapsed:.0f}s] " + " ".join(fields), flush=True)

    @staticmethod
    def _print_completion_usage(*, data: dict[str, Any], label: str, elapsed: float) -> None:
        usage = data.get("usage") or {}
        message = (data.get("choices") or [{}])[0].get("message") or {}
        details = usage.get("completion_tokens_details") or {}
        parts = [f"elapsed={elapsed:.1f}s"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if usage.get(key) is not None:
                parts.append(f"{key}={usage[key]}")
        if details.get("reasoning_tokens") is not None:
            parts.append(f"reasoning_tokens={details['reasoning_tokens']}")
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            parts.append(f"reasoning_chars={len(reasoning_content)}")
        print(f"[QWEN][{label}] completed " + " ".join(parts), flush=True)

    def _post(
        self,
        payload: dict[str, Any],
        *,
        progress_label: str | None = None,
    ) -> dict[str, Any]:
        label = progress_label or "request"
        wait_started = time.monotonic()
        gate = _request_gate(self.request_concurrency)
        acquired = gate.acquire(blocking=False)
        if not acquired:
            print(
                f"[QWEN][{label}] waiting for one of {self.request_concurrency} client inference slots; "
                "all allowed concurrent requests are currently generating.",
                flush=True,
            )
            gate.acquire()
        try:
            waited = time.monotonic() - wait_started
            if waited >= 0.5:
                print(f"[QWEN][{label}] inference slot acquired after {waited:.1f}s queue wait.", flush=True)
            request_started = time.monotonic()
            print(
                f"[QWEN][{label}] generation started "
                f"(reasoning={self.config.reasoning_effort}, seed={self.config.seed}).",
                flush=True,
            )
            stop_event = threading.Event()
            monitor = threading.Thread(
                target=self._progress_monitor,
                kwargs={
                    "stop_event": stop_event,
                    "label": label,
                    "request_started": request_started,
                },
                name=f"stage1-progress-{label}",
                daemon=True,
            )
            monitor.start()
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                detail = ""
                response = getattr(exc, "response", None)
                if response is not None:
                    detail = f" Response body: {response.text[:2000]}"
                raise RuntimeError(
                    f"llama.cpp server request failed at {self.endpoint}: {exc}.{detail}"
                ) from exc
            finally:
                stop_event.set()
                monitor.join(timeout=2)
            data = response.json()
            if data.get("error"):
                raise RuntimeError(f"llama.cpp server returned an error: {data['error']}")
            if not data.get("choices"):
                raise RuntimeError(f"llama.cpp server returned no choices: {data}")
            self._print_completion_usage(
                data=data,
                label=label,
                elapsed=time.monotonic() - request_started,
            )
            return data
        finally:
            gate.release()

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        data = self._post(self._base_payload(messages), progress_label="unstructured")
        message = data["choices"][0]["message"]
        return AIMessage(
            content=message.get("content") or "",
            additional_kwargs={
                "reasoning_content": message.get("reasoning_content"),
                "llama_cpp_usage": data.get("usage"),
            },
        )

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        return await asyncio.to_thread(self.invoke, messages)

    def with_structured_output(
        self,
        schema: type[TModel],
        *,
        method: str = "json_schema",
        progress_label: str | None = None,
        semantic_validator: Callable[[TModel], TModel] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> _StructuredRunner:
        if method != "json_schema":
            raise ValueError("Stage 1 requires llama.cpp JSON-schema constrained output")
        return _StructuredRunner(
            self,
            schema,
            progress_label=progress_label,
            semantic_validator=semantic_validator,
            response_schema=response_schema,
        )

    def _invoke_structured(
        self,
        messages: list[BaseMessage],
        schema: type[TModel],
        *,
        progress_label: str | None = None,
        semantic_validator: Callable[[TModel], TModel] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> TModel:
        current_messages = list(messages)
        last_content = ""
        last_error: Exception | None = None
        for attempt in range(self.config.structured_validation_retries + 1):
            payload = self._base_payload(current_messages)
            payload["response_format"] = {
                "type": "json_object",
                "schema": response_schema or schema.model_json_schema(),
            }
            attempt_label = progress_label or schema.__name__
            if attempt:
                attempt_label = f"{attempt_label}:repair-{attempt}"
            data = self._post(payload, progress_label=attempt_label)
            content = data["choices"][0]["message"].get("content") or ""
            last_content = content
            try:
                parsed = schema.model_validate_json(content)
                if semantic_validator is not None:
                    parsed = semantic_validator(parsed)
                return parsed
            except Exception as exc:
                last_error = exc
                print(
                    f"[QWEN][{attempt_label}] structured schema/semantic validation failed "
                    f"(attempt {attempt + 1}/{self.config.structured_validation_retries + 1}): "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt >= self.config.structured_validation_retries:
                    break
                current_messages = [
                    *messages,
                    AIMessage(content=content),
                    HumanMessage(content=STRUCTURED_REPAIR_INSTRUCTION.format(error=str(exc))),
                ]

        raise RuntimeError(
            f"llama.cpp returned content that does not match {schema.__name__} after "
            f"{self.config.structured_validation_retries} semantic repair attempt(s): {last_content!r}"
        ) from last_error


def assert_llama_server_ready(
    *,
    expected_context_size: int | None = None,
    expected_parallel_slots: int | None = None,
    profile_name: str = "Stage 1",
) -> None:
    cfg = get_runtime_config()
    required_context_size = (
        cfg.expected_context_size
        if expected_context_size is None
        else int(expected_context_size)
    )
    required_parallel_slots = (
        cfg.expected_parallel_slots
        if expected_parallel_slots is None
        else int(expected_parallel_slots)
    )
    base = cfg.api_base.rstrip("/")
    health_url = f"{base}/health"
    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Qwen3.8 llama.cpp server is not ready. Start it with "
            "Multi-Agent/Stage1/start_qwen38.ps1 before running a new Stage 1 generation. "
            f"Health check failed at {health_url}: {exc}"
        ) from exc

    try:
        props_response = requests.get(f"{base}/props", timeout=10)
        props_response.raise_for_status()
        props = props_response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Could not verify llama.cpp runtime properties at {base}/props: {exc}") from exc

    model_path = str(props.get("model_path") or "")
    build_info = str(props.get("build_info") or "")
    defaults = props.get("default_generation_settings") or {}
    n_ctx = defaults.get("n_ctx")
    total_slots = props.get("total_slots")

    try:
        slots_response = requests.get(f"{base}/slots", timeout=10)
        slots_response.raise_for_status()
        slots = slots_response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Could not verify llama.cpp slot state at {base}/slots: {exc}") from exc

    violations: list[str] = []
    if not model_path.endswith(cfg.expected_model_filename):
        violations.append(f"model_path={model_path!r}")
    if cfg.expected_llama_build not in build_info or cfg.expected_llama_commit not in build_info:
        violations.append(f"build_info={build_info!r}")
    if total_slots != required_parallel_slots:
        violations.append(
            f"total_slots={total_slots!r} (expected {required_parallel_slots} for {profile_name})"
        )
    if len(slots) != required_parallel_slots:
        violations.append(f"slots={len(slots)!r} (expected {required_parallel_slots} inference slots)")
    else:
        slot_contexts = [slot.get("n_ctx") for slot in slots if isinstance(slot, dict)]
        if any(value != required_context_size for value in slot_contexts):
            violations.append(
                f"slot.n_ctx={slot_contexts!r} (expected {required_context_size} per slot for {profile_name})"
            )
        speculative = [slot.get("speculative") for slot in slots if isinstance(slot, dict)]
        if any(value is not True for value in speculative):
            violations.append(f"slot.speculative={speculative!r} (MTP must be active on every slot)")

    # b10919 may report /props default n_ctx as either per-slot capacity or the
    # aggregate server context when multiple slots are configured. The /slots
    # endpoint above is authoritative for request capacity. For one-slot
    # historical Stage 1/2 runs, preserve the old exact /props check as well.
    if required_parallel_slots == 1 and n_ctx != required_context_size:
        violations.append(
            f"n_ctx={n_ctx!r} (expected {required_context_size} for {profile_name})"
        )
    if violations:
        raise RuntimeError(
            f"llama.cpp runtime does not match the fixed {profile_name} profile: "
            + "; ".join(violations)
        )

