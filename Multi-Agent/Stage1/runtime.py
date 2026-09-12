from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel


TModel = TypeVar("TModel", bound=BaseModel)
_REQUEST_LOCK = threading.Lock()


STRUCTURED_REPAIR_INSTRUCTION = (
    "Your previous JSON failed deterministic schema/semantic validation. Correct only the reported "
    "validation errors and return the complete corrected JSON object. Preserve the original case, "
    "evidence boundary, and substantive judgment. Do not invent facts or evidence IDs. Validation error: {error}"
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
    expected_context_size: int = 131072
    expected_parallel_slots: int = 1
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


def get_experiment_runtime_profile() -> dict[str, Any]:
    """Return the fixed, paper-reportable Stage 1 inference profile."""
    raw = asdict(get_runtime_config())
    raw.pop("api_base", None)
    raw.pop("timeout_seconds", None)
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
    ):
        self.client = client
        self.schema = schema
        self.progress_label = progress_label

    def invoke(self, messages: list[BaseMessage]) -> TModel:
        return self.client._invoke_structured(
            messages,
            self.schema,
            progress_label=self.progress_label,
        )

    async def ainvoke(self, messages: list[BaseMessage]) -> TModel:
        return await asyncio.to_thread(self.invoke, messages)


class LlamaCppChatClient:
    """Minimal OpenAI-compatible llama.cpp client with fixed Qwen settings."""

    def __init__(self, config: QwenRuntimeConfig | None = None):
        self.config = config or get_runtime_config()

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

        return {
            "state": first_value("state", "is_processing"),
            "prompt_tokens": first_value("n_prompt_tokens", "tokens_evaluated", "n_prompt_tokens_processed"),
            "generated_tokens": first_value("n_decoded", "tokens_predicted"),
            "remaining_tokens": first_value("n_remaining"),
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
            if progress["prompt_tokens"] is not None:
                fields.append(f"prompt_tokens={progress['prompt_tokens']}")
            if progress["generated_tokens"] is not None:
                fields.append(f"generated_tokens={progress['generated_tokens']}")
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
        if _REQUEST_LOCK.locked():
            print(
                f"[QWEN][{label}] waiting for the single pinned inference slot; "
                "another independent critic is currently generating.",
                flush=True,
            )
        with _REQUEST_LOCK:
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
    ) -> _StructuredRunner:
        if method != "json_schema":
            raise ValueError("Stage 1 requires llama.cpp JSON-schema constrained output")
        return _StructuredRunner(self, schema, progress_label=progress_label)

    def _invoke_structured(
        self,
        messages: list[BaseMessage],
        schema: type[TModel],
        *,
        progress_label: str | None = None,
    ) -> TModel:
        current_messages = list(messages)
        last_content = ""
        last_error: Exception | None = None
        for attempt in range(self.config.structured_validation_retries + 1):
            payload = self._base_payload(current_messages)
            payload["response_format"] = {
                "type": "json_object",
                "schema": schema.model_json_schema(),
            }
            attempt_label = progress_label or schema.__name__
            if attempt:
                attempt_label = f"{attempt_label}:repair-{attempt}"
            data = self._post(payload, progress_label=attempt_label)
            content = data["choices"][0]["message"].get("content") or ""
            last_content = content
            try:
                return schema.model_validate_json(content)
            except Exception as exc:
                last_error = exc
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


def assert_llama_server_ready() -> None:
    cfg = get_runtime_config()
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
    if n_ctx != cfg.expected_context_size:
        violations.append(f"n_ctx={n_ctx!r}")
    if total_slots != cfg.expected_parallel_slots:
        violations.append(
            f"total_slots={total_slots!r} (expected {cfg.expected_parallel_slots} for current CUDA/MTP safety)"
        )
    if len(slots) != cfg.expected_parallel_slots:
        violations.append(f"slots={len(slots)!r} (expected {cfg.expected_parallel_slots} inference slot)")
    elif slots[0].get("speculative") is not True:
        violations.append(f"slot.speculative={slots[0].get('speculative')!r} (MTP must be active)")
    if violations:
        raise RuntimeError("llama.cpp runtime does not match the fixed Stage 1 profile: " + "; ".join(violations))

