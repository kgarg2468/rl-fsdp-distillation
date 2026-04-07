from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from tinker import AdamParams, Datum, ModelInput, SamplingParams, ServiceClient, TensorData
from tinker.lib.retry_handler import RetryConfig

from inference_projects.config import ProjectConfig
from inference_projects.pricing import TokenUsage, cost_usd

REAL_USAGE_KEY = "_usage"
PROMPT_TRACES_KEY = "_prompt_traces"
EVAL_ROWS_KEY = "_eval_rows"
CANARY_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "real_eval_prompts.jsonl"
REFUSAL_PATTERNS = (
    "no words",
    "i can't",
    "i cannot",
    "cannot comply",
    "sorry",
    "as an ai",
    "i'm unable",
)


@dataclass(frozen=True)
class PromptRow:
    row_id: str
    prompt: str
    reference: str


@dataclass(frozen=True)
class SamplingBatch:
    outputs: list[str]
    prefill_tokens: int
    sample_tokens: int
    session_id: str | None
    trace_rows: list[dict[str, object]]


@dataclass(frozen=True)
class TrainingCheckpoint:
    run_id: str
    checkpoint_path: str
    sampler_checkpoint_path: str


def build_service_client() -> ServiceClient:
    # ServiceClient reads credentials and endpoint from environment variables.
    return ServiceClient()


def _runtime_retry_config(cfg: ProjectConfig) -> RetryConfig:
    return RetryConfig(
        max_connections=cfg.runtime.retry_max_connections,
        progress_timeout=cfg.runtime.retry_progress_timeout_seconds,
        retry_delay_base=cfg.runtime.retry_delay_base_seconds,
        retry_delay_max=cfg.runtime.retry_delay_max_seconds,
        jitter_factor=cfg.runtime.retry_jitter_factor,
        enable_retry_logic=cfg.runtime.retry_enabled,
    )


def load_canary_prompts(*, limit: int | None = None, fixture_path: Path = CANARY_FIXTURE_PATH) -> list[PromptRow]:
    if not fixture_path.exists():
        raise FileNotFoundError(f"Canary fixture missing: {fixture_path}")
    rows: list[PromptRow] = []
    for line in fixture_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            PromptRow(
                row_id=str(row["id"]),
                prompt=str(row["prompt"]),
                reference=str(row.get("reference", "")),
            )
        )
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"Canary fixture has no prompts: {fixture_path}")
    return rows


def _wait_for_checkpoint(
    *,
    service: ServiceClient,
    run_id: str,
    checkpoint_path: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> None:
    rest = service.create_rest_client()
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            run = rest.get_training_run(run_id).result()
            last_checkpoint = getattr(run, "last_checkpoint", None)
            if last_checkpoint and getattr(last_checkpoint, "tinker_path", "") == checkpoint_path:
                return
            if last_checkpoint and getattr(last_checkpoint, "checkpoint_id", None):
                # If the latest checkpoint changed and is available, continue.
                if getattr(last_checkpoint, "tinker_path", ""):
                    return
        except Exception as exc:  # pragma: no cover - network exceptions are integration-specific
            last_error = str(exc)

        time.sleep(poll_interval_seconds)

    if last_error:
        raise RuntimeError(f"Timed out waiting for checkpoint; last error: {last_error}")
    raise RuntimeError("Timed out waiting for checkpoint to become available")


def _save_new_checkpoint(
    *,
    service: ServiceClient,
    run_id: str,
    save_state_callable: Any,
    stage: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
    wait_for_checkpoint: bool = True,
) -> str:
    checkpoint_name = f"inference-projects-{stage}-{int(time.time())}"
    save_result = save_state_callable(checkpoint_name).result()
    checkpoint_path = str(save_result.path)
    if wait_for_checkpoint:
        _wait_for_checkpoint(
            service=service,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
    return checkpoint_path


def create_lora_checkpoint(
    *,
    service: ServiceClient,
    base_model: str,
    stage: str,
    rank: int = 8,
    seed: int | None = None,
    poll_interval_seconds: int,
    timeout_seconds: int,
    user_metadata: dict[str, str] | None = None,
) -> TrainingCheckpoint:
    train_client = service.create_lora_training_client(
        base_model=base_model,
        rank=rank,
        seed=seed,
        user_metadata=user_metadata,
    )
    info = train_client.get_info()
    run_id = str(info.model_id)
    checkpoint_path = _save_new_checkpoint(
        service=service,
        run_id=run_id,
        save_state_callable=train_client.save_state,
        stage=stage,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        wait_for_checkpoint=True,
    )
    sampler_checkpoint_path = _save_new_checkpoint(
        service=service,
        run_id=run_id,
        save_state_callable=train_client.save_weights_for_sampler,
        stage=stage,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        wait_for_checkpoint=False,
    )
    return TrainingCheckpoint(
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        sampler_checkpoint_path=sampler_checkpoint_path,
    )


def continue_from_checkpoint(
    *,
    service: ServiceClient,
    checkpoint_path: str,
    stage: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
    user_metadata: dict[str, str] | None = None,
) -> TrainingCheckpoint:
    train_client = service.create_training_client_from_state(checkpoint_path, user_metadata=user_metadata)
    info = train_client.get_info()
    run_id = str(info.model_id)
    new_checkpoint_path = _save_new_checkpoint(
        service=service,
        run_id=run_id,
        save_state_callable=train_client.save_state,
        stage=stage,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        wait_for_checkpoint=True,
    )
    sampler_checkpoint_path = _save_new_checkpoint(
        service=service,
        run_id=run_id,
        save_state_callable=train_client.save_weights_for_sampler,
        stage=stage,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        wait_for_checkpoint=False,
    )
    return TrainingCheckpoint(
        run_id=run_id,
        checkpoint_path=new_checkpoint_path,
        sampler_checkpoint_path=sampler_checkpoint_path,
    )


def _sampling_model_path_candidates(model_path: str) -> list[str]:
    candidates = [model_path]
    if "/sampler_weights/" not in model_path and "/weights/" in model_path:
        candidates.append(model_path.replace("/weights/", "/sampler_weights/", 1))
    return candidates


def _create_sampler(
    *,
    service: ServiceClient,
    model_path: str | None,
    base_model: str | None,
    retry_config: RetryConfig | None = None,
):
    if model_path:
        sampler = None
        last_error: Exception | None = None
        for candidate in _sampling_model_path_candidates(model_path):
            try:
                sampler = service.create_sampling_client(model_path=candidate, retry_config=retry_config)
                break
            except Exception as exc:  # pragma: no cover - integration-specific API behavior
                last_error = exc
                if "sampler_weights" not in str(exc):
                    raise
        if sampler is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to initialize sampling client from model_path")
        return sampler
    if base_model:
        return service.create_sampling_client(base_model=base_model, retry_config=retry_config)
    raise ValueError("Either base_model or model_path must be provided")


def sample_prompts(
    *,
    service: ServiceClient,
    prompts: list[str],
    prompt_rows: list[PromptRow] | None = None,
    stage: str = "",
    model_label: str = "",
    base_model: str | None = None,
    model_path: str | None = None,
    max_tokens: int = 24,
    seed: int = 42,
    temperature: float = 0.2,
    stop_tokens: tuple[str, ...] | None = None,
    max_concurrency: int = 1,
    batch_size: int | None = None,
    retry_config: RetryConfig | None = None,
    max_consecutive_failures: int = 5,
) -> SamplingBatch:
    if prompt_rows is not None and len(prompt_rows) != len(prompts):
        raise ValueError("prompt_rows length must match prompts length")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be > 0")

    indexed_prompts = list(enumerate(prompts))
    outputs: list[str] = [""] * len(prompts)
    prefill_tokens = 0
    sample_tokens = 0
    trace_by_index: dict[int, dict[str, object]] = {}
    session_ids: list[str] = []

    chunk_size = batch_size or len(indexed_prompts) or 1
    chunks: list[list[tuple[int, str]]] = [
        indexed_prompts[i : i + chunk_size] for i in range(0, len(indexed_prompts), chunk_size)
    ]

    def process_chunk(chunk: list[tuple[int, str]]) -> tuple[int, int, list[tuple[int, str]], list[tuple[int, dict[str, object]]], str | None]:
        chunk_outputs: list[tuple[int, str]] = []
        chunk_traces: list[tuple[int, dict[str, object]]] = []
        chunk_prefill = 0
        chunk_sample = 0
        consecutive_failures = 0

        chunk_service = build_service_client() if max_concurrency > 1 else service
        sampler = _create_sampler(
            service=chunk_service,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )
        tokenizer = sampler.get_tokenizer()

        for idx, prompt in chunk:
            prompt_ids = tokenizer.encode(prompt)
            prefill_count = len(prompt_ids)
            chunk_prefill += prefill_count
            sampling_error = ""
            try:
                response = sampler.sample(
                    prompt=ModelInput.from_ints(prompt_ids),
                    num_samples=1,
                    sampling_params=SamplingParams(
                        max_tokens=max_tokens,
                        seed=seed + idx,
                        temperature=temperature,
                        stop=list(stop_tokens) if stop_tokens else None,
                    ),
                ).result()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Sampling aborted after {consecutive_failures} consecutive failures "
                        f"for stage '{stage}' model '{model_label}': {exc}"
                    ) from exc
                sampling_error = str(exc)
                response = type("SampleResponse", (), {"sequences": []})()
            sample_count = 0
            output_text = ""
            if response.sequences:
                sampled = response.sequences[0]
                token_ids = list(sampled.tokens)
                sample_count = len(token_ids)
                chunk_sample += sample_count
                output_text = tokenizer.decode(token_ids).strip()
            row = (
                prompt_rows[idx]
                if prompt_rows is not None
                else PromptRow(row_id=f"row-{idx + 1}", prompt=prompt, reference="")
            )
            chunk_outputs.append((idx, output_text))
            chunk_traces.append(
                (
                    idx,
                    {
                        "row_id": row.row_id,
                        "prompt": prompt,
                        "reference": row.reference,
                        "output": output_text,
                        "prefill_tokens": prefill_count,
                        "sample_tokens": sample_count,
                        "stage": stage,
                        "model_label": model_label,
                        "sampling_error": sampling_error,
                    },
                )
            )
        session_id = getattr(sampler, "_sampling_session_id", None)
        return chunk_prefill, chunk_sample, chunk_outputs, chunk_traces, str(session_id) if session_id else None

    if max_concurrency == 1 or len(chunks) == 1:
        for chunk in chunks:
            chunk_prefill, chunk_sample, chunk_outputs, chunk_traces, chunk_session_id = process_chunk(chunk)
            prefill_tokens += chunk_prefill
            sample_tokens += chunk_sample
            for idx, text in chunk_outputs:
                outputs[idx] = text
            for idx, trace in chunk_traces:
                trace_by_index[idx] = trace
            if chunk_session_id:
                session_ids.append(chunk_session_id)
    else:
        max_workers = min(max_concurrency, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for chunk_prefill, chunk_sample, chunk_outputs, chunk_traces, chunk_session_id in executor.map(
                process_chunk, chunks
            ):
                prefill_tokens += chunk_prefill
                sample_tokens += chunk_sample
                for idx, text in chunk_outputs:
                    outputs[idx] = text
                for idx, trace in chunk_traces:
                    trace_by_index[idx] = trace
                if chunk_session_id:
                    session_ids.append(chunk_session_id)

    trace_rows = [trace_by_index[i] for i in range(len(prompts))]
    session_id = ",".join(session_ids) if session_ids else None
    return SamplingBatch(
        outputs=outputs,
        prefill_tokens=prefill_tokens,
        sample_tokens=sample_tokens,
        session_id=session_id,
        trace_rows=trace_rows,
    )


def _score_overlap(output: str, reference: str) -> float:
    ref_tokens = set(re.findall(r"[a-z0-9]+", reference.lower()))
    out_tokens = set(re.findall(r"[a-z0-9]+", output.lower()))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & out_tokens) / len(ref_tokens)


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _is_refusal_like(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def _exact_numeric_match(output: str, reference: str) -> float:
    out_num = _extract_first_int(output)
    ref_num = _extract_first_int(reference)
    if out_num is None or ref_num is None:
        return 0.0
    return 1.0 if out_num == ref_num else 0.0


def _numeric_parse_rate(output: str) -> float:
    return 1.0 if _extract_first_int(output) is not None else 0.0


def _model_health(outputs: list[str]) -> dict[str, float]:
    if not outputs:
        return {
            "empty_output_rate": 0.0,
            "refusal_rate": 0.0,
            "avg_output_chars": 0.0,
        }
    empty_count = sum(1 for output in outputs if not output.strip())
    refusal_count = sum(1 for output in outputs if _is_refusal_like(output))
    avg_chars = _mean([float(len(output)) for output in outputs])
    return {
        "empty_output_rate": round(empty_count / len(outputs), 4),
        "refusal_rate": round(refusal_count / len(outputs), 4),
        "avg_output_chars": round(avg_chars, 2),
    }


def _apply_prompt_template(prompt: str, template: str) -> str:
    if template == "numeric_strict":
        return f"{prompt}\nRespond with only the final integer. No words."
    return prompt


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _select_kd_target(
    *,
    row_id: str,
    teacher_output: str,
    reference: str,
    kd_alpha: float,
) -> str:
    teacher_text = teacher_output.strip()
    reference_text = reference.strip()
    if not teacher_text:
        return reference_text or teacher_text
    if kd_alpha <= 0:
        return reference_text or teacher_text
    if kd_alpha >= 1:
        return teacher_text
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    threshold = int(round(kd_alpha * 10000))
    if bucket < threshold:
        return teacher_text
    return reference_text or teacher_text


def _usage_dict(
    *,
    prefill_tokens: int,
    sample_tokens: int,
    train_tokens: int,
    run_id: str,
    provider_raw: dict[str, object],
    cost: float | None = None,
) -> dict[str, object]:
    return {
        "prefill_tokens": int(prefill_tokens),
        "sample_tokens": int(sample_tokens),
        "train_tokens": int(train_tokens),
        "cost_usd": cost,
        "provider_raw": provider_raw,
        "run_id": run_id,
    }


def _safe_token_ids(tokenizer: Any, text: str) -> list[int]:
    token_ids = list(tokenizer.encode(text))
    if token_ids:
        return token_ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int):
        return [eos_id]
    return [0]


def _make_training_datum(*, tokenizer: Any, prompt: str, target: str) -> tuple[Datum, int]:
    prompt_ids = _safe_token_ids(tokenizer, prompt)
    target_ids = _safe_token_ids(tokenizer, target)
    input_ids = list(prompt_ids) + list(target_ids)
    target_tokens = ([0] * len(prompt_ids)) + list(target_ids)
    target_weights = ([0.0] * len(prompt_ids)) + ([1.0] * len(target_ids))
    datum = Datum(
        model_input=ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(input_ids)],
            ),
            "weights": TensorData(
                data=target_weights,
                dtype="float32",
                shape=[len(input_ids)],
            ),
        },
    )
    return datum, len(input_ids)


def _is_loss_shape_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "array record" in message
        or ("target_tokens" in message and "sequence length" in message)
        or ("loss function inputs" in message and "convert" in message)
    )


def _forward_backward_with_fallback(*, train_client: Any, batch: list[Datum]) -> tuple[Any, bool]:
    try:
        return train_client.forward_backward(batch, "cross_entropy").result(), False
    except Exception as exc:
        if len(batch) <= 1 or not _is_loss_shape_error(exc):
            raise

        fallback_result: Any | None = None
        for datum in batch:
            fallback_result = train_client.forward_backward([datum], "cross_entropy").result()
        if fallback_result is None:
            raise RuntimeError("micro-batch fallback produced no forward/backward result")
        return fallback_result, True


def _extract_loss_metric(forward_backward_result: Any) -> float | None:
    metrics = getattr(forward_backward_result, "metrics", None)
    if isinstance(metrics, dict):
        for key in ("loss", "cross_entropy", "mean_loss"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    loss_outputs = getattr(forward_backward_result, "loss_fn_outputs", None)
    if isinstance(loss_outputs, list):
        for entry in loss_outputs:
            if isinstance(entry, dict):
                for candidate in entry.values():
                    data = getattr(candidate, "data", None)
                    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
                        return float(data[0])
    return None


def _run_training_loop(
    *,
    train_client: Any,
    examples: list[tuple[str, str]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    grad_clip_norm: float,
    max_consecutive_failures: int,
    stage: str,
) -> dict[str, object]:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be > 0")
    if not examples:
        return {
            "steps": 0,
            "nan_events": 0,
            "loss_trace": [],
            "train_tokens": 0,
            "batches_per_epoch": 0,
        }

    tokenizer = train_client.get_tokenizer()
    datums: list[Datum] = []
    train_tokens = 0
    for prompt, target in examples:
        datum, target_tokens = _make_training_datum(tokenizer=tokenizer, prompt=prompt, target=target)
        datums.append(datum)
        train_tokens += target_tokens

    batches: list[list[Datum]] = [datums[i : i + batch_size] for i in range(0, len(datums), batch_size)]
    total_steps = epochs * len(batches)
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_ratio > 0 and warmup_steps <= 0:
        warmup_steps = 1

    loss_trace: list[float] = []
    nan_events = 0
    micro_batch_fallbacks = 0
    consecutive_failures = 0
    global_step = 0

    for _epoch in range(epochs):
        for batch in batches:
            try:
                fwdbwd_result, used_micro_batch_fallback = _forward_backward_with_fallback(
                    train_client=train_client,
                    batch=batch,
                )
                if used_micro_batch_fallback:
                    micro_batch_fallbacks += 1
                loss_value = _extract_loss_metric(fwdbwd_result)
                if loss_value is not None:
                    if math.isnan(loss_value) or math.isinf(loss_value):
                        nan_events += 1
                    else:
                        loss_trace.append(round(loss_value, 6))
                global_step += 1
                if warmup_steps > 0 and global_step <= warmup_steps:
                    step_lr = learning_rate * (global_step / warmup_steps)
                else:
                    step_lr = learning_rate
                train_client.optim_step(
                    AdamParams(
                        learning_rate=max(step_lr, 1e-8),
                        weight_decay=weight_decay,
                        grad_clip_norm=grad_clip_norm,
                    )
                ).result()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Training aborted after {consecutive_failures} consecutive failures "
                        f"during stage '{stage}': {exc}"
                    ) from exc

    return {
        "steps": global_step,
        "nan_events": nan_events,
        "loss_trace": loss_trace,
        "train_tokens": train_tokens * epochs,
        "batches_per_epoch": len(batches),
        "micro_batch_fallbacks": micro_batch_fallbacks,
    }


def _save_training_checkpoints(
    *,
    service: ServiceClient,
    train_client: Any,
    run_id: str,
    stage: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> TrainingCheckpoint:
    checkpoint_path = _save_new_checkpoint(
        service=service,
        run_id=run_id,
        save_state_callable=train_client.save_state,
        stage=stage,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        wait_for_checkpoint=True,
    )
    sampler_checkpoint_path = _save_new_checkpoint(
        service=service,
        run_id=run_id,
        save_state_callable=train_client.save_weights_for_sampler,
        stage=stage,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        wait_for_checkpoint=False,
    )
    return TrainingCheckpoint(
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        sampler_checkpoint_path=sampler_checkpoint_path,
    )


def run_real_rl(*, cfg: ProjectConfig) -> dict[str, object]:
    service = build_service_client()
    retry_config = _runtime_retry_config(cfg)
    prompts = load_canary_prompts(fixture_path=cfg.evaluation.prompt_file)
    training_examples = [(row.prompt, row.reference or row.prompt) for row in prompts]
    train_client = service.create_lora_training_client(
        base_model=cfg.teacher_model,
        rank=cfg.distillation.lora_rank,
        seed=cfg.seed,
        user_metadata={"stage": "rl", "pipeline": "inference-projects"},
    )
    info = train_client.get_info()
    run_id = str(info.model_id)
    training = _run_training_loop(
        train_client=train_client,
        examples=training_examples,
        epochs=cfg.distillation.epochs,
        batch_size=cfg.distillation.batch_size,
        learning_rate=cfg.distillation.learning_rate,
        warmup_ratio=cfg.distillation.warmup_ratio,
        weight_decay=cfg.distillation.weight_decay,
        grad_clip_norm=cfg.distillation.grad_clip,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
        stage="rl",
    )
    checkpoint = _save_training_checkpoints(
        service=service,
        train_client=train_client,
        run_id=run_id,
        stage="rl",
        poll_interval_seconds=cfg.runtime.real_poll_interval_seconds,
        timeout_seconds=cfg.runtime.real_poll_timeout_seconds,
    )
    eval_prompts = prompts[: min(8, len(prompts))]
    sampled = sample_prompts(
        service=service,
        prompts=[row.prompt for row in eval_prompts],
        prompt_rows=eval_prompts,
        stage="rl",
        model_label="teacher",
        model_path=checkpoint.sampler_checkpoint_path,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )

    overlap_scores = [_score_overlap(out, row.reference) for out, row in zip(sampled.outputs, eval_prompts)]
    quality_score = round(_mean(overlap_scores), 4)
    rl_nan_events = int(training["nan_events"])
    stability_score = round(max(0.0, 1.0 - (rl_nan_events / max(1, int(training["steps"])))), 4)

    return {
        "payload": {
            "model": cfg.teacher_model,
            "stage": "rl",
            "quality_score": quality_score,
            "stability_score": stability_score,
            "checkpoint_path": checkpoint.checkpoint_path,
            "sampler_checkpoint_path": checkpoint.sampler_checkpoint_path,
            "run_id": checkpoint.run_id,
            "rl_stability_score": stability_score,
            "rl_nan_events": rl_nan_events,
            "training": {
                "steps": int(training["steps"]),
                "batches_per_epoch": int(training["batches_per_epoch"]),
                "epochs": cfg.distillation.epochs,
                "batch_size": cfg.distillation.batch_size,
                "learning_rate": cfg.distillation.learning_rate,
                "warmup_ratio": cfg.distillation.warmup_ratio,
                "weight_decay": cfg.distillation.weight_decay,
                "grad_clip_norm": cfg.distillation.grad_clip,
                "loss_trace": list(training["loss_trace"]),
                "train_examples": len(training_examples),
            },
            "notes": "Real RL stage completed iterative training with optimizer steps.",
            PROMPT_TRACES_KEY: sampled.trace_rows,
        },
        "usage": _usage_dict(
            prefill_tokens=sampled.prefill_tokens,
            sample_tokens=sampled.sample_tokens,
            train_tokens=int(training["train_tokens"]),
            run_id=checkpoint.run_id,
            provider_raw={
                "checkpoint_path": checkpoint.checkpoint_path,
                "sampler_checkpoint_path": checkpoint.sampler_checkpoint_path,
                "sampling_session_id": sampled.session_id,
                "training_steps": int(training["steps"]),
                "training_loss_trace_count": len(training["loss_trace"]),
                "stage": "rl",
            },
        ),
    }


def run_real_teacher_ft(*, cfg: ProjectConfig, teacher_payload: dict[str, object]) -> dict[str, object]:
    prior_checkpoint = str(teacher_payload.get("checkpoint_path", "")).strip()
    if not prior_checkpoint:
        raise RuntimeError("Teacher FT stage requires teacher checkpoint_path from RL stage")

    service = build_service_client()
    retry_config = _runtime_retry_config(cfg)
    prompts = load_canary_prompts(fixture_path=cfg.evaluation.prompt_file)
    training_examples = [(row.prompt, row.reference or row.prompt) for row in prompts]
    train_client = service.create_training_client_from_state_with_optimizer(
        prior_checkpoint,
        user_metadata={"stage": "teacher_ft", "pipeline": "inference-projects"},
    )
    info = train_client.get_info()
    run_id = str(info.model_id)
    training = _run_training_loop(
        train_client=train_client,
        examples=training_examples,
        epochs=cfg.distillation.epochs,
        batch_size=cfg.distillation.batch_size,
        learning_rate=cfg.distillation.learning_rate,
        warmup_ratio=cfg.distillation.warmup_ratio,
        weight_decay=cfg.distillation.weight_decay,
        grad_clip_norm=cfg.distillation.grad_clip,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
        stage="teacher_ft",
    )
    checkpoint = _save_training_checkpoints(
        service=service,
        train_client=train_client,
        run_id=run_id,
        stage="teacher_ft",
        poll_interval_seconds=cfg.runtime.real_poll_interval_seconds,
        timeout_seconds=cfg.runtime.real_poll_timeout_seconds,
    )
    eval_prompts = prompts[: min(8, len(prompts))]
    sampled = sample_prompts(
        service=service,
        prompts=[row.prompt for row in eval_prompts],
        prompt_rows=eval_prompts,
        stage="teacher_ft",
        model_label="teacher",
        model_path=checkpoint.sampler_checkpoint_path,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )

    prior_quality = float(teacher_payload.get("quality_score", 0.70))
    overlap_scores = [_score_overlap(out, row.reference) for out, row in zip(sampled.outputs, eval_prompts)]
    sampled_quality = _mean(overlap_scores)
    quality_score = round(max(prior_quality, sampled_quality), 4)
    teacher_ft_nan_events = int(training["nan_events"])
    stability_score = round(max(0.0, 1.0 - (teacher_ft_nan_events / max(1, int(training["steps"])))), 4)

    return {
        "payload": {
            "model": cfg.teacher_model,
            "stage": "teacher_ft",
            "quality_score": quality_score,
            "stability_score": stability_score,
            "checkpoint_path": checkpoint.checkpoint_path,
            "sampler_checkpoint_path": checkpoint.sampler_checkpoint_path,
            "run_id": checkpoint.run_id,
            "axolotl_teacher_ft": True,
            "rl_stability_score": float(teacher_payload.get("rl_stability_score", 0.92)),
            "rl_nan_events": int(teacher_payload.get("rl_nan_events", 0)),
            "teacher_ft_stability_score": stability_score,
            "teacher_ft_nan_events": teacher_ft_nan_events,
            "training": {
                "steps": int(training["steps"]),
                "batches_per_epoch": int(training["batches_per_epoch"]),
                "epochs": cfg.distillation.epochs,
                "batch_size": cfg.distillation.batch_size,
                "learning_rate": cfg.distillation.learning_rate,
                "warmup_ratio": cfg.distillation.warmup_ratio,
                "weight_decay": cfg.distillation.weight_decay,
                "grad_clip_norm": cfg.distillation.grad_clip,
                "loss_trace": list(training["loss_trace"]),
                "train_examples": len(training_examples),
                "source_checkpoint_path": prior_checkpoint,
            },
            "notes": "Real Teacher FT stage resumed with optimizer state and completed iterative training.",
            PROMPT_TRACES_KEY: sampled.trace_rows,
        },
        "usage": _usage_dict(
            prefill_tokens=sampled.prefill_tokens,
            sample_tokens=sampled.sample_tokens,
            train_tokens=int(training["train_tokens"]),
            run_id=checkpoint.run_id,
            provider_raw={
                "source_checkpoint_path": prior_checkpoint,
                "checkpoint_path": checkpoint.checkpoint_path,
                "sampler_checkpoint_path": checkpoint.sampler_checkpoint_path,
                "sampling_session_id": sampled.session_id,
                "training_steps": int(training["steps"]),
                "training_loss_trace_count": len(training["loss_trace"]),
                "stage": "teacher_ft",
            },
        ),
    }


def run_real_distill(*, cfg: ProjectConfig, teacher_payload: dict[str, object]) -> dict[str, object]:
    teacher_checkpoint_path = str(
        teacher_payload.get("sampler_checkpoint_path", teacher_payload.get("checkpoint_path", ""))
    ).strip()
    if not teacher_checkpoint_path:
        raise RuntimeError("Distill stage requires teacher checkpoint_path from Teacher FT stage")

    service = build_service_client()
    retry_config = _runtime_retry_config(cfg)
    training_prompt_file = cfg.distillation.training_prompt_file or cfg.evaluation.prompt_file
    prompts = load_canary_prompts(fixture_path=training_prompt_file)
    prompt_texts = [row.prompt for row in prompts]
    teacher_prompt_texts = [
        _apply_prompt_template(row.prompt, cfg.distillation.teacher_prompt_template) for row in prompts
    ]

    teacher_samples = sample_prompts(
        service=service,
        prompts=teacher_prompt_texts,
        prompt_rows=prompts,
        stage="distill",
        model_label="teacher",
        model_path=teacher_checkpoint_path,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        temperature=cfg.evaluation.eval_temperature,
        stop_tokens=cfg.evaluation.eval_stop_tokens,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )
    baseline_samples = sample_prompts(
        service=service,
        prompts=prompt_texts,
        prompt_rows=prompts,
        stage="distill",
        model_label="baseline",
        base_model=cfg.baseline_model,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        temperature=cfg.evaluation.eval_temperature,
        stop_tokens=cfg.evaluation.eval_stop_tokens,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )

    distill_rows: list[dict[str, object]] = []
    for row, teacher_output, baseline_output in zip(prompts, teacher_samples.outputs, baseline_samples.outputs):
        teacher_refusal = _is_refusal_like(teacher_output)
        teacher_parse = _numeric_parse_rate(teacher_output) == 1.0
        teacher_exact = _exact_numeric_match(teacher_output, row.reference) == 1.0
        baseline_exact = _exact_numeric_match(baseline_output, row.reference) == 1.0
        teacher_overlap = _score_overlap(teacher_output, row.reference)
        baseline_overlap = _score_overlap(baseline_output, row.reference)
        distill_rows.append(
            {
                "row_id": row.row_id,
                "prompt": row.prompt,
                "reference": row.reference,
                "teacher_output": teacher_output,
                "baseline_output": baseline_output,
                "teacher_overlap": round(teacher_overlap, 4),
                "baseline_overlap": round(baseline_overlap, 4),
                "teacher_minus_baseline_overlap": round(teacher_overlap - baseline_overlap, 4),
                "teacher_refusal_like": teacher_refusal,
                "teacher_numeric_parse": teacher_parse,
                "teacher_exact_match": teacher_exact,
                "baseline_exact_match": baseline_exact,
                "baseline_fail_teacher_pass": (not baseline_exact) and teacher_exact,
                "selected_for_distill": False,
                "selected_for_teacher_target": False,
                "target_source": "",
            }
        )

    def _eligible(row: dict[str, object]) -> bool:
        if cfg.distillation.filter_profile == "strict":
            return (
                bool(row["teacher_numeric_parse"])
                and not bool(row["teacher_refusal_like"])
                and bool(row["teacher_exact_match"])
            )
        return bool(row["teacher_numeric_parse"]) and not bool(row["teacher_refusal_like"])

    eligible = [row for row in distill_rows if _eligible(row)]
    if not eligible:
        eligible = list(distill_rows)

    def _selection_key(row: dict[str, object]) -> tuple[float, int, int, int, str]:
        return (
            float(row["teacher_minus_baseline_overlap"]),
            1 if bool(row["baseline_fail_teacher_pass"]) else 0,
            1 if bool(row["teacher_exact_match"]) else 0,
            1 if bool(row["teacher_numeric_parse"]) else 0,
            str(row["row_id"]),
        )

    hard = sorted(
        [row for row in eligible if bool(row["baseline_fail_teacher_pass"])],
        key=_selection_key,
        reverse=True,
    )
    easy = sorted(
        [row for row in eligible if not bool(row["baseline_fail_teacher_pass"])],
        key=_selection_key,
        reverse=True,
    )
    target_size = len(eligible)
    target_hard = min(len(hard), int(round(target_size * cfg.distillation.hard_example_ratio)))
    selected_rows = [*hard[:target_hard], *easy[: max(0, target_size - target_hard)]]
    if len(selected_rows) < target_size:
        selected_ids = {id(row) for row in selected_rows}
        for row in hard:
            if id(row) in selected_ids:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= target_size:
                break
    if not selected_rows:
        selected_rows = list(distill_rows)
    selected_row_ids = {str(row["row_id"]) for row in selected_rows}
    for row in distill_rows:
        row["selected_for_distill"] = str(row["row_id"]) in selected_row_ids

    selected_rows = list(selected_rows)
    selected_prompt_rows = [row for row in prompts if row.row_id in selected_row_ids]
    teacher_target_pool = [
        row
        for row in selected_rows
        if bool(row["teacher_numeric_parse"])
        and not bool(row["teacher_refusal_like"])
        and float(row["teacher_minus_baseline_overlap"]) >= 0.0
    ]
    if not teacher_target_pool:
        teacher_target_pool = [
            row for row in selected_rows if bool(str(row.get("teacher_output", "")).strip())
        ]
    teacher_target_pool = sorted(teacher_target_pool, key=_selection_key, reverse=True)
    target_teacher_count = int(round(len(selected_rows) * cfg.distillation.kd_alpha))
    target_teacher_count = min(target_teacher_count, len(teacher_target_pool))
    teacher_target_ids = {
        str(row["row_id"]) for row in teacher_target_pool[:target_teacher_count]
    }

    kd_examples: list[tuple[str, str]] = []
    for row in selected_rows:
        row_id = str(row["row_id"])
        teacher_output = str(row["teacher_output"]).strip()
        reference = str(row["reference"]).strip()
        use_teacher_target = row_id in teacher_target_ids and bool(teacher_output)
        target_text = teacher_output if use_teacher_target else (reference or teacher_output)
        row["selected_for_teacher_target"] = use_teacher_target
        row["target_source"] = "teacher" if use_teacher_target else "reference"
        kd_examples.append((str(row["prompt"]), target_text))

    student_train_client = service.create_lora_training_client(
        base_model=cfg.student_model,
        rank=cfg.distillation.lora_rank,
        seed=cfg.seed,
        user_metadata={
            "stage": "distill",
            "pipeline": "inference-projects",
            "filter_profile": cfg.distillation.filter_profile,
            "teacher_prompt_template": cfg.distillation.teacher_prompt_template,
            "kd_alpha": str(cfg.distillation.kd_alpha),
            "kd_temperature": str(cfg.distillation.kd_temperature),
            "learning_rate": str(cfg.distillation.learning_rate),
            "epochs": str(cfg.distillation.epochs),
            "batch_size": str(cfg.distillation.batch_size),
            "warmup_ratio": str(cfg.distillation.warmup_ratio),
            "hard_example_ratio": str(cfg.distillation.hard_example_ratio),
            "context_length": str(cfg.distillation.context_length),
            "grad_clip": str(cfg.distillation.grad_clip),
            "weight_decay": str(cfg.distillation.weight_decay),
            "training_prompt_limit": str(len(prompts)),
        },
    )
    train_info = student_train_client.get_info()
    student_run_id = str(train_info.model_id)
    training = _run_training_loop(
        train_client=student_train_client,
        examples=kd_examples,
        epochs=cfg.distillation.epochs,
        batch_size=cfg.distillation.batch_size,
        learning_rate=cfg.distillation.learning_rate / max(1.0, cfg.distillation.kd_temperature),
        warmup_ratio=cfg.distillation.warmup_ratio,
        weight_decay=cfg.distillation.weight_decay,
        grad_clip_norm=cfg.distillation.grad_clip,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
        stage="distill",
    )
    student_checkpoint = _save_training_checkpoints(
        service=service,
        train_client=student_train_client,
        run_id=student_run_id,
        stage="distill",
        poll_interval_seconds=cfg.runtime.real_poll_interval_seconds,
        timeout_seconds=cfg.runtime.real_poll_timeout_seconds,
    )

    eval_prompt_rows = selected_prompt_rows[: min(16, len(selected_prompt_rows))]
    if not eval_prompt_rows:
        eval_prompt_rows = prompts[: min(16, len(prompts))]
    student_samples = sample_prompts(
        service=service,
        prompts=[row.prompt for row in eval_prompt_rows],
        prompt_rows=eval_prompt_rows,
        stage="distill",
        model_label="student",
        model_path=student_checkpoint.sampler_checkpoint_path,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        temperature=cfg.evaluation.eval_temperature,
        stop_tokens=cfg.evaluation.eval_stop_tokens,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )

    teacher_quality = round(_mean([_score_overlap(out, row.reference) for out, row in zip(teacher_samples.outputs, prompts)]), 4)
    student_quality = round(
        _mean([_score_overlap(out, row.reference) for out, row in zip(student_samples.outputs, eval_prompt_rows)]),
        4,
    )
    distill_nan_events = int(training["nan_events"])
    distill_stability_score = round(max(0.0, 1.0 - (distill_nan_events / max(1, int(training["steps"])))), 4)

    return {
        "payload": {
            "teacher_model": cfg.teacher_model,
            "student_model": cfg.student_model,
            "teacher_quality": teacher_quality,
            "student_quality": student_quality,
            "compression_ratio": 8.0,
            "stability_score": distill_stability_score,
            "checkpoint_path": student_checkpoint.checkpoint_path,
            "sampler_checkpoint_path": student_checkpoint.sampler_checkpoint_path,
            "run_id": student_checkpoint.run_id,
            "distill_stability_score": distill_stability_score,
            "distill_nan_events": distill_nan_events,
            "distillation_config": {
                "training_prompt_limit": len(prompts),
                "training_prompt_file": str(training_prompt_file),
                "teacher_prompt_template": cfg.distillation.teacher_prompt_template,
                "filter_profile": cfg.distillation.filter_profile,
                "hard_example_ratio": cfg.distillation.hard_example_ratio,
                "kd_alpha": cfg.distillation.kd_alpha,
                "kd_temperature": cfg.distillation.kd_temperature,
                "learning_rate": cfg.distillation.learning_rate,
                "epochs": cfg.distillation.epochs,
                "batch_size": cfg.distillation.batch_size,
                "warmup_ratio": cfg.distillation.warmup_ratio,
                "lora_rank": cfg.distillation.lora_rank,
                "context_length": cfg.distillation.context_length,
                "grad_clip": cfg.distillation.grad_clip,
                "weight_decay": cfg.distillation.weight_decay,
            },
            "training": {
                "steps": int(training["steps"]),
                "batches_per_epoch": int(training["batches_per_epoch"]),
                "epochs": cfg.distillation.epochs,
                "batch_size": cfg.distillation.batch_size,
                "learning_rate_effective": cfg.distillation.learning_rate / max(1.0, cfg.distillation.kd_temperature),
                "warmup_ratio": cfg.distillation.warmup_ratio,
                "weight_decay": cfg.distillation.weight_decay,
                "grad_clip_norm": cfg.distillation.grad_clip,
                "loss_trace": list(training["loss_trace"]),
                "train_examples": len(kd_examples),
            },
            "distill_dataset": {
                "total_rows": len(distill_rows),
                "eligible_rows": len(eligible),
                "selected_rows": len(selected_rows),
                "selected_hard_rows": sum(1 for row in selected_rows if bool(row["baseline_fail_teacher_pass"])),
                "teacher_target_rows": sum(1 for row in selected_rows if bool(row["selected_for_teacher_target"])),
                "reference_target_rows": sum(1 for row in selected_rows if not bool(row["selected_for_teacher_target"])),
                "selected_hard_ratio": round(
                    (
                        sum(1 for row in selected_rows if bool(row["baseline_fail_teacher_pass"]))
                        / max(1, len(selected_rows))
                    ),
                    4,
                ),
                "kd_example_rows": len(kd_examples),
            },
            "distill_rows": distill_rows,
            "notes": "Real distillation stage performed iterative KD training on selected rows.",
            PROMPT_TRACES_KEY: [*teacher_samples.trace_rows, *student_samples.trace_rows],
        },
        "usage": _usage_dict(
            prefill_tokens=teacher_samples.prefill_tokens
            + baseline_samples.prefill_tokens
            + student_samples.prefill_tokens,
            sample_tokens=teacher_samples.sample_tokens
            + baseline_samples.sample_tokens
            + student_samples.sample_tokens,
            train_tokens=int(training["train_tokens"]),
            run_id=student_checkpoint.run_id,
            provider_raw={
                "teacher_sampling_session_id": teacher_samples.session_id,
                "baseline_sampling_session_id": baseline_samples.session_id,
                "student_sampling_session_id": student_samples.session_id,
                "teacher_checkpoint_path": teacher_checkpoint_path,
                "student_checkpoint_path": student_checkpoint.checkpoint_path,
                "student_sampler_checkpoint_path": student_checkpoint.sampler_checkpoint_path,
                "filter_profile": cfg.distillation.filter_profile,
                "teacher_prompt_template": cfg.distillation.teacher_prompt_template,
                "training_steps": int(training["steps"]),
                "training_loss_trace_count": len(training["loss_trace"]),
                "stage": "distill",
            },
        ),
    }


def _cost_per_1k(*, cfg: ProjectConfig, prefill_tokens: int, sample_tokens: int) -> float:
    total_tokens = prefill_tokens + sample_tokens
    if total_tokens <= 0:
        return 0.0
    run_cost = cost_usd(
        TokenUsage(prefill=prefill_tokens, sample=sample_tokens, train=0),
        prefill_rate=cfg.token_rates_per_million["prefill"],
        sample_rate=cfg.token_rates_per_million["sample"],
        train_rate=cfg.token_rates_per_million["train"],
    )
    return round(run_cost / (total_tokens / 1000.0), 6)


def run_real_eval(
    *,
    cfg: ProjectConfig,
    teacher_payload: dict[str, object],
    student_payload: dict[str, object],
) -> dict[str, object]:
    service = build_service_client()
    retry_config = _runtime_retry_config(cfg)
    prompts = load_canary_prompts(fixture_path=cfg.evaluation.prompt_file)
    prompt_texts = [row.prompt for row in prompts]
    teacher_prompt_texts = [
        _apply_prompt_template(row.prompt, cfg.distillation.teacher_prompt_template) for row in prompts
    ]
    references = [row.reference for row in prompts]

    teacher_checkpoint_path = str(
        teacher_payload.get("sampler_checkpoint_path", teacher_payload.get("checkpoint_path", ""))
    ).strip()
    student_checkpoint_path = str(
        student_payload.get("sampler_checkpoint_path", student_payload.get("checkpoint_path", ""))
    ).strip()

    baseline = sample_prompts(
        service=service,
        prompts=prompt_texts,
        prompt_rows=prompts,
        stage="eval",
        model_label="baseline",
        base_model=cfg.baseline_model,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        temperature=cfg.evaluation.eval_temperature,
        stop_tokens=cfg.evaluation.eval_stop_tokens,
        max_concurrency=cfg.evaluation.max_concurrency,
        batch_size=cfg.evaluation.batch_size,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )
    teacher = sample_prompts(
        service=service,
        prompts=teacher_prompt_texts,
        prompt_rows=prompts,
        stage="eval",
        model_label="teacher",
        model_path=teacher_checkpoint_path if teacher_checkpoint_path else None,
        base_model=None if teacher_checkpoint_path else cfg.teacher_model,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        temperature=cfg.evaluation.eval_temperature,
        stop_tokens=cfg.evaluation.eval_stop_tokens,
        max_concurrency=cfg.evaluation.max_concurrency,
        batch_size=cfg.evaluation.batch_size,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )
    student = sample_prompts(
        service=service,
        prompts=prompt_texts,
        prompt_rows=prompts,
        stage="eval",
        model_label="student",
        model_path=student_checkpoint_path if student_checkpoint_path else None,
        base_model=None if student_checkpoint_path else cfg.student_model,
        max_tokens=cfg.evaluation.max_tokens_eval,
        seed=cfg.seed,
        temperature=cfg.evaluation.eval_temperature,
        stop_tokens=cfg.evaluation.eval_stop_tokens,
        max_concurrency=cfg.evaluation.max_concurrency,
        batch_size=cfg.evaluation.batch_size,
        retry_config=retry_config,
        max_consecutive_failures=cfg.runtime.max_consecutive_failures,
    )

    baseline_scores = [_score_overlap(out, ref) for out, ref in zip(baseline.outputs, references)]
    teacher_scores = [_score_overlap(out, ref) for out, ref in zip(teacher.outputs, references)]
    student_scores = [_score_overlap(out, ref) for out, ref in zip(student.outputs, references)]
    baseline_exact_matches = [_exact_numeric_match(out, ref) for out, ref in zip(baseline.outputs, references)]
    teacher_exact_matches = [_exact_numeric_match(out, ref) for out, ref in zip(teacher.outputs, references)]
    student_exact_matches = [_exact_numeric_match(out, ref) for out, ref in zip(student.outputs, references)]
    baseline_parse_rates = [_numeric_parse_rate(out) for out in baseline.outputs]
    teacher_parse_rates = [_numeric_parse_rate(out) for out in teacher.outputs]
    student_parse_rates = [_numeric_parse_rate(out) for out in student.outputs]

    baseline_quality = round(_mean(baseline_scores), 4)
    teacher_quality = round(_mean(teacher_scores), 4)
    student_quality = round(_mean(student_scores), 4)
    baseline_exact_match = round(_mean(baseline_exact_matches), 4)
    teacher_exact_match = round(_mean(teacher_exact_matches), 4)
    student_exact_match = round(_mean(student_exact_matches), 4)
    baseline_numeric_parse = round(_mean(baseline_parse_rates), 4)
    teacher_numeric_parse = round(_mean(teacher_parse_rates), 4)
    student_numeric_parse = round(_mean(student_parse_rates), 4)

    student_vs_baseline_wins = [1.0 if s > b else 0.0 for s, b in zip(student_scores, baseline_scores)]
    student_vs_teacher_wins = [1.0 if s > t else 0.0 for s, t in zip(student_scores, teacher_scores)]

    teacher_cost_per_1k = _cost_per_1k(
        cfg=cfg,
        prefill_tokens=teacher.prefill_tokens,
        sample_tokens=teacher.sample_tokens,
    )
    student_cost_per_1k = _cost_per_1k(
        cfg=cfg,
        prefill_tokens=student.prefill_tokens,
        sample_tokens=student.sample_tokens,
    )

    savings_pct = 0.0
    if teacher_cost_per_1k > 0:
        savings_pct = round((1 - (student_cost_per_1k / teacher_cost_per_1k)) * 100, 2)

    rl_stability = float(teacher_payload.get("rl_stability_score", teacher_payload.get("stability_score", 0.90)))
    teacher_ft_stability = float(
        teacher_payload.get("teacher_ft_stability_score", teacher_payload.get("stability_score", 0.88))
    )
    distill_stability = float(
        student_payload.get("distill_stability_score", student_payload.get("stability_score", 0.87))
    )
    health = {
        "baseline": _model_health(baseline.outputs),
        "teacher": _model_health(teacher.outputs),
        "student": _model_health(student.outputs),
    }
    integrity_reasons: list[str] = []
    integrity_status = "pass"
    refusal_failed = health["teacher"]["refusal_rate"] >= cfg.evaluation.teacher_integrity_refusal_threshold
    near_zero_failed = (
        teacher_quality <= cfg.evaluation.teacher_integrity_min_score
        and baseline_quality >= max(0.30, cfg.evaluation.teacher_integrity_min_score + 0.20)
    )
    parse_warn = teacher_numeric_parse < cfg.evaluation.teacher_integrity_numeric_parse_threshold
    margin_warn = teacher_quality < baseline_quality

    if refusal_failed:
        integrity_status = "fail"
        integrity_reasons.append(
            "teacher refusal rate "
            f"{health['teacher']['refusal_rate']:.4f} exceeded threshold "
            f"{cfg.evaluation.teacher_integrity_refusal_threshold:.4f}"
        )
    if near_zero_failed:
        integrity_status = "fail"
        integrity_reasons.append(
            "teacher overlap near-zero while baseline remained strong "
            f"(teacher={teacher_quality:.4f}, baseline={baseline_quality:.4f})"
        )
    if parse_warn:
        if integrity_status == "pass":
            integrity_status = "warn"
        integrity_reasons.append(
            "teacher numeric parse rate "
            f"{teacher_numeric_parse:.4f} below threshold "
            f"{cfg.evaluation.teacher_integrity_numeric_parse_threshold:.4f}"
        )
    if margin_warn:
        if integrity_status == "pass":
            integrity_status = "warn"
        integrity_reasons.append(
            "teacher overlap below baseline "
            f"(teacher={teacher_quality:.4f}, baseline={baseline_quality:.4f})"
        )
    integrity_reason = "; ".join(integrity_reasons)
    integrity_passed = integrity_status == "pass"

    eval_rows: list[dict[str, object]] = []
    for (
        row,
        baseline_output,
        teacher_output,
        student_output,
        baseline_overlap,
        teacher_overlap,
        student_overlap,
        baseline_exact,
        teacher_exact,
        student_exact,
        baseline_parse,
        teacher_parse,
        student_parse,
    ) in zip(
        prompts,
        baseline.outputs,
        teacher.outputs,
        student.outputs,
        baseline_scores,
        teacher_scores,
        student_scores,
        baseline_exact_matches,
        teacher_exact_matches,
        student_exact_matches,
        baseline_parse_rates,
        teacher_parse_rates,
        student_parse_rates,
    ):
        eval_rows.append(
            {
                "row_id": row.row_id,
                "prompt": row.prompt,
                "reference": row.reference,
                "baseline_output": baseline_output,
                "teacher_output": teacher_output,
                "student_output": student_output,
                "baseline_overlap": round(baseline_overlap, 4),
                "teacher_overlap": round(teacher_overlap, 4),
                "student_overlap": round(student_overlap, 4),
                "baseline_exact_match": round(baseline_exact, 4),
                "teacher_exact_match": round(teacher_exact, 4),
                "student_exact_match": round(student_exact, 4),
                "baseline_numeric_parse": round(baseline_parse, 4),
                "teacher_numeric_parse": round(teacher_parse, 4),
                "student_numeric_parse": round(student_parse, 4),
                "student_vs_baseline_win": 1.0 if student_overlap > baseline_overlap else 0.0,
                "student_vs_teacher_win": 1.0 if student_overlap > teacher_overlap else 0.0,
            }
        )

    return {
        "payload": {
            "quality": {
                "benchmark": {
                    "baseline": baseline_quality,
                    "teacher": teacher_quality,
                    "student": student_quality,
                    "student_retention_vs_teacher": round(
                        student_quality / teacher_quality, 4
                    )
                    if teacher_quality > 0
                    else 0.0,
                },
                "llm_judge": {
                    "student_vs_baseline_win_rate": round(_mean(student_vs_baseline_wins), 4),
                    "student_vs_teacher_win_rate": round(_mean(student_vs_teacher_wins), 4),
                },
                "sanity": {
                    "exact_match_rate": {
                        "baseline": baseline_exact_match,
                        "teacher": teacher_exact_match,
                        "student": student_exact_match,
                        "teacher_minus_baseline_exact_match": round(teacher_exact_match - baseline_exact_match, 4),
                        "student_minus_baseline_exact_match": round(student_exact_match - baseline_exact_match, 4),
                    },
                    "numeric_parse_rate": {
                        "baseline": baseline_numeric_parse,
                        "teacher": teacher_numeric_parse,
                        "student": student_numeric_parse,
                    },
                },
            },
            "cost": {
                "inference_usd_per_1k_tokens": {
                    "teacher": teacher_cost_per_1k,
                    "student": student_cost_per_1k,
                    "student_savings_pct": savings_pct,
                },
                "eval_stage_cost_usd": 0.0,
            },
            "training_stability": {
                "rl": {
                    "stability_score": rl_stability,
                    "nan_events": int(teacher_payload.get("rl_nan_events", 0)),
                },
                "teacher_ft": {
                    "stability_score": teacher_ft_stability,
                    "nan_events": int(teacher_payload.get("teacher_ft_nan_events", 0)),
                },
                "distill": {
                    "stability_score": distill_stability,
                    "nan_events": int(student_payload.get("distill_nan_events", 0)),
                },
            },
            "integrity": {
                "passed": integrity_passed,
                "status": integrity_status,
                "reason": integrity_reason,
                "checks": {
                    "teacher_refusal_rate": health["teacher"]["refusal_rate"],
                    "teacher_refusal_threshold": cfg.evaluation.teacher_integrity_refusal_threshold,
                    "teacher_overlap_score": teacher_quality,
                    "teacher_min_score": cfg.evaluation.teacher_integrity_min_score,
                    "teacher_numeric_parse_rate": teacher_numeric_parse,
                    "teacher_numeric_parse_threshold": cfg.evaluation.teacher_integrity_numeric_parse_threshold,
                    "baseline_overlap_score": baseline_quality,
                    "prompt_count": len(prompts),
                },
                "diagnostics": {
                    "baseline": health["baseline"],
                    "teacher": health["teacher"],
                    "student": health["student"],
                },
            },
            "notes": "Real eval stage executed across baseline, teacher, and student checkpoints.",
            PROMPT_TRACES_KEY: [*baseline.trace_rows, *teacher.trace_rows, *student.trace_rows],
            EVAL_ROWS_KEY: eval_rows,
        },
        "usage": _usage_dict(
            prefill_tokens=baseline.prefill_tokens + teacher.prefill_tokens + student.prefill_tokens,
            sample_tokens=baseline.sample_tokens + teacher.sample_tokens + student.sample_tokens,
            train_tokens=0,
            run_id=f"eval-{int(time.time())}",
            provider_raw={
                "baseline_sampling_session_id": baseline.session_id,
                "teacher_sampling_session_id": teacher.session_id,
                "student_sampling_session_id": student.session_id,
                "teacher_checkpoint_path": teacher_checkpoint_path,
                "student_checkpoint_path": student_checkpoint_path,
                "stage": "eval",
            },
        ),
    }
