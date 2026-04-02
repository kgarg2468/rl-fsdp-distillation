from inference_projects.config import load_config
from dataclasses import replace
from inference_projects.tinker_runtime import (
    CANARY_FIXTURE_PATH,
    SamplingBatch,
    TrainingCheckpoint,
    _cost_per_1k,
    _exact_numeric_match,
    _make_training_datum,
    _is_refusal_like,
    _model_health,
    _numeric_parse_rate,
    _run_training_loop,
    _sampling_model_path_candidates,
    continue_from_checkpoint,
    create_lora_checkpoint,
    load_canary_prompts,
    run_real_teacher_ft,
    run_real_distill,
    run_real_rl,
)


def test_canary_fixture_has_expected_prompt_count():
    rows = load_canary_prompts(fixture_path=CANARY_FIXTURE_PATH)
    assert len(rows) == 8
    assert all(row.prompt for row in rows)


def test_canary_fixture_limit_is_respected():
    rows = load_canary_prompts(limit=3, fixture_path=CANARY_FIXTURE_PATH)
    assert len(rows) == 3


def test_cost_per_1k_returns_zero_for_no_tokens():
    cfg = load_config()
    assert _cost_per_1k(cfg=cfg, prefill_tokens=0, sample_tokens=0) == 0.0


def test_cost_per_1k_positive_for_token_usage():
    cfg = load_config()
    value = _cost_per_1k(cfg=cfg, prefill_tokens=1000, sample_tokens=1000)
    assert value > 0


def test_sampling_model_path_candidates_adds_sampler_weights_variant():
    candidates = _sampling_model_path_candidates("tinker://run-1/weights/checkpoint-001")
    assert candidates[0] == "tinker://run-1/weights/checkpoint-001"
    assert "tinker://run-1/sampler_weights/checkpoint-001" in candidates


def test_sampling_model_path_candidates_does_not_duplicate_sampler_weights():
    candidates = _sampling_model_path_candidates("tinker://run-1/sampler_weights/checkpoint-001")
    assert candidates == ["tinker://run-1/sampler_weights/checkpoint-001"]


def test_exact_numeric_match_and_parse_rate():
    assert _exact_numeric_match("answer: 12", "12") == 1.0
    assert _exact_numeric_match("answer: 13", "12") == 0.0
    assert _numeric_parse_rate("value 42") == 1.0
    assert _numeric_parse_rate("no number present") == 0.0


def test_refusal_detection_and_model_health():
    assert _is_refusal_like("Sorry, I cannot comply.")
    health = _model_health(["", "No words.", "42"])
    assert health["empty_output_rate"] == 0.3333
    assert health["refusal_rate"] >= 0.3333


def test_create_lora_checkpoint_uses_sampler_weights_save(monkeypatch):
    captured: list[dict[str, object]] = []

    class FakeTrainClient:
        def __init__(self):
            self.model_id = "model-1"

        def get_info(self):
            return type("Info", (), {"model_id": self.model_id})()

        def save_state(self, name):
            raise AssertionError("save_state should not be used for sampling checkpoints")

        def save_weights_for_sampler(self, name):
            return type("Resp", (), {"result": lambda self: type("Saved", (), {"path": "tinker://x/sampler_weights/y"})()})()

    class FakeService:
        def create_lora_training_client(self, **kwargs):
            _ = kwargs
            return FakeTrainClient()

    def fake_save_new_checkpoint(**kwargs):
        entry = {
            "callable_name": kwargs["save_state_callable"].__name__,
            "wait_for_checkpoint": kwargs["wait_for_checkpoint"],
        }
        captured.append(entry)
        if kwargs["save_state_callable"].__name__ == "save_state":
            return "tinker://x/weights/y"
        return "tinker://x/sampler_weights/y"

    monkeypatch.setattr("inference_projects.tinker_runtime._save_new_checkpoint", fake_save_new_checkpoint)
    checkpoint = create_lora_checkpoint(
        service=FakeService(),
        base_model="meta-llama/Llama-3.1-8B",
        stage="rl",
        poll_interval_seconds=1,
        timeout_seconds=1,
    )
    assert checkpoint.checkpoint_path == "tinker://x/weights/y"
    assert checkpoint.sampler_checkpoint_path == "tinker://x/sampler_weights/y"
    assert [entry["callable_name"] for entry in captured] == ["save_state", "save_weights_for_sampler"]
    assert [entry["wait_for_checkpoint"] for entry in captured] == [True, False]


def test_continue_from_checkpoint_uses_sampler_weights_save(monkeypatch):
    captured: list[dict[str, object]] = []

    class FakeTrainClient:
        def __init__(self):
            self.model_id = "model-2"

        def get_info(self):
            return type("Info", (), {"model_id": self.model_id})()

        def save_state(self, name):
            raise AssertionError("save_state should not be used for sampling checkpoints")

        def save_weights_for_sampler(self, name):
            return type("Resp", (), {"result": lambda self: type("Saved", (), {"path": "tinker://x/sampler_weights/z"})()})()

    class FakeService:
        def create_training_client_from_state(self, checkpoint_path, user_metadata=None):
            _ = (checkpoint_path, user_metadata)
            return FakeTrainClient()

    def fake_save_new_checkpoint(**kwargs):
        entry = {
            "callable_name": kwargs["save_state_callable"].__name__,
            "wait_for_checkpoint": kwargs["wait_for_checkpoint"],
        }
        captured.append(entry)
        if kwargs["save_state_callable"].__name__ == "save_state":
            return "tinker://x/weights/z"
        return "tinker://x/sampler_weights/z"

    monkeypatch.setattr("inference_projects.tinker_runtime._save_new_checkpoint", fake_save_new_checkpoint)
    checkpoint = continue_from_checkpoint(
        service=FakeService(),
        checkpoint_path="tinker://x/weights/y",
        stage="teacher_ft",
        poll_interval_seconds=1,
        timeout_seconds=1,
    )
    assert checkpoint.checkpoint_path == "tinker://x/weights/z"
    assert checkpoint.sampler_checkpoint_path == "tinker://x/sampler_weights/z"
    assert [entry["callable_name"] for entry in captured] == ["save_state", "save_weights_for_sampler"]
    assert [entry["wait_for_checkpoint"] for entry in captured] == [True, False]


def test_make_training_datum_falls_back_for_empty_tokens():
    class EmptyTokenizer:
        eos_token_id = 7

        def encode(self, text):
            _ = text
            return []

    datum, target_tokens = _make_training_datum(tokenizer=EmptyTokenizer(), prompt="", target="")
    assert target_tokens == 2
    assert datum.model_input.chunks[0].tokens == [7, 7]
    assert datum.loss_fn_inputs["target_tokens"].data == [0, 7]
    assert datum.loss_fn_inputs["weights"].data == [0.0, 1.0]


def test_run_training_loop_tracks_steps_and_losses():
    class FakeTokenizer:
        def encode(self, text):
            return [len(text), 1]

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeTrainClient:
        def __init__(self):
            self.learning_rates = []

        def get_tokenizer(self):
            return FakeTokenizer()

        def forward_backward(self, batch, loss_fn):
            _ = loss_fn
            for datum in batch:
                input_tokens = datum.model_input.chunks[0].tokens
                target_tokens = datum.loss_fn_inputs["target_tokens"].data
                weights = datum.loss_fn_inputs["weights"].data
                assert len(target_tokens) == len(input_tokens)
                assert len(weights) == len(input_tokens)
            return FakeFuture(type("FwdBwd", (), {"metrics": {"loss": 0.25}, "loss_fn_outputs": []})())

        def optim_step(self, adam_params):
            self.learning_rates.append(adam_params.learning_rate)
            return FakeFuture(type("Optim", (), {"metrics": {}})())

    client = FakeTrainClient()
    summary = _run_training_loop(
        train_client=client,
        examples=[("p1", "t1"), ("p2", "t2"), ("p3", "t3")],
        epochs=2,
        batch_size=2,
        learning_rate=2e-5,
        warmup_ratio=0.5,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        max_consecutive_failures=3,
        stage="distill",
    )

    assert summary["steps"] == 4
    assert summary["batches_per_epoch"] == 2
    assert len(summary["loss_trace"]) == 4
    assert summary["micro_batch_fallbacks"] == 0
    assert client.learning_rates[0] < 2e-5
    assert client.learning_rates[-1] == 2e-5


def test_run_training_loop_uses_micro_batch_fallback_on_shape_errors():
    class FakeTokenizer:
        def encode(self, text):
            return [len(text), 1]

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeTrainClient:
        def __init__(self):
            self.batch_calls = 0
            self.single_calls = 0
            self.optim_calls = 0

        def get_tokenizer(self):
            return FakeTokenizer()

        def forward_backward(self, batch, loss_fn):
            _ = loss_fn
            if len(batch) > 1:
                self.batch_calls += 1
                raise RuntimeError("Could not convert loss function inputs to array record")
            self.single_calls += 1
            return FakeFuture(type("FwdBwd", (), {"metrics": {"loss": 0.5}, "loss_fn_outputs": []})())

        def optim_step(self, adam_params):
            _ = adam_params
            self.optim_calls += 1
            return FakeFuture(type("Optim", (), {"metrics": {}})())

    client = FakeTrainClient()
    summary = _run_training_loop(
        train_client=client,
        examples=[("p1", "t1"), ("p2", "t2"), ("p3", "t3")],
        epochs=1,
        batch_size=2,
        learning_rate=1e-5,
        warmup_ratio=0.0,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        max_consecutive_failures=3,
        stage="rl",
    )

    assert summary["steps"] == 2
    assert summary["micro_batch_fallbacks"] == 1
    assert client.batch_calls == 1
    assert client.single_calls == 3
    assert client.optim_calls == 2


def test_run_real_rl_records_training_metadata(monkeypatch):
    cfg = load_config()
    cfg = replace(cfg, evaluation=replace(cfg.evaluation, prompt_limit=2))

    class FakeTrainClient:
        def get_info(self):
            return type("Info", (), {"model_id": "run-rl"})()

    class FakeService:
        def create_lora_training_client(self, **kwargs):
            _ = kwargs
            return FakeTrainClient()

    monkeypatch.setattr("inference_projects.tinker_runtime.build_service_client", lambda: FakeService())
    monkeypatch.setattr(
        "inference_projects.tinker_runtime.load_canary_prompts",
        lambda **kwargs: [
            type("PromptRow", (), {"row_id": "r1", "prompt": "1+1", "reference": "2"})(),
            type("PromptRow", (), {"row_id": "r2", "prompt": "2+2", "reference": "4"})(),
        ],
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._run_training_loop",
        lambda **kwargs: {
            "steps": 3,
            "batches_per_epoch": 2,
            "nan_events": 0,
            "loss_trace": [0.5, 0.4, 0.3],
            "train_tokens": 12,
        },
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._save_training_checkpoints",
        lambda **kwargs: TrainingCheckpoint(
            run_id="run-rl",
            checkpoint_path="tinker://ckpt/rl",
            sampler_checkpoint_path="tinker://sampler/rl",
        ),
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime.sample_prompts",
        lambda **kwargs: SamplingBatch(
            outputs=["2", "4"],
            prefill_tokens=8,
            sample_tokens=2,
            session_id="sess-1",
            trace_rows=[],
        ),
    )

    result = run_real_rl(cfg=cfg)
    assert result["payload"]["training"]["steps"] == 3
    assert result["payload"]["rl_nan_events"] == 0
    assert result["usage"]["train_tokens"] == 12


def test_run_real_teacher_ft_records_training_metadata(monkeypatch):
    cfg = load_config()
    cfg = replace(cfg, evaluation=replace(cfg.evaluation, prompt_limit=2))

    class FakeTrainClient:
        def get_info(self):
            return type("Info", (), {"model_id": "run-teacher_ft"})()

    class FakeService:
        def create_training_client_from_state_with_optimizer(self, checkpoint_path, user_metadata=None):
            _ = (checkpoint_path, user_metadata)
            return FakeTrainClient()

    monkeypatch.setattr("inference_projects.tinker_runtime.build_service_client", lambda: FakeService())
    monkeypatch.setattr(
        "inference_projects.tinker_runtime.load_canary_prompts",
        lambda **kwargs: [
            type("PromptRow", (), {"row_id": "r1", "prompt": "1+1", "reference": "2"})(),
            type("PromptRow", (), {"row_id": "r2", "prompt": "2+2", "reference": "4"})(),
        ],
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._run_training_loop",
        lambda **kwargs: {
            "steps": 4,
            "batches_per_epoch": 2,
            "nan_events": 1,
            "loss_trace": [0.7, 0.5],
            "train_tokens": 16,
        },
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._save_training_checkpoints",
        lambda **kwargs: TrainingCheckpoint(
            run_id="run-teacher_ft",
            checkpoint_path="tinker://ckpt/teacher_ft",
            sampler_checkpoint_path="tinker://sampler/teacher_ft",
        ),
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime.sample_prompts",
        lambda **kwargs: SamplingBatch(
            outputs=["2", "4"],
            prefill_tokens=8,
            sample_tokens=2,
            session_id="sess-2",
            trace_rows=[],
        ),
    )

    result = run_real_teacher_ft(
        cfg=cfg,
        teacher_payload={
            "checkpoint_path": "tinker://ckpt/rl",
            "quality_score": 0.4,
            "rl_stability_score": 0.9,
            "rl_nan_events": 0,
        },
    )
    assert result["payload"]["training"]["steps"] == 4
    assert result["payload"]["teacher_ft_nan_events"] == 1
    assert result["usage"]["train_tokens"] == 16


def test_run_real_distill_uses_scaled_training_data(monkeypatch):
    cfg = load_config()
    cfg = replace(cfg, distillation=replace(cfg.distillation, training_prompt_limit=3))

    class FakeTrainClient:
        def get_info(self):
            return type("Info", (), {"model_id": "run-distill"})()

    class FakeService:
        def create_lora_training_client(self, **kwargs):
            _ = kwargs
            return FakeTrainClient()

    rows = [
        type("PromptRow", (), {"row_id": "r1", "prompt": "1+1", "reference": "2"})(),
        type("PromptRow", (), {"row_id": "r2", "prompt": "2+2", "reference": "4"})(),
        type("PromptRow", (), {"row_id": "r3", "prompt": "3+3", "reference": "6"})(),
    ]
    monkeypatch.setattr("inference_projects.tinker_runtime.build_service_client", lambda: FakeService())
    monkeypatch.setattr("inference_projects.tinker_runtime.load_canary_prompts", lambda **kwargs: rows)
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._run_training_loop",
        lambda **kwargs: {
            "steps": 5,
            "batches_per_epoch": 2,
            "nan_events": 0,
            "loss_trace": [0.8, 0.6],
            "train_tokens": 20,
        },
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._save_training_checkpoints",
        lambda **kwargs: TrainingCheckpoint(
            run_id="run-distill",
            checkpoint_path="tinker://ckpt/distill",
            sampler_checkpoint_path="tinker://sampler/distill",
        ),
    )

    def fake_sample_prompts(**kwargs):
        label = kwargs["model_label"]
        if label == "teacher":
            outputs = ["2", "4", "6"]
            return SamplingBatch(outputs=outputs, prefill_tokens=10, sample_tokens=3, session_id="teacher", trace_rows=[])
        if label == "baseline":
            outputs = ["1", "4", "0"]
            return SamplingBatch(outputs=outputs, prefill_tokens=9, sample_tokens=3, session_id="baseline", trace_rows=[])
        return SamplingBatch(
            outputs=["2", "4", "6"],
            prefill_tokens=8,
            sample_tokens=3,
            session_id="student",
            trace_rows=[],
        )

    monkeypatch.setattr("inference_projects.tinker_runtime.sample_prompts", fake_sample_prompts)

    result = run_real_distill(
        cfg=cfg,
        teacher_payload={
            "checkpoint_path": "tinker://ckpt/teacher_ft",
            "sampler_checkpoint_path": "tinker://sampler/teacher_ft",
            "quality_score": 0.8,
        },
    )
    assert result["payload"]["training"]["steps"] == 5
    assert result["payload"]["distillation_config"]["training_prompt_limit"] == 3
    assert result["payload"]["distill_dataset"]["total_rows"] == 3
    assert result["usage"]["train_tokens"] == 20


def test_run_real_distill_uses_distillation_training_prompt_file_when_set(monkeypatch):
    cfg = load_config()
    cfg = replace(
        cfg,
        distillation=replace(
            cfg.distillation,
            training_prompt_limit=2,
            training_prompt_file=CANARY_FIXTURE_PATH,
        ),
    )

    class FakeTrainClient:
        def get_info(self):
            return type("Info", (), {"model_id": "run-distill"})()

    class FakeService:
        def create_lora_training_client(self, **kwargs):
            _ = kwargs
            return FakeTrainClient()

    rows = [
        type("PromptRow", (), {"row_id": "r1", "prompt": "1+1", "reference": "2"})(),
        type("PromptRow", (), {"row_id": "r2", "prompt": "2+2", "reference": "4"})(),
    ]
    seen_fixture_paths = []

    def fake_load_canary_prompts(**kwargs):
        seen_fixture_paths.append(kwargs["fixture_path"])
        return rows

    monkeypatch.setattr("inference_projects.tinker_runtime.build_service_client", lambda: FakeService())
    monkeypatch.setattr("inference_projects.tinker_runtime.load_canary_prompts", fake_load_canary_prompts)
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._run_training_loop",
        lambda **kwargs: {
            "steps": 4,
            "batches_per_epoch": 2,
            "nan_events": 0,
            "loss_trace": [0.7, 0.5],
            "train_tokens": 12,
        },
    )
    monkeypatch.setattr(
        "inference_projects.tinker_runtime._save_training_checkpoints",
        lambda **kwargs: TrainingCheckpoint(
            run_id="run-distill",
            checkpoint_path="tinker://ckpt/distill",
            sampler_checkpoint_path="tinker://sampler/distill",
        ),
    )

    def fake_sample_prompts(**kwargs):
        label = kwargs["model_label"]
        if label == "teacher":
            return SamplingBatch(
                outputs=["2", "4"],
                prefill_tokens=8,
                sample_tokens=2,
                session_id="teacher",
                trace_rows=[],
            )
        if label == "baseline":
            return SamplingBatch(
                outputs=["0", "0"],
                prefill_tokens=8,
                sample_tokens=2,
                session_id="baseline",
                trace_rows=[],
            )
        return SamplingBatch(
            outputs=["2", "4"],
            prefill_tokens=8,
            sample_tokens=2,
            session_id="student",
            trace_rows=[],
        )

    monkeypatch.setattr("inference_projects.tinker_runtime.sample_prompts", fake_sample_prompts)

    run_real_distill(
        cfg=cfg,
        teacher_payload={
            "checkpoint_path": "tinker://ckpt/teacher_ft",
            "sampler_checkpoint_path": "tinker://sampler/teacher_ft",
        },
    )
    assert seen_fixture_paths == [CANARY_FIXTURE_PATH]
