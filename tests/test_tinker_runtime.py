from inference_projects.config import load_config
from inference_projects.tinker_runtime import (
    CANARY_FIXTURE_PATH,
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
        stage="fsdp",
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
    assert target_tokens == 1
    assert datum.model_input.chunks[0].tokens == [7]
    assert datum.loss_fn_inputs["target_tokens"].data == [7]


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
            _ = (batch, loss_fn)
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
    assert client.learning_rates[0] < 2e-5
    assert client.learning_rates[-1] == 2e-5
