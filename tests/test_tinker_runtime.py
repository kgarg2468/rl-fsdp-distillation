from inference_projects.config import load_config
from inference_projects.tinker_runtime import (
    CANARY_FIXTURE_PATH,
    _cost_per_1k,
    _sampling_model_path_candidates,
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
