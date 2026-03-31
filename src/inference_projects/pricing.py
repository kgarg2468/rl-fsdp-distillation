from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    prefill: int
    sample: int
    train: int

    def as_dict(self) -> dict[str, int]:
        return {"prefill": self.prefill, "sample": self.sample, "train": self.train}


def cost_usd(usage: TokenUsage, *, prefill_rate: float, sample_rate: float, train_rate: float) -> float:
    return round(
        (usage.prefill / 1_000_000.0) * prefill_rate
        + (usage.sample / 1_000_000.0) * sample_rate
        + (usage.train / 1_000_000.0) * train_rate,
        4,
    )
