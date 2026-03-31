from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from inference_projects.pricing import TokenUsage


@dataclass
class StageRecord:
    mode: str
    stage: str
    projected_cost_usd: float
    actual_cost_usd: float
    projected_tokens: TokenUsage
    actual_tokens: TokenUsage
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "stage": self.stage,
            "projected_cost_usd": round(self.projected_cost_usd, 4),
            "actual_cost_usd": round(self.actual_cost_usd, 4),
            "projected_tokens": self.projected_tokens.as_dict(),
            "actual_tokens": self.actual_tokens.as_dict(),
            "status": self.status,
        }


@dataclass
class Ledger:
    total_spend_usd: float
    stage_spend_usd: dict[str, float]
    token_totals: TokenUsage
    records: list[StageRecord]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "total_spend_usd": round(self.total_spend_usd, 4),
            "stage_spend_usd": {k: round(v, 4) for k, v in self.stage_spend_usd.items()},
            "token_totals": self.token_totals.as_dict(),
            "records": [record.as_dict() for record in self.records],
        }


def new_ledger() -> Ledger:
    return Ledger(
        total_spend_usd=0.0,
        stage_spend_usd={"rl": 0.0, "fsdp": 0.0, "distill": 0.0, "eval": 0.0},
        token_totals=TokenUsage(prefill=0, sample=0, train=0),
        records=[],
    )


def load_ledger(path: Path) -> Ledger:
    if not path.exists():
        return new_ledger()
    raw = json.loads(path.read_text())
    records = [
        StageRecord(
            mode=row.get("mode", "mock"),
            stage=row["stage"],
            projected_cost_usd=float(row["projected_cost_usd"]),
            actual_cost_usd=float(row["actual_cost_usd"]),
            projected_tokens=TokenUsage(**row["projected_tokens"]),
            actual_tokens=TokenUsage(**row["actual_tokens"]),
            status=row["status"],
        )
        for row in raw.get("records", [])
    ]
    totals = raw.get("token_totals", {})
    return Ledger(
        total_spend_usd=float(raw.get("total_spend_usd", 0.0)),
        stage_spend_usd={k: float(v) for k, v in raw.get("stage_spend_usd", {}).items()},
        token_totals=TokenUsage(
            prefill=int(totals.get("prefill", 0)),
            sample=int(totals.get("sample", 0)),
            train=int(totals.get("train", 0)),
        ),
        records=records,
    )


def save_ledger(path: Path, ledger: Ledger) -> None:
    from inference_projects.schemas import validate_ledger_payload

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ledger.as_dict()
    validate_ledger_payload(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def add_record(ledger: Ledger, record: StageRecord) -> Ledger:
    updated_stage_spend = dict(ledger.stage_spend_usd)
    updated_stage_spend[record.stage] = round(updated_stage_spend.get(record.stage, 0.0) + record.actual_cost_usd, 4)
    updated = Ledger(
        total_spend_usd=round(ledger.total_spend_usd + record.actual_cost_usd, 4),
        stage_spend_usd=updated_stage_spend,
        token_totals=TokenUsage(
            prefill=ledger.token_totals.prefill + record.actual_tokens.prefill,
            sample=ledger.token_totals.sample + record.actual_tokens.sample,
            train=ledger.token_totals.train + record.actual_tokens.train,
        ),
        records=[*ledger.records, record],
    )
    return updated
