# RL + FSDP + Distillation Eval Report

## Run Metadata
- Run mode: mock
- Setup status: ready
- Schema version: 1.0

## Disclaimer
- This report was generated in mock mode with deterministic stage adapters for showcase workflow validation.

## Quality
- Baseline benchmark score: 0.61
- Teacher benchmark score: 0.74
- Student benchmark score: 0.7
- Student retention vs teacher: 0.9459
- LLM judge win rate (student vs baseline): 0.66
- LLM judge win rate (student vs teacher): 0.44

## Cost
- Projected spend (USD): 20.24
- Actual spend (USD): 19.03
- Teacher inference cost / 1k tokens (USD): 0.00027
- Student inference cost / 1k tokens (USD): 0.00011
- Student inference savings (%): 59.26
- Projected token totals: prefill=8000000, sample=27000000, train=21000000
- Actual token totals: prefill=7500000, sample=25260000, train=19880000

## Training Stability
- RL stability score: 0.91 (NaN events: 0)
- FSDP stability score: 0.89 (NaN events: 0)
- Distill stability score: 0.9 (NaN events: 0)

## Stage Spend
- rl: $7.89
- fsdp: $5.24
- distill: $4.71
- eval: $1.20
