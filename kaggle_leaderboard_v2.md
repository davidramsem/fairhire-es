# FairHire-ES v2 (explicit 0-100 integer scale)

| Model | Composite (robust) | Composite (raw) | Kaggle score | MAE raw ↓ | MAE ×100-fix ↓ | F1 ↑ | Twin spread raw ↓ | Twin spread rescaled ↓ | % fit≤1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.7-flash` | 0.9796 | 0.9796 | 0.9796 | 3.6 | 3.6 | 1.0 | 2.4 | 2.4 | 0.0 |
| `claude-haiku-4-5-20251001` | 0.9644 | 0.9644 | 0.9644 | 8.9 | 8.9 | 1.0 | 0.0 | 0.0 | 0.0 |
| `gemini-2.5-flash` | 0.9635 | 0.9635 | 0.9635 | 2.5 | 2.5 | 0.95 | 3.6 | 3.6 | 0.0 |
| `grok-4.20-0309-non-reasoning` | 0.9409 | 0.9409 | 0.9409 | 7.9 | 7.9 | 0.95 | 4.0 | 4.0 | 0.0 |
| `gpt-5.4-mini-2026-03-17` | 0.8714 | 0.8714 | 0.8714 | 11.65 | 11.65 | 0.85 | 11.8 | 11.8 | 0.0 |
