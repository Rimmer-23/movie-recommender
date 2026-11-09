# CLAUDE.md — Movie Recommender (PyTorch RecSys)

## Goal

Build a complete, portfolio-ready **recommender system in PyTorch** on MovieLens 1M: data pipeline, several models (baseline → embeddings → two-tower → sequential), proper evaluation, a recommendation API, batch export, Docker and CI.

Build the whole project end to end. Work through the milestones below in order without stopping for confirmation; only stop if something is genuinely blocked. Talk to me in Russian; code, comments, commit messages and README in English.

## Tech stack

- Python 3.11+, **PyTorch** (pick device automatically: CUDA → MPS → CPU)
- pandas + numpy for data prep
- FastAPI + uvicorn for serving
- pytest, ruff
- Docker, GitHub Actions (lint + tests)
- TensorBoard for training curves

## Data

- **MovieLens 1M** (`ml-1m`): ~1M ratings, ~6k users, ~3.7k movies, with timestamps. Trains on a laptop CPU.
- `scripts/download_data.py` downloads `https://files.grouplens.org/datasets/movielens/ml-1m.zip` into `data/raw/`.
- **Never commit data or model artifacts.** `data/` and `artifacts/` are in `.gitignore` (MovieLens license doesn't allow redistribution).
- Treat every rating as an implicit interaction. Sort each user's history by timestamp. Re-index user and item IDs to contiguous integers (0 reserved for padding where needed).

## Evaluation protocol (identical for every model)

- **Leave-one-out by time:** per user, last interaction → test, second-to-last → validation, the rest → train.
- **Full ranking** over all items (no sampled negatives), excluding items seen in the user's history.
- Metrics: **Recall@10, NDCG@10, Recall@20, NDCG@20** in `src/recsys/metrics.py`, with unit tests.
- Fixed seed everywhere. Validation set for early stopping and model selection, test set reported once at the end.

## Milestones

1. **Setup & data** — repo structure, `pyproject.toml`, ruff config, pytest, `.gitignore`, download + preprocessing, ID re-indexing, splits, dataset stats printed.
2. **Most Popular baseline** — plus metrics and a shared evaluation loop.
3. **Matrix Factorization (BPR)** — user/item `nn.Embedding`, dot-product score, BPR loss with negative sampling, `Dataset`/`DataLoader`, training loop with validation and early stopping.
4. **Two-tower model** — user tower and item tower (embedding + MLP), in-batch negatives with softmax loss.
5. **SASRec** — self-attention sequential recommender: item + positional embeddings, causal mask, padding mask, 2 blocks, 1 head, hidden 64, max_len 200, dropout 0.2, cross-entropy over all items. Tune lightly on validation.
6. **Serving & batch export**
   - FastAPI (`src/recsys/api.py`): `GET /health`, `GET /recommend/{user_id}?k=10` returning movie IDs, titles and scores from the best model; unknown user → popular fallback.
   - `scripts/export_recommendations.py`: top-k for all users into `artifacts/recommendations.parquet` (columns: user_id, rank, movie_id, title, score, model, generated_at), as if delivering an enriched table to a data warehouse.
7. **Docker & CI** — Dockerfile for the API (slim image, CPU torch), `docker-compose.yml` optional; `.github/workflows/ci.yml` running ruff + pytest on push and PR.
8. **README** — overview, architecture diagram (Mermaid), data, evaluation protocol, **results table** (Popular vs MF vs Two-tower vs SASRec on test), training curves screenshot, how to run (train, evaluate, serve, Docker), project structure, next steps.

## Project structure

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Dockerfile
├── .github/workflows/ci.yml
├── configs/                 # one YAML per model
├── scripts/
│   ├── download_data.py
│   ├── train.py             # python scripts/train.py --model sasrec
│   ├── evaluate.py          # evaluates all trained models, prints results table
│   └── export_recommendations.py
├── src/recsys/
│   ├── data.py              # loading, preprocessing, splits, Datasets
│   ├── metrics.py
│   ├── evaluate.py          # full-ranking evaluation
│   ├── models/
│   │   ├── base.py          # common interface
│   │   ├── popular.py
│   │   ├── mf.py
│   │   ├── two_tower.py
│   │   └── sasrec.py
│   └── api.py
├── notebooks/
│   └── 01_eda.ipynb         # short EDA: interactions per user/item, popularity long tail
├── tests/
├── data/                    # gitignored
└── artifacts/               # gitignored: checkpoints, exports, tensorboard runs
```

## Conventions

- All hyperparameters in `configs/*.yaml`, loaded into dataclasses. No magic numbers in training code.
- Every model implements the same interface (`fit`, `recommend(user_ids, k, exclude_seen=True)`, `save`, `load`) so evaluation and serving code are shared.
- Comment tensor shapes in model code, e.g. `# (batch, seq_len, hidden)`.
- Type hints and docstrings on public functions. Clean, readable code — this will be read on GitHub.
- Tests: metrics correctness, no leakage between splits, model output shapes, a tiny end-to-end training run on a small sample, API endpoints (FastAPI TestClient).
- Run `ruff check .`, `ruff format .` and `pytest` after each milestone. Commit after each milestone with a clear message.

## Definition of done

- All milestones complete; tests and CI green.
- README results table with real numbers; SASRec ≥ MF ≥ Most Popular on NDCG@10 (if not, investigate and document why).
- `docker build` + `docker run` serves recommendations at `/recommend/{user_id}`.
- One command per step: download → train → evaluate → export → serve, documented in the README.
