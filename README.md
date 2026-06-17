# Chess Oracle

Predicting **Titled Tuesday** blitz game outcomes (`loss` / `draw` / `win`, from
White's perspective) on Chess.com, using only leakage-safe pre-game signals:
player Elo and title.

The short version: an LLM explored the Chess.com API and wrote the data pipeline;
we found the entire predictable signal lives in the **rating gap**, so we kept
just six rating/title features; a broad model bake-off (Naive Bayes, Random
Forest, Logistic Regression, ordinal linear regression, XGBoost, MLP) all plateau
around **66% accuracy**; **draws are essentially unpredictable** from these
features; and a closed-form **Bayesian Dirichlet-Multinomial "smart counting"
model** matches the best black box while staying interpretable and updatable in
`O(1)`.

An LLM was further used to help with the redaction of this document.

## Layout

- **[`chess_oracle.ipynb`](chess_oracle.ipynb)** — the full story end to end: data
  pipeline, EDA, feature selection, model bake-off, the Bayesian model, a
  head-to-head comparison, and future directions.
- [`build_dataset.py`](build_dataset.py) — crawls the Chess.com API and builds
  `data/processed/games.parquet` (leakage-safe, lagged pre-game ratings).
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) — column reference and leakage tiers.
- `data/` — cached raw API responses and the processed dataset.
- `api_output_samples/` — example API payloads the pipeline was built from.

## Getting started

```bash
pip install -r requirements.txt
python build_dataset.py        # builds data/processed/games.parquet
```

Then open **[`chess_oracle.ipynb`](chess_oracle.ipynb)** and run it top to bottom.

**For details, check [`chess_oracle.ipynb`](chess_oracle.ipynb).**
