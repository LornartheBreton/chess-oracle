# Titled Tuesday Game-Outcome Prediction — Data, Features & Modeling Notes

Predicting the result of a chess game (win / draw / loss from White's perspective) **before it starts**, using the Chess.com PubAPI for two Titled Tuesday events:

| Event | Games kept | Players |
|---|---|---|
| Feb 10 2026 (`6221327`) | 2,015 | ~455 |
| Mar 10 2026 (`6277141`) | 2,033 | ~476 |

Total: **4,048 games, 700 unique players** (15 games dropped for having no moves — forfeits/no-shows).

## Reproducing

```bash
pip install -r requirements.txt
python build_dataset.py        # writes data/processed/games.{csv,parquet}
jupyter notebook eda.ipynb     # EDA + baselines
```

All HTTP responses are cached under `data/raw/` (keyed by URL), so the second run is fully offline and deterministic. Requests carry a `User-Agent` and are rate-limited with retry/backoff. Note: the tournament *root* endpoint currently 404s for these events, so the crawler probes round endpoints (`/{id}/{round}`) sequentially until the first 404.

## 1. Dataset construction

One row per game. The target `outcome ∈ {win, draw, loss}` is derived from the API's per-side result codes:

- **win**: `win`
- **draw**: `agreed`, `repetition`, `stalemate`, `insufficient`, `50move`, `timevsinsufficient`
- **loss**: `checkmated`, `resigned`, `timeout`, `abandoned`

A numeric companion `outcome_score ∈ {1, 0.5, 0}` is included for score-style analyses.

### 1.1 The rating-leakage audit (the subtle part)

The per-game `white.rating` / `black.rating` fields are not documented as pre- or post-game. I tested this empirically: for each player's consecutive rounds $r \to r+1$, the rating delta $\Delta_r = R_{r+1} - R_r$ should correlate with the result of round $r$ if the field is **pre**-game (the delta absorbs round $r$'s K-factor update), and with the result of round $r+1$ if it is **post**-game.

On 7,124 consecutive-round pairs:

$$\operatorname{corr}(\Delta, s_{\text{prev}}) = -0.000 \qquad \operatorname{corr}(\Delta, s_{\text{same}}) = 0.615$$

The field is unambiguously **post-game** — using it raw would leak the label (a win inflates your reported rating *in that same row*). The pipeline therefore reconstructs pre-game ratings by **lagging each player's reported rating by one round** within the tournament. For round 1 there is no prior observation; the fallback is the reported value itself, whose bias is bounded by one game's K-factor update (~±10 Elo, negligible against rating gaps of hundreds). I preferred this over the stats-endpoint blitz rating because that endpoint is a *today* snapshot, months after the games (see §2.3).

### 1.2 Excluded fields (post-game information)

PGN movetext, final FEN, `Termination`, `end_time` / game duration, and — easy to miss — the **ECO opening code**: the opening is determined by moves played *during* the game. Result codes are used only to construct the label. Usernames and game URLs are retained as identifiers (for grouped splits), not features.

## 2. Features

All features come in `white_*`, `black_*`, and (where sensible) `diff_* = white − black` variants.

### 2.1 Ratings — the workhorse

- `white_elo`, `black_elo` (leakage-corrected, §1.1), `diff_elo`, `mean_elo`, `abs_diff_elo`
- `white_elo_expected` $= \big(1+10^{-d/400}\big)^{-1}$, the Elo expected score — a domain-informed nonlinearity so linear models don't have to learn the sigmoid themselves.

**Why both $d$ and $|d|$ / mean?** The win–loss axis is monotone in $d$, but the *draw* probability is unimodal around $d=0$ and (in classical chess) increases with pairing strength. $|d|$ and `mean_elo` let even a linear-in-features model express the draw bump.

### 2.2 In-tournament history (strictly prior rounds)

Computed by a forward pass in round order; round-1 values are 0/NaN by construction (verified):

- `points_so_far`, `score_rate`, `wins/draws/losses_so_far`, signed `streak`
- `perf_vs_expected` $= \sum_i (s_i - E_i)$: realized minus Elo-expected points — a *form residual* orthogonal to rating by construction. Raw correlation with outcome is negative (−0.29) because Swiss pairing confounds it: overperformers get paired up. Against the Elo-residual outcome it is **positive (+0.08)** — modest genuine form signal, and a nice example of why marginal correlations mislead under adaptive pairing.
- `white_games_so_far` (color balance), `avg_opp_elo_so_far` (strength of schedule), `round`.

### 2.3 Career & account features (profile + stats endpoints)

Per side: title (raw + ordinal GM=7 … WNM=1), account age at game time, log followers, streamer flag, league, country; blitz Glicko RD (skill *uncertainty* — directly usable as known measurement error in a Bayesian model), log career blitz games, career win/draw rates, best-minus-current blitz gap, bullet/rapid ratings, FIDE rating, puzzle-rush best, tactics peak.

**Caveat documented up front:** these endpoints return a *current snapshot* (June 2026), not values as of game time. Slow-moving fields (account age, career volume, title) are safe; `last.rating`-type fields are mildly stale, which is why the per-game lagged Elo remains the primary rating signal. In a production system you'd snapshot these at event time.

Missingness is real and informative (FIDE 51%, rapid 22%, bullet 12% of game-sides): left as NaN in the parquet so tree models can split on it natively; impute + indicator for linear models.

### 2.4 What the EDA says (see `eda.ipynb`)

- Class balance: **48.1% win / 44.1% loss / 7.8% draw**. White scores 52.0% overall — the first-move advantage is visible and worth an intercept.
- The empirical mean-score-vs-$d$ curve tracks the Elo logistic closely over ±800 Elo.
- Draw rate decays with $|d|$; rows are non-i.i.d. across rounds (Swiss pairing shrinks $|d|$ as the field sorts).
- Baselines (train = Feb, test = Mar, multiclass log loss): **marginal 0.907 → Elo-only 0.780 → full-feature multinomial LR 0.780**. Ratings carry nearly all the signal; everything else buys calibration and marginal refinement. This is the honest headline, and it is what you'd expect: Elo *is* a fitted outcome-prediction model, maintained online over millions of games. Beating it with two tournaments of data is intrinsically hard.

## 3. Recommended split

**Train on Feb 10, test on Mar 10** — a between-tournament temporal split.

Rationale: (i) it matches deployment — you predict a *future* event; (ii) within-tournament rows are strongly dependent (same players ~11 times, Swiss pairing couples rows; history features are deterministic functions of earlier rows), so a random row split leaks and overstates performance; (iii) player overlap across events is fine — the player population persists in deployment too; what changes is the event, which is exactly what the split holds out. For hyperparameter tuning inside the training event, use `GroupKFold` grouped by player (or round-blocked CV) to respect the dependence structure.

## 4. Modeling recommendations

Throughout, let $x \in \mathbb{R}^p$ be the feature vector and $Y \in \{\text{loss} \prec \text{draw} \prec \text{win}\}$.

### 4.1 Baselines (non-negotiable)

Marginal frequencies and the Elo-only model above. Any proposed model must beat **0.780** test log loss; the marginal baseline (0.907) catches gross miscalibration.

### 4.2 Ordinal (proportional-odds) logistic regression — recommended primary

$$P(Y \le k \mid x) = \sigma(\tau_k - \beta^\top x), \qquad \tau_{\text{loss}} < \tau_{\text{draw}}$$

The latent-variable view is exactly the right generative story: a latent "game advantage" $Z = \beta^\top x + \varepsilon$, $\varepsilon \sim \text{Logistic}$, with the outcome determined by which band $(-\infty,\tau_1], (\tau_1,\tau_2], (\tau_2,\infty)$ it falls in. The draw is a *band* around equality — matching the EDA's unimodal draw structure — and the model is Thurstone–Mosteller/Elo-consistent: with $x = d/400 \cdot \ln 10$ alone it nests the Elo curve. It spends $p + 2$ parameters versus the multinomial's $2p+2$, which matters at $n \approx 2{,}000$ training rows, and the single $\beta$ is directly interpretable as "what moves the advantage scale".

### 4.3 Multinomial logistic — as an ablation

Fit $P(Y=k|x) \propto e^{\beta_k^\top x}$ and compare to 4.2. If proportional odds doesn't lose log loss, the ordinality assumption is empirically supported (the parallel-slopes assumption is the thing to check; a draw class may genuinely want its own slope on $|d|$).

### 4.4 Gradient-boosted trees (LightGBM / XGBoost, multiclass)

Nonlinearity and interactions (round × rating gap, form × strength-of-schedule) for free, native NaN handling (FIDE 51% missing), strong at this data size with heavy regularization (shallow trees, small learning rate, early stopping on grouped CV). Expect at best a small log-loss gain over 4.2 given how dominant `diff_elo` is — worth running mostly as an upper-bound probe on "is there nonlinear signal the linear model misses?"

### 4.5 Bayesian: Bradley–Terry–Davidson with skill priors from Glicko

The structurally honest model. Each player has latent skill $\theta_i$; Davidson's (1970) extension handles draws:

$$P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j + \nu\sqrt{\pi_i \pi_j}}, \qquad P(\text{draw}) = \frac{\nu\sqrt{\pi_i \pi_j}}{\pi_i + \pi_j + \nu\sqrt{\pi_i \pi_j}}$$

with $\pi_i = e^{\theta_i + \gamma \cdot \mathbb{1}[i = \text{white}]}$ ($\gamma$ = first-move advantage, $\nu$ = draw propensity, both estimable; $\nu$ can be regressed on `mean_elo`). The key move given this dataset: **don't learn $\theta_i$ from scratch** — 700 players × ~11 games each is exactly the regime where per-player MLE overfits. Instead use the Glicko output as an informative prior, $\theta_i \sim \mathcal{N}\big(c \cdot R_i,\; (c \cdot RD_i)^2\big)$: the blitz RD we collected *is* the posterior standard deviation of Chess.com's own skill model, so this is principled measurement-error modeling, not a convenience prior. Partial pooling then shrinks low-data players toward their rating while letting high-volume players' tournament results speak. Fits in minutes in PyMC/NumPyro; yields full predictive distributions and calibrated uncertainty. This is the version I'd present as the "if I had a second iteration" model.

### 4.6 Ensemble & calibration

Stack 4.2 + 4.4 (logistic meta-learner on out-of-fold probabilities) or simply average; apply temperature scaling / Dirichlet calibration fitted on held-out folds. With log loss as the metric, calibration *is* performance — expect ensembling gains of the same order as model-choice gains here.

### 4.7 What I would *not* do

Deep learning on 4k tabular rows with one dominant feature — no inductive bias to exploit, and the sequence models that genuinely shine on chess (move-sequence transformers) consume post-game information by definition.

### Evaluation protocol

Multiclass **log loss** as the primary metric (proper scoring rule; rewards calibrated probabilities, which is what a pre-game predictor is *for*), Brier score as a bounded companion, per-class recall (draws are the hard 7.8% class — accuracy alone would just reward never predicting them), and reliability diagrams per class. Always reported against both baselines.

## 5. With more time

1. **More events** — the single biggest lever. Dozens of Titled Tuesdays exist; the pipeline already generalizes (it's a list of tournament IDs).
2. **As-of-time player stats** via the monthly games archives (`/pub/player/{u}/games/{YYYY}/{MM}`): true pre-event form, fatigue (games played that day), head-to-head records.
3. Fit the hierarchical Davidson model (4.5) properly, with posterior predictive checks on the draw rate.
4. Round-stratified evaluation (round 1 is easy, round 11 is hard — a single aggregate number hides this).
5. Per-player color tendencies (some players overperform with White) once there's enough data per player to estimate them.
