# Data Dictionary — `data/processed/games.{parquet,csv}`

One row per game from the two Titled Tuesday blitz events (Feb 10 & Mar 10, 2026),
produced by [build_dataset.py](build_dataset.py). **4,048 rows × 96 columns**; 700 unique
players. All ratings are on the Chess.com blitz (Glicko) scale unless noted.

## Conventions

- **Perspective.** Every game is recorded from **White's** point of view. The label
  `outcome` and `outcome_score` are White-relative.
- **Side prefixes.** `white_*` and `black_*` give the corresponding player's value;
  `diff_* = white_* − black_*`.
- **Leakage tiers** (the critical column for modeling — see [ANALYSIS.md](ANALYSIS.md) §1–2):

  | tier | meaning | use as input feature? |
  |---|---|---|
  | **FEATURE** | known strictly before the game starts | ✅ yes |
  | **LABEL** | the target, or used to derive it | ❌ never |
  | **POST** (`pg_` prefix, raw result codes) | describes the game itself | ❌ not raw; ✅ only as a *lagged* per-player aggregate |
  | **ID** | identifier / metadata | ⚠️ for joins & grouped splits, not as a feature |

---

## Identifiers & metadata

| column | type | tier | description |
|---|---|---|---|
| `game_url` | str | ID | Chess.com game URL; unique row key. |
| `tournament` | str | ID | Tournament slug (`...february-10-2026-...` / `...march-10-2026-...`); the temporal-split key. |
| `round` | int | FEATURE | Swiss round number (1–11). Known pre-game; also a history feature. |
| `end_time` | int | ID/POST | Unix epoch seconds when the game **ended**. Post-game; used only to order games and compute account age. |
| `white_username` / `black_username` | str | ID | Lowercased Chess.com handles; player keys for grouped CV and joins. |

## Label

| column | type | tier | description |
|---|---|---|---|
| `outcome` | str | **LABEL** | Target: `win` / `draw` / `loss` (White perspective). |
| `outcome_score` | float | **LABEL** | Numeric companion: `1.0` / `0.5` / `0.0`. |
| `white_result_code` / `black_result_code` | str | **LABEL** | Raw API result codes (`win`, `resigned`, `timeout`, `agreed`, …) used to derive `outcome`. Retained for transparency; never a feature. |

## Post-game behavioral stats (`pg_` prefix)

Parsed from PGN movetext / `[%clk]` annotations. **They describe the game in their own row**,
so they are post-game and must only be consumed as *lagged* per-player aggregates (as the
`beh_*` features in [filtering_model.ipynb](filtering_model.ipynb) do). 5-minute control,
no increment.

| column | type | description |
|---|---|---|
| `pg_n_moves` | float | Full moves played (≈ White plies). NaN if the PGN had no clock tags. |
| `pg_white_mean_move_time` / `pg_black_mean_move_time` | float | Mean seconds per move for that side. |
| `pg_white_min_clock` / `pg_black_min_clock` | float | Lowest clock value (s) that side reached — a time-pressure marker. |
| `pg_white_time_trouble_frac` / `pg_black_time_trouble_frac` | float | Fraction of that side's moves played with < 10 s on the clock. |
| `pg_ended_by_time` | int (0/1) | Game ended on a flag (either side timed out). |
| `pg_white_resigned` / `pg_black_resigned` | int (0/1) | That side resigned. |

## Pre-game ratings (leakage-corrected)

The per-game reported rating is empirically **post-game** (audit in
[build_dataset.py](build_dataset.py) `audit_rating_field`), so the usable Elo is reconstructed
by lagging each player's reported rating one round within the event.

| column | type | tier | description |
|---|---|---|---|
| `white_rating_reported` / `black_rating_reported` | int | **POST** | Raw rating from the API game record (post-game; do not use raw). |
| `white_elo` / `black_elo` | float | FEATURE | Leakage-free pre-game Elo (lagged; round-1 falls back to reported). The primary skill signal. |
| `mean_elo` | float | FEATURE | `(white_elo + black_elo) / 2` — pairing strength (drives draw rate). |
| `abs_diff_elo` | float | FEATURE | `|white_elo − black_elo|` — mismatch magnitude. |
| `white_elo_expected` | float | FEATURE | Elo expected score for White, $\big(1+10^{-\text{diff\_elo}/400}\big)^{-1}$. |

## In-tournament history (strictly prior rounds of the same event)

Running aggregates computed by a forward pass; **round-1 values are 0 or NaN by construction**.
Available for both `white_*` and `black_*`.

| column (per side) | type | tier | description |
|---|---|---|---|
| `{side}_games_so_far` | int | FEATURE | Games already played this event. |
| `{side}_points_so_far` | float | FEATURE | Accumulated score (win=1, draw=0.5). |
| `{side}_score_rate` | float | FEATURE | `points_so_far / games_so_far` (NaN at round 1). |
| `{side}_wins_so_far` / `{side}_draws_so_far` / `{side}_losses_so_far` | int | FEATURE | Result tallies so far. |
| `{side}_streak` | int | FEATURE | Signed current streak (+ wins, − losses, 0 after a draw). |
| `{side}_perf_vs_expected` | float | FEATURE | Realized minus Elo-expected points so far — a form residual orthogonal to rating. |
| `{side}_white_games_so_far` | int | FEATURE | How many of the prior games were with White (color balance). |
| `{side}_avg_opp_elo_so_far` | float | FEATURE | Mean pre-game Elo of prior opponents (strength of schedule). |

## Player career & account features (profile + stats endpoints)

A **current snapshot** (≈ June 2026), not as-of-game-time — slow-moving fields (title, account
age, career volume) are safe; rating snapshots are mildly stale (see [ANALYSIS.md](ANALYSIS.md)
§2.3). Available for both sides. Missingness is real and informative (e.g. FIDE ~51% null).

| column (per side) | type | tier | description |
|---|---|---|---|
| `{side}_title` | str | FEATURE | Chess.com title (`GM`, `IM`, `FM`, `WGM`, …); null = untitled. |
| `{side}_title_ordinal` | float | FEATURE | Ordinal encoding (`GM`=7 … `WNM`=1); null = untitled. |
| `{side}_log_followers` | float | FEATURE | `log1p` of follower count. |
| `{side}_is_streamer` | bool | FEATURE | Chess.com streamer flag. |
| `{side}_league` | str | FEATURE | Current league tier (Legend, Champion, …). |
| `{side}_country` | str | FEATURE | ISO-style country code. |
| `{side}_blitz_rd` | float | FEATURE | Blitz Glicko rating deviation — skill *uncertainty* (used as a measurement-error prior in the Bayesian models). |
| `{side}_log_blitz_n_games` | float | FEATURE | `log1p` of career blitz games. |
| `{side}_blitz_win_rate` / `{side}_blitz_draw_rate` | float | FEATURE | Career blitz win / draw fractions. |
| `{side}_blitz_best_minus_current` | float | FEATURE | Peak − current blitz rating (rust / trajectory proxy). |
| `{side}_bullet_rating` / `{side}_rapid_rating` | float | FEATURE | Current bullet / rapid ratings (null if unrated). |
| `{side}_fide` | float | FEATURE | FIDE rating (~51% null). |
| `{side}_puzzle_rush_best` | float | FEATURE | Best Puzzle Rush score. |
| `{side}_tactics_highest` | float | FEATURE | Peak tactics (puzzles) rating. |
| `{side}_account_age_days` | float | FEATURE | Days between account creation and `end_time`. |

## Pairwise difference features (`diff_*`)

White minus Black for the bases below (all **FEATURE**-tier when their components are):

`diff_elo`, `diff_points_so_far`, `diff_score_rate`, `diff_streak`,
`diff_perf_vs_expected`, `diff_avg_opp_elo_so_far`, `diff_title_ordinal`,
`diff_account_age_days`, `diff_blitz_rd`, `diff_log_blitz_n_games`,
`diff_blitz_win_rate`, `diff_bullet_rating`, `diff_rapid_rating`, `diff_fide`,
`diff_puzzle_rush_best`.

> `diff_elo` is the single dominant predictor; the win–loss axis is monotone in it,
> while the draw probability is unimodal around `diff_elo = 0` (hence `abs_diff_elo`,
> `mean_elo`).

---

## Notes for modeling

- **Recommended split:** train = February, test = March (between-tournament temporal split).
  Within-tournament rows are dependent (same players, Swiss pairing, history features are
  deterministic functions of earlier rows), so random row splits leak.
- **Excluded by design** (post-game, never persisted as features): PGN movetext, final FEN,
  `Termination`, game duration, and the ECO opening code (determined by moves played).
- **Derived feature families** built downstream from this table: leakage-corrected lagged
  skills, behavioral `beh_*` aggregates ([filtering_model.ipynb](filtering_model.ipynb)),
  and Glicko-informed priors ([bayesian_model.ipynb](bayesian_model.ipynb)).
