"""
Hierarchical Dirichlet-Multinomial baseline for chess outcome prediction
(loss / draw / win), with partial pooling across feature-defined strata.

----------------------------------------------------------------------
Model
----------------------------------------------------------------------
    p_s | alpha, m  ~ Dirichlet(alpha * m)        (shared prior, all strata)
    n_s | p_s       ~ Multinomial(N_s, p_s)       (observed counts in stratum s)

    =>  p_s | data  ~ Dirichlet(alpha * m + n_s)   <-- EXACT, closed form

(alpha, m) — the prior's total concentration and its mean simplex — are
fit by empirical Bayes: maximize the Dirichlet-Multinomial marginal
likelihood, integrating p_s out analytically. No sampling anywhere.

----------------------------------------------------------------------
Stratification
----------------------------------------------------------------------
Per the earlier feature analysis, the win/loss axis and the draw axis
are driven by largely disjoint features:
  - white_elo_expected  -> win/loss axis
  - mean_elo            -> draw axis (stronger players draw more)
  - abs_diff_elo        -> draw axis (closer matches draw more)

Strata are formed as the cross-product of quantile bins on these three
features. With ~4000 games and 5 x 3 x 2 = 30 strata, average stratum
size is ~130 -- enough for the empirical counts to carry real signal,
while the shared prior pools across strata for the inevitable sparse
cells (especially draw-heavy ones).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, softmax

OUTCOMES = ["loss", "draw", "win"]  # fixed ordering used everywhere (also the natural ordinal order)


# ----------------------------------------------------------------------
# 1. Stratify games
# ----------------------------------------------------------------------
def assign_strata(df, expected_edges=None, elo_edges=None, mismatch_edges=None,
                   n_expected_bins=5, n_elo_bins=3, n_mismatch_bins=2):
    """
    Bin games on the two axes:
      - white_elo_expected -> win/loss axis
      - mean_elo, abs_diff_elo -> draw-propensity axis

    If *_edges are provided, reuse them (so test data is binned with
    train-derived cutpoints). Otherwise compute quantile-based edges
    from this dataframe.

    Returns (df_with_stratum_column, edges_dict).
    """
    out = df.copy()

    if expected_edges is None:
        _, expected_edges = pd.qcut(df["white_elo_expected"], n_expected_bins,
                                     retbins=True, duplicates="drop")
    if elo_edges is None:
        _, elo_edges = pd.qcut(df["mean_elo"], n_elo_bins,
                                retbins=True, duplicates="drop")
    if mismatch_edges is None:
        _, mismatch_edges = pd.qcut(df["abs_diff_elo"], n_mismatch_bins,
                                     retbins=True, duplicates="drop")

    # Open up the outer edges so out-of-range test values still get binned
    expected_edges = np.asarray(expected_edges, dtype=float).copy()
    elo_edges = np.asarray(elo_edges, dtype=float).copy()
    mismatch_edges = np.asarray(mismatch_edges, dtype=float).copy()
    expected_edges[0], expected_edges[-1] = -np.inf, np.inf
    elo_edges[0], elo_edges[-1] = -np.inf, np.inf
    mismatch_edges[0], mismatch_edges[-1] = -np.inf, np.inf

    e_bin = pd.cut(out["white_elo_expected"], expected_edges, labels=False, include_lowest=True)
    m_bin = pd.cut(out["mean_elo"], elo_edges, labels=False, include_lowest=True)
    d_bin = pd.cut(out["abs_diff_elo"], mismatch_edges, labels=False, include_lowest=True)

    out["stratum"] = (e_bin.astype(str) + "_" + m_bin.astype(str) + "_" + d_bin.astype(str))

    edges = dict(expected_edges=expected_edges, elo_edges=elo_edges, mismatch_edges=mismatch_edges)
    return out, edges


# ----------------------------------------------------------------------
# 2. Count table
# ----------------------------------------------------------------------
def build_count_table(df):
    """Return (strata_labels, counts[n_strata x 3]) with columns ordered as OUTCOMES."""
    counts = (df.groupby("stratum")["outcome"]
                .value_counts()
                .unstack(fill_value=0)
                .reindex(columns=OUTCOMES, fill_value=0))
    return counts.index.tolist(), counts.values.astype(float)


# ----------------------------------------------------------------------
# 3. Empirical-Bayes fit of (alpha, m) via the Dirichlet-Multinomial
#    marginal likelihood (Polya distribution)
# ----------------------------------------------------------------------
def dirichlet_multinomial_loglik(alpha, m, counts):
    """
    Sum over strata of log P(n_s | alpha, m), where
        n_s ~ DirichletMultinomial(N_s, alpha * m).

    counts: [n_strata x K]
    """
    a = alpha * m              # concentration vector, shape (K,)
    N = counts.sum(axis=1)      # total games per stratum
    A = a.sum()                  # == alpha

    term1 = gammaln(N + 1) - gammaln(counts + 1).sum(axis=1)
    term2 = gammaln(A) - gammaln(A + N)
    term3 = (gammaln(a[None, :] + counts) - gammaln(a[None, :])).sum(axis=1)
    return (term1 + term2 + term3).sum()


def fit_hyperparameters(counts, m_init=None, alpha_init=10.0):
    """
    Maximize the Dirichlet-Multinomial marginal log-likelihood over
    alpha > 0 (total prior "pseudo-games") and m on the simplex
    (prior mean outcome distribution).

    Reparametrization for unconstrained optimization:
        alpha = exp(log_alpha)
        m     = softmax([z_1, ..., z_{K-1}, 0])
    """
    K = counts.shape[1]
    if m_init is None:
        m_init = counts.sum(axis=0) / counts.sum()  # global outcome frequencies

    z_init = np.log(np.clip(m_init[:-1], 1e-6, None) / np.clip(m_init[-1], 1e-6, None))
    x0 = np.concatenate([[np.log(alpha_init)], z_init])

    def neg_loglik(x):
        log_alpha = x[0]
        z = np.concatenate([x[1:], [0.0]])
        m = softmax(z)
        return -dirichlet_multinomial_loglik(np.exp(log_alpha), m, counts)

    res = minimize(neg_loglik, x0, method="L-BFGS-B")
    log_alpha = res.x[0]
    z = np.concatenate([res.x[1:], [0.0]])
    m = softmax(z)
    alpha = float(np.exp(log_alpha))
    return alpha, m, res


# ----------------------------------------------------------------------
# 4. Posterior predictive per stratum (closed form)
# ----------------------------------------------------------------------
def posterior_predictive(counts, alpha, m):
    """
    For each stratum s:
        p_s | data ~ Dirichlet(alpha*m + n_s)
        E[p_s | data] = (alpha*m + n_s) / (alpha + N_s)

    Returns (posterior_mean [n_strata x K], posterior_concentration [n_strata x K]).
    The concentration vector fully characterizes the posterior -- use it
    to get credible intervals via the Dirichlet marginals (Beta distributions).
    """
    a = alpha * m
    N = counts.sum(axis=1, keepdims=True)
    posterior_alpha = a[None, :] + counts
    p_hat = posterior_alpha / (alpha + N)
    return p_hat, posterior_alpha


def credible_interval(posterior_alpha_row, k, level=0.90):
    """
    90% (or other level) credible interval for outcome k in a given
    stratum, from the Beta marginal of the Dirichlet posterior.
    """
    from scipy.stats import beta
    a_k = posterior_alpha_row[k]
    a_rest = posterior_alpha_row.sum() - a_k
    lo, hi = beta.ppf([(1 - level) / 2, 1 - (1 - level) / 2], a_k, a_rest)
    return lo, hi


# ----------------------------------------------------------------------
# 5. Predict for new games
# ----------------------------------------------------------------------
def predict(new_df, edges, p_hat_lookup, m):
    """
    new_df: dataframe with white_elo_expected, mean_elo, abs_diff_elo
    edges:  bin edges from training (output of assign_strata on train set)
    p_hat_lookup: dict {stratum_label: posterior_mean_vector}
    m: global prior mean -- fallback for strata never seen in training

    Returns (probs [n x 3], stratum_labels).
    """
    binned, _ = assign_strata(new_df,
                               expected_edges=edges["expected_edges"],
                               elo_edges=edges["elo_edges"],
                               mismatch_edges=edges["mismatch_edges"])
    probs = np.array([p_hat_lookup.get(s, m) for s in binned["stratum"]])
    return probs, binned["stratum"]


# ----------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------
def rps(probs, true_outcome_idx):
    """
    Ranked Probability Score for the ordered outcome loss < draw < win.
    probs: [n x 3], true_outcome_idx: [n] in {0,1,2}.
    Lower is better; in [0, 1] for K=3.
    """
    K = probs.shape[1]
    cum_probs = np.cumsum(probs, axis=1)
    true_onehot = np.eye(K)[true_outcome_idx]
    cum_true = np.cumsum(true_onehot, axis=1)
    return ((cum_probs - cum_true) ** 2).sum(axis=1) / (K - 1)


def multiclass_log_loss(probs, true_outcome_idx, eps=1e-12):
    p = np.clip(probs[np.arange(len(true_outcome_idx)), true_outcome_idx], eps, 1)
    return -np.log(p)


# ----------------------------------------------------------------------
# 7. End-to-end example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet("data/processed/games.parquet")

    # Temporal split per the data dictionary's recommended setup:
    # train on the February event, test on March.
    train = df[df["tournament"].str.contains("february")].copy()
    test = df[df["tournament"].str.contains("march")].copy()

    # --- fit on training data ---
    train_binned, edges = assign_strata(train, n_expected_bins=5, n_elo_bins=3, n_mismatch_bins=2)
    strata_labels, counts = build_count_table(train_binned)

    alpha, m, opt_res = fit_hyperparameters(counts)
    print(f"Fitted alpha (prior strength, in 'pseudo-games') = {alpha:.2f}")
    print(f"Fitted m     (prior outcome distribution)        = "
          f"{dict(zip(OUTCOMES, np.round(m, 3)))}")
    print(f"n strata = {len(strata_labels)}, "
          f"mean stratum size = {counts.sum(axis=1).mean():.1f}, "
          f"min stratum size = {counts.sum(axis=1).min():.0f}")

    p_hat, post_alpha = posterior_predictive(counts, alpha, m)
    p_hat_lookup = dict(zip(strata_labels, p_hat))

    # Example: show a few strata with their posterior mean + 90% CI on P(draw)
    print("\nExample strata (posterior mean, 90% CI for P(draw)):")
    draw_idx = OUTCOMES.index("draw")
    for s, mean_row, post_row in list(zip(strata_labels, p_hat, post_alpha))[:5]:
        lo, hi = credible_interval(post_row, draw_idx)
        print(f"  stratum {s}: P(draw) = {mean_row[draw_idx]:.3f}  "
              f"[{lo:.3f}, {hi:.3f}]   (n={post_row.sum() - alpha:.0f})")

    # --- evaluate on held-out March tournament ---
    test_probs, test_strata = predict(test, edges, p_hat_lookup, m)
    y_true = test["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).values

    print(f"\nHeld-out RPS:      {rps(test_probs, y_true).mean():.4f}")
    print(f"Held-out log loss: {multiclass_log_loss(test_probs, y_true).mean():.4f}")

    # naive baseline: global training-set frequencies for everyone
    global_probs = np.tile(
        train["outcome"].value_counts(normalize=True).reindex(OUTCOMES).values,
        (len(test), 1)
    )
    print(f"\nGlobal-frequency baseline RPS:      {rps(global_probs, y_true).mean():.4f}")
    print(f"Global-frequency baseline log loss: {multiclass_log_loss(global_probs, y_true).mean():.4f}")
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = test_probs.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(pd.DataFrame(cm,
                        index=[f"true_{o}" for o in OUTCOMES],
                        columns=[f"pred_{o}" for o in OUTCOMES]))
    print(classification_report(y_true, y_pred, target_names=OUTCOMES, zero_division=0))
    from sklearn.metrics import roc_auc_score

    is_draw_true = (y_true == OUTCOMES.index("draw")).astype(int)
    p_draw = test_probs[:, OUTCOMES.index("draw")]
    print(f"AUC for P(draw) discriminating draw vs. non-draw: {roc_auc_score(is_draw_true, p_draw):.3f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    draw_features = train[[
        "mean_elo", "abs_diff_elo",
        "white_elo_expected"
    ]].copy()
    draw_features["mean_blitz_draw_rate"] = (train["white_blitz_draw_rate"] + train["black_blitz_draw_rate"]) / 2
    draw_features["mean_title_ordinal"] = (train["white_title_ordinal"].fillna(0) + train["black_title_ordinal"].fillna(0)) / 2
    draw_features["mean_blitz_rd"] = (train["white_blitz_rd"] + train["black_blitz_rd"]) / 2

    is_draw = (train["outcome"] == "draw").astype(int)

    X = StandardScaler().fit_transform(draw_features.fillna(draw_features.median()))
    clf = LogisticRegression(max_iter=1000).fit(X, is_draw)
    print(f"Train AUC: {roc_auc_score(is_draw, clf.predict_proba(X)[:, 1]):.3f}")
    print(dict(zip(draw_features.columns, clf.coef_[0].round(3))))