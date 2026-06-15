"""
Build a one-row-per-game dataset from Chess.com Titled Tuesday tournaments.

Pipeline:
  1. Crawl tournament -> rounds -> groups -> games (cached on disk).
  2. Fetch player profiles and stats for every unique participant (cached).
  3. Audit whether the per-game `rating` field is pre- or post-game and
     construct leakage-free pre-game ratings accordingly.
  4. Engineer pre-game features (ratings, in-tournament history, career
     stats, account metadata) and export CSV + parquet.

Usage:
    python build_dataset.py

All HTTP responses are cached under data/raw/, so reruns are offline and
deterministic. Outputs land in data/processed/games.{csv,parquet}.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

TOURNAMENTS = [
    "titled-tuesday-blitz-february-10-2026-6221327",
    "titled-tuesday-blitz-march-10-2026-6277141",
    "titled-tuesday-blitz-march-17-2026-6282783",
    "titled-tuesday-blitz-march-24-2026-6292855",
    "titled-tuesday-blitz-march-31-2026-6322539",
    "titled-tuesday-blitz-april-07-2026-6342683",
    "titled-tuesday-blitz-april-14-2026-6362193"
]
API_BASE = "https://api.chess.com/pub"
USER_AGENT = "titled-tuesday-dataset-builder (take-home exercise; contact: candidate)"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

REQUEST_SLEEP = 0.15  # seconds between uncached requests (politeness)
MAX_RETRIES = 5

# White-perspective outcome mapping from the API result codes.
DRAW_CODES = {
    "agreed", "repetition", "stalemate", "insufficient",
    "50move", "timevsinsufficient",
}
LOSS_CODES = {"checkmated", "resigned", "timeout", "abandoned", "lose"}

TITLE_ORDINAL = {
    "GM": 7, "IM": 6, "WGM": 5, "FM": 5, "WIM": 4, "CM": 4,
    "NM": 3, "WFM": 3, "WCM": 2, "WNM": 1,
}

# --------------------------------------------------------------------------
# HTTP with on-disk cache
# --------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})


def cache_path(url: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.replace(API_BASE, "")).strip("_")[:120]
    digest = hashlib.md5(url.encode()).hexdigest()[:10]
    return RAW_DIR / f"{slug}__{digest}.json"


def fetch_json(url: str) -> dict | None:
    """GET a PubAPI URL with caching and retry/backoff. Returns None on 404."""
    path = cache_path(url)
    if path.exists():
        return json.loads(path.read_text())

    for attempt in range(MAX_RETRIES):
        resp = _session.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            time.sleep(REQUEST_SLEEP)
            return data
        if resp.status_code == 404:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("null")
            return None
        # 429 / 5xx: exponential backoff
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} retries")


# --------------------------------------------------------------------------
# Crawl
# --------------------------------------------------------------------------

MAX_ROUNDS = 30  # safety cap when probing rounds


def crawl_games(tournament_id: str) -> list[dict]:
    """Return raw game dicts annotated with tournament/round/group.

    Note: the tournament root endpoint 404s for these events, so we probe
    round endpoints sequentially until the first 404.
    """
    info = fetch_json(f"{API_BASE}/tournament/{tournament_id}")
    n_rounds = info["settings"]["total_rounds"] if info else MAX_ROUNDS
    games = []
    for rnd in range(1, n_rounds + 1):
        round_data = fetch_json(f"{API_BASE}/tournament/{tournament_id}/{rnd}")
        if round_data is None:
            break
        for group_url in round_data.get("groups", []):
            group = fetch_json(group_url)
            if group is None:
                continue
            for g in group.get("games", []):
                g["_tournament"] = tournament_id
                g["_round"] = rnd
                games.append(g)
    return games


def has_moves(pgn: str) -> bool:
    """True if at least one move was played (filters forfeits/no-shows)."""
    body = pgn.split("\n\n", 1)[-1] if pgn else ""
    return bool(re.search(r"\b1\.\s", body))


# --------------------------------------------------------------------------
# PGN movetext features (POST-game: describe the game itself, so they are
# only usable as *lagged* per-player aggregates in downstream models.
# All such columns carry the `pg_` prefix to make the leakage contract
# explicit.)
# --------------------------------------------------------------------------

_CLK_RE = re.compile(r"\{\[%clk ([0-9:.]+)\]\}")
TIME_TROUBLE_SEC = 10.0  # clock threshold defining "time trouble"


def _clk_to_seconds(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60.0 + p
    return sec


def _parse_time_control(tc: str) -> tuple[float, float]:
    """'300' -> (300, 0); '180+1' -> (180, 1)."""
    base, _, inc = (tc or "").partition("+")
    try:
        return float(base), float(inc) if inc else 0.0
    except ValueError:
        return np.nan, 0.0


def pgn_clock_features(pgn: str, time_control: str) -> dict:
    """Per-side clock-usage stats from the %clk annotations of one game.

    Move time for ply k of a side = previous clock - current clock + increment
    (clamped at 0; the first 'previous clock' is the base time).
    """
    out = {"pg_n_moves": np.nan}
    for side in ("white", "black"):
        for stat in ("mean_move_time", "min_clock", "time_trouble_frac"):
            out[f"pg_{side}_{stat}"] = np.nan

    clocks = [_clk_to_seconds(c) for c in _CLK_RE.findall(pgn or "")]
    if not clocks:
        return out
    base, inc = _parse_time_control(time_control)

    out["pg_n_moves"] = (len(clocks) + 1) // 2  # full moves = white plies
    for offset, side in ((0, "white"), (1, "black")):
        side_clocks = clocks[offset::2]
        if not side_clocks:
            continue
        prevs = [base] + side_clocks[:-1]
        move_times = [max(p - c + inc, 0.0) for p, c in zip(prevs, side_clocks)]
        out[f"pg_{side}_mean_move_time"] = float(np.mean(move_times))
        out[f"pg_{side}_min_clock"] = float(min(side_clocks))
        out[f"pg_{side}_time_trouble_frac"] = float(
            np.mean([c < TIME_TROUBLE_SEC for c in side_clocks]))
    return out


def outcome_from_result(white_result: str) -> str | None:
    if white_result == "win":
        return "win"
    if white_result in DRAW_CODES:
        return "draw"
    if white_result in LOSS_CODES:
        return "loss"
    return None


def parse_games(raw_games: list[dict]) -> pd.DataFrame:
    rows, skipped_no_moves, skipped_unknown = [], 0, 0
    for g in raw_games:
        if not has_moves(g.get("pgn", "")):
            skipped_no_moves += 1
            continue
        outcome = outcome_from_result(g["white"]["result"])
        if outcome is None:
            skipped_unknown += 1
            continue
        row = {
            "game_url": g["url"],
            "tournament": g["_tournament"],
            "round": g["_round"],
            "end_time": g["end_time"],
            "white_username": g["white"]["username"].lower(),
            "black_username": g["black"]["username"].lower(),
            "white_rating_reported": g["white"]["rating"],
            "black_rating_reported": g["black"]["rating"],
            "white_result_code": g["white"]["result"],
            "black_result_code": g["black"]["result"],
            "outcome": outcome,
        }
        # POST-game behavioral stats (pg_ prefix; lag before use as features).
        row.update(pgn_clock_features(g.get("pgn", ""), g.get("time_control", "")))
        row["pg_ended_by_time"] = int("timeout" in (g["white"]["result"],
                                                    g["black"]["result"]))
        row["pg_white_resigned"] = int(g["white"]["result"] == "resigned")
        row["pg_black_resigned"] = int(g["black"]["result"] == "resigned")
        rows.append(row)
    print(f"  parsed {len(rows)} games "
          f"(skipped {skipped_no_moves} without moves, {skipped_unknown} unknown results)")
    df = pd.DataFrame(rows)
    return df.sort_values(["tournament", "round", "game_url"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Rating audit: is the per-game `rating` field pre- or post-game?
# --------------------------------------------------------------------------

def player_round_view(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (player, game) with reported rating and score."""
    score_map = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    w = df[["tournament", "round", "white_username", "white_rating_reported", "outcome"]].copy()
    w.columns = ["tournament", "round", "username", "rating", "outcome"]
    w["score"] = w["outcome"].map(score_map)
    b = df[["tournament", "round", "black_username", "black_rating_reported", "outcome"]].copy()
    b.columns = ["tournament", "round", "username", "rating", "outcome"]
    b["score"] = 1.0 - b["outcome"].map(score_map)
    return pd.concat([w, b]).sort_values(["tournament", "username", "round"]).reset_index(drop=True)


def audit_rating_field(df: pd.DataFrame) -> str:
    """
    For consecutive rounds r -> r+1 played by the same player, the rating
    delta correlates with the score of round r if the field is PRE-game
    (the delta absorbs round r's result), and with the score of round r+1
    if it is POST-game. Returns 'pre' or 'post'.
    """
    long = player_round_view(df)
    g = long.groupby(["tournament", "username"])
    long["delta"] = g["rating"].diff()
    long["score_prev"] = g["score"].shift(1)
    long["consecutive"] = g["round"].diff() == 1
    sub = long[long["consecutive"] & long["delta"].notna()]
    corr_pre = sub["delta"].corr(sub["score_prev"])    # pre-game hypothesis
    corr_post = sub["delta"].corr(sub["score"])        # post-game hypothesis
    verdict = "pre" if corr_pre > corr_post else "post"
    print(f"  rating audit on {len(sub)} consecutive-round pairs: "
          f"corr(delta, prev result) = {corr_pre:.3f} | "
          f"corr(delta, same-round result) = {corr_post:.3f} -> field is {verdict.upper()}-game")
    return verdict


def add_pregame_ratings(df: pd.DataFrame, verdict: str) -> pd.DataFrame:
    """Attach leakage-free pre-game ratings (lag reported ratings if post-game)."""
    if verdict == "pre":
        df["white_elo"] = df["white_rating_reported"]
        df["black_elo"] = df["black_rating_reported"]
        return df

    long = player_round_view(df)
    long["pre_rating"] = long.groupby(["tournament", "username"])["rating"].shift(1)
    # Round-1 fallback: the reported value (bias is at most one game's K-factor).
    long["pre_rating"] = long["pre_rating"].fillna(long["rating"])
    key = long.set_index(["tournament", "round", "username"])["pre_rating"]
    df["white_elo"] = key.loc[
        list(zip(df["tournament"], df["round"], df["white_username"]))].to_numpy()
    df["black_elo"] = key.loc[
        list(zip(df["tournament"], df["round"], df["black_username"]))].to_numpy()
    return df


# --------------------------------------------------------------------------
# In-tournament history features (strictly prior rounds only)
# --------------------------------------------------------------------------

def elo_expected(d: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(d, dtype=float) / 400.0))


HIST_COLS = [
    "games_so_far", "points_so_far", "score_rate", "wins_so_far", "draws_so_far",
    "losses_so_far", "streak", "perf_vs_expected", "white_games_so_far",
    "avg_opp_elo_so_far",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Running per-player aggregates over earlier rounds of the same event."""
    state: dict[tuple, dict] = {}

    def blank():
        return {"games": 0, "points": 0.0, "wins": 0, "draws": 0, "losses": 0,
                "streak": 0, "perf": 0.0, "white_games": 0, "opp_elo_sum": 0.0}

    feats = {f"{side}_{c}": [] for side in ("white", "black") for c in HIST_COLS}

    df = df.sort_values(["tournament", "round", "game_url"]).reset_index(drop=True)
    for _, row in df.iterrows():
        for side, opp in (("white", "black"), ("black", "white")):
            s = state.setdefault((row["tournament"], row[f"{side}_username"]), blank())
            n = s["games"]
            feats[f"{side}_games_so_far"].append(n)
            feats[f"{side}_points_so_far"].append(s["points"])
            feats[f"{side}_score_rate"].append(s["points"] / n if n else np.nan)
            feats[f"{side}_wins_so_far"].append(s["wins"])
            feats[f"{side}_draws_so_far"].append(s["draws"])
            feats[f"{side}_losses_so_far"].append(s["losses"])
            feats[f"{side}_streak"].append(s["streak"])
            feats[f"{side}_perf_vs_expected"].append(s["perf"])
            feats[f"{side}_white_games_so_far"].append(s["white_games"])
            feats[f"{side}_avg_opp_elo_so_far"].append(
                s["opp_elo_sum"] / n if n else np.nan)

        # Update state *after* recording pre-game values.
        score_w = {"win": 1.0, "draw": 0.5, "loss": 0.0}[row["outcome"]]
        for side, opp, score in (("white", "black", score_w),
                                 ("black", "white", 1.0 - score_w)):
            s = state[(row["tournament"], row[f"{side}_username"])]
            exp = float(elo_expected(row[f"{side}_elo"] - row[f"{opp}_elo"]))
            s["games"] += 1
            s["points"] += score
            s["perf"] += score - exp
            s["opp_elo_sum"] += row[f"{opp}_elo"]
            if score == 1.0:
                s["wins"] += 1
                s["streak"] = s["streak"] + 1 if s["streak"] > 0 else 1
            elif score == 0.0:
                s["losses"] += 1
                s["streak"] = s["streak"] - 1 if s["streak"] < 0 else -1
            else:
                s["draws"] += 1
                s["streak"] = 0
            if side == "white":
                s["white_games"] += 1

    for col, values in feats.items():
        df[col] = values
    return df


# --------------------------------------------------------------------------
# Player enrichment (profiles + stats)
# --------------------------------------------------------------------------

def fetch_player_tables(usernames: list[str]) -> pd.DataFrame:
    rows = []
    for u in tqdm(sorted(usernames), desc="players"):
        profile = fetch_json(f"{API_BASE}/player/{u}") or {}
        stats = fetch_json(f"{API_BASE}/player/{u}/stats") or {}
        blitz = stats.get("chess_blitz", {})
        bl_rec = blitz.get("record", {})
        bl_n = sum(bl_rec.get(k, 0) for k in ("win", "loss", "draw"))
        rows.append({
            "username": u,
            "title": profile.get("title"),
            "joined": profile.get("joined"),
            "followers": profile.get("followers"),
            "is_streamer": profile.get("is_streamer"),
            "league": profile.get("league"),
            "country": (profile.get("country") or "").rsplit("/", 1)[-1] or None,
            "blitz_rating_current": blitz.get("last", {}).get("rating"),
            "blitz_rd": blitz.get("last", {}).get("rd"),
            "blitz_best_rating": blitz.get("best", {}).get("rating"),
            "blitz_n_games": bl_n if bl_n else None,
            "blitz_win_rate": bl_rec.get("win", 0) / bl_n if bl_n else None,
            "blitz_draw_rate": bl_rec.get("draw", 0) / bl_n if bl_n else None,
            "bullet_rating": stats.get("chess_bullet", {}).get("last", {}).get("rating"),
            "rapid_rating": stats.get("chess_rapid", {}).get("last", {}).get("rating"),
            "fide": stats.get("fide") or None,
            "puzzle_rush_best": stats.get("puzzle_rush", {}).get("best", {}).get("score"),
            "tactics_highest": stats.get("tactics", {}).get("highest", {}).get("rating"),
        })
    return pd.DataFrame(rows)


def merge_player_features(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    players["title_ordinal"] = players["title"].map(TITLE_ORDINAL)
    players["log_followers"] = np.log1p(players["followers"].astype(float))
    players["log_blitz_n_games"] = np.log1p(players["blitz_n_games"].astype(float))
    players["blitz_best_minus_current"] = (
        players["blitz_best_rating"] - players["blitz_rating_current"])

    keep = ["username", "title", "title_ordinal", "joined", "log_followers",
            "is_streamer", "league", "country", "blitz_rd", "log_blitz_n_games",
            "blitz_win_rate", "blitz_draw_rate", "blitz_best_minus_current",
            "bullet_rating", "rapid_rating", "fide", "puzzle_rush_best",
            "tactics_highest"]
    players = players[keep]

    for side in ("white", "black"):
        sided = players.add_prefix(f"{side}_")
        df = df.merge(sided, on=f"{side}_username", how="left")
        df[f"{side}_account_age_days"] = (
            (df["end_time"] - df[f"{side}_joined"]) / 86400.0)
        df = df.drop(columns=[f"{side}_joined"])
    return df


# --------------------------------------------------------------------------
# Pairwise diff features + final assembly
# --------------------------------------------------------------------------

DIFF_BASES = [
    "elo", "points_so_far", "score_rate", "streak", "perf_vs_expected",
    "avg_opp_elo_so_far", "title_ordinal", "account_age_days", "blitz_rd",
    "log_blitz_n_games", "blitz_win_rate", "bullet_rating", "rapid_rating",
    "fide", "puzzle_rush_best",
]


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    for base in DIFF_BASES:
        df[f"diff_{base}"] = df[f"white_{base}"] - df[f"black_{base}"]
    df["abs_diff_elo"] = df["diff_elo"].abs()
    df["mean_elo"] = (df["white_elo"] + df["black_elo"]) / 2.0
    df["white_elo_expected"] = elo_expected(df["diff_elo"])
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = []
    for tid in TOURNAMENTS:
        print(f"Crawling {tid} ...")
        raw.extend(crawl_games(tid))
    df = parse_games(raw)

    print("Auditing rating field ...")
    verdict = audit_rating_field(df)
    df = add_pregame_ratings(df, verdict)

    print("Computing in-tournament history features ...")
    df = add_history_features(df)

    usernames = sorted(set(df["white_username"]) | set(df["black_username"]))
    print(f"Enriching {len(usernames)} unique players ...")
    players = fetch_player_tables(usernames)
    df = merge_player_features(df, players)

    df = add_diff_features(df)

    df["outcome_score"] = df["outcome"].map({"win": 1.0, "draw": 0.5, "loss": 0.0})

    df.to_parquet(OUT_DIR / "games.parquet", index=False)
    df.to_csv(OUT_DIR / "games.csv", index=False)
    print(f"\nWrote {len(df)} rows x {df.shape[1]} cols to {OUT_DIR}/games.[parquet|csv]")
    print(df["outcome"].value_counts(normalize=True).rename("share").to_string())


if __name__ == "__main__":
    main()
