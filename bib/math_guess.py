# perfect_player.py
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional
from collections import defaultdict, Counter
import math, os, csv
from concurrent.futures import ProcessPoolExecutor, as_completed

from hint import hint  # your canonical hint()

# ---------- small utils ----------
def load_words(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def hint_key(guess: str, target: str) -> str:
    return " ".join(hint(guess, target))

def entropy_from_counts(counts: Iterable[int]) -> float:
    N = sum(counts)
    H = 0.0
    for c in counts:
        p = c / N
        H -= p * math.log2(p)
    return H

def summarize_buckets(buckets: Dict[str,int]) -> Dict[str, float | int]:
    counts = list(buckets.values())
    counts.sort(reverse=True)
    N = sum(counts)
    H = entropy_from_counts(counts)
    worst = counts[0]
    num_buckets = len(counts)
    num_singletons = sum(1 for x in counts if x == 1)
    perplex = 2.0 ** H
    return {
        "entropy_bits": H,
        "worst_bucket_size": worst,
        "worst_bucket_frac": worst / N if N else 0.0,
        "num_buckets": num_buckets,
        "num_singletons": num_singletons,
        "perplexity": perplex,
        "avg_bucket_size": (N / num_buckets) if num_buckets else 0.0,
    }

# ---------- heuristic prefilter for turn 1 ----------
def score_word_heuristic(w: str, letter_freq: Dict[str,int], bigram_freq: Dict[str,int]) -> float:
    uniq = set(w)  # letter coverage, not multiplicity
    s_letters = sum(letter_freq.get(ch, 0) for ch in uniq)
    bigs = {w[i:i+2] for i in range(len(w)-1)}
    s_bigrams = sum(bigram_freq.get(bg, 0) for bg in bigs)
    # contiguous signals matter in your rules → weight bigrams more
    return 1.0 * s_letters + 1.8 * s_bigrams

def prefilter_turn1(words: List[str], keep: int = 1500) -> List[str]:
    lf = Counter()
    bf = Counter()
    for w in words:
        lf.update(set(w))
        bf.update({w[i:i+2] for i in range(len(w)-1)})
    scored = [(score_word_heuristic(w, lf, bf), w) for w in words]
    scored.sort(reverse=True)
    return [w for _, w in scored[:keep]]

# ---------- worker globals for multiprocessing ----------
_PP_CANDIDATES: List[str] = []

def _pp_init_candidates(candidates: List[str]) -> None:
    # Set once per worker
    global _PP_CANDIDATES
    _PP_CANDIDATES = candidates

def _pp_summarize_guess(g: str) -> Tuple[str, Dict[str, float | int]]:
    """Worker: compute buckets and metrics for guess g against global candidates."""
    buckets: Dict[str,int] = defaultdict(int)
    # alias for speed
    hk = hint_key
    for t in _PP_CANDIDATES:
        buckets[hk(g, t)] += 1
    m = summarize_buckets(buckets)
    return g, m

# ---------- parallel entropy for a given candidate set ----------
def best_entropy_guesses(
    guess_space: List[str],
    candidates: List[str],
    top_n: int = 5,
    workers: Optional[int] = None
) -> List[Dict[str, float | int | str]]:
    """
    Evaluate all guesses in guess_space against 'candidates', return top_n guesses
    by entropy with tie-breakers: max entropy → min worst bucket → lexicographic.
    """
    if not guess_space:
        return []
    if len(guess_space) == 1:
        g = guess_space[0]
        # run single-threaded to avoid spinning up pool needlessly
        buckets: Dict[str,int] = defaultdict(int)
        hk = hint_key
        for t in candidates:
            buckets[hk(g, t)] += 1
        m = summarize_buckets(buckets)
        m["guess"] = g
        return [m]

    out: List[Tuple[str, Dict[str, float | int]]] = []

    with ProcessPoolExecutor(max_workers=workers, initializer=_pp_init_candidates, initargs=(candidates,)) as ex:
        futures = {ex.submit(_pp_summarize_guess, g): g for g in guess_space}
        for fut in as_completed(futures):
            g, m = fut.result()
            out.append((g, m))

    # rank with deterministic tie-breakers
    ranked = sorted(
        ({"guess": g, **m} for g, m in out),
        key=lambda r: (-float(r["entropy_bits"]), int(r["worst_bucket_size"]), str(r["guess"]))
    )
    return ranked[:top_n]

# ---------- read top-entropy from CSV (fast first turn) ----------
def topN_from_metrics_csv(path: str, N: int = 5) -> Optional[List[Dict[str, float | int | str]]]:
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append({
                "guess": row["guess"],
                "entropy_bits": float(row["entropy_bits"]),
                "num_buckets": int(row["num_buckets"]),
                "num_singletons": int(row["num_singletons"]),
                "worst_bucket_size": int(row["worst_bucket_size"]),
                "worst_bucket_frac": float(row["worst_bucket_frac"]),
                "perplexity": float(row["perplexity"]),
                "avg_bucket_size": float(row["avg_bucket_size"]),
            })
    # Sort just in case the CSV isn’t pre-sorted
    rows.sort(key=lambda r: (-r["entropy_bits"], r["worst_bucket_size"], r["guess"]))
    return rows[:N]

# ---------- exported API ----------
def math_choose5(
    words_path: str,
    state: List[Tuple[str, List[str]]],
    allow_non_candidates: bool = True,
    limit_guesses: int = 0,
    workers: Optional[int] = None,
    metrics_csv: str = "per_guess_metrics.csv",
    top_n: int = 5,
) -> List[Dict[str, float | int | str]]:
    """
    Return the top-N next guesses (with metrics) that maximize expected information (entropy)
    for the current state.

    Args:
      words_path: path to words.txt
      state: list of (guess_str, hint_tokens_list)
      allow_non_candidates: if True, may guess non-answer probes
      limit_guesses: cap guesses evaluated (after prefilter for empty state); 0 = no cap
      workers: number of processes; None = auto
      metrics_csv: if present and state is empty, use its top-N (O(1) fast-path)
      top_n: how many suggestions to return (default 5)

    Returns:
      List of dicts, each with keys:
        'guess', 'entropy_bits', 'worst_bucket_size', 'worst_bucket_frac',
        'num_buckets', 'num_singletons', 'perplexity', 'avg_bucket_size'
    """
    words = load_words(words_path)

    # Build current candidate set by filtering history
    def key_obs(tokens: List[str]) -> str:
        return " ".join(tokens)

    candidates = list(words)
    for g_prev, tokens in state:
        k = key_obs(tokens)
        candidates = [t for t in candidates if hint_key(g_prev, t) == k]

    if not candidates:
        raise ValueError("No candidates remain consistent with the provided state.")

    # If singleton remains, return it as the only suggestion
    if len(candidates) == 1:
        return [{
            "guess": candidates[0],
            "entropy_bits": 0.0,
            "num_buckets": 1,
            "num_singletons": 1,
            "worst_bucket_size": 1,
            "worst_bucket_frac": 1.0,
            "perplexity": 1.0,
            "avg_bucket_size": 1.0,
        }]

    # Decide the guess space
    guess_space = words if allow_non_candidates else candidates

    # FAST PATH for the very first turn
    if not state:
        # 1) CSV shortcut if available
        top_rows = topN_from_metrics_csv(metrics_csv, N=top_n)
        if top_rows is not None:
            # If restricted to candidates and the CSV best isn't in candidates (rare), recompute.
            if not allow_non_candidates:
                top_rows = [r for r in top_rows if r["guess"] in candidates]
                if len(top_rows) >= top_n:
                    return top_rows[:top_n]
            else:
                return top_rows

        # 2) Otherwise, prefilter to avoid 10k×10k and compute
        keep = 1500 if limit_guesses == 0 else min(limit_guesses, len(guess_space))
        pre = prefilter_turn1(guess_space, keep=keep)
        return best_entropy_guesses(pre, candidates, top_n=top_n, workers=workers)

    # NON-FIRST TURN: |S| has shrunk → can evaluate more broadly
    if limit_guesses and limit_guesses < len(guess_space):
        guess_space = guess_space[:limit_guesses]

    return best_entropy_guesses(guess_space, candidates, top_n=top_n, workers=workers)

# For easy backward compatibility with your previous name:
def math_choose(
    words_path: str,
    state: List[Tuple[str, List[str]]],
    allow_non_candidates: bool = True,
    limit_guesses: int = 0,
    workers: Optional[int] = None,
    metrics_csv: str = "per_guess_metrics.csv",
) -> str:
    """Return only the single best guess (string)."""
    top = math_choose5(words_path, state, allow_non_candidates, limit_guesses, workers, metrics_csv, top_n=1)
    return top[0]["guess"]
