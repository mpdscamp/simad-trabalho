# poshanka!

# Requires: hint.py providing `hint(guess, target) -> List[str]`

import argparse
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------- import your canonical hint -----------
from hint import hint

# ---------- utilities ---------------------------
def load_words(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def hint_key(guess: str, target: str) -> str:
    # canonical string key for bucketing
    return " ".join(hint(guess, target))

def entropy_from_counts(counts: List[int]) -> float:
    N = sum(counts)
    H = 0.0
    for c in counts:
        p = c / N
        H -= p * math.log2(p)
    return H

@dataclass
class GuessMetrics:
    guess: str
    H: float
    num_buckets: int
    num_singletons: int
    worst_bucket_size: int
    worst_bucket_frac: float
    perplexity: float            # 2^H
    hhi: float                   # sum p^2
    eff_num_buckets_hhi: float   # 1 / sum p^2
    avg_bucket_size: float       # N / num_buckets

def bucket_counts_for_guess(g: str, S: List[str]) -> Dict[str, int]:
    buckets: Dict[str, int] = defaultdict(int)
    for t in S:
        buckets[hint_key(g, t)] += 1
    return buckets

def summarize_buckets(buckets: Dict[str, int]) -> GuessMetrics:
    counts = list(buckets.values())
    counts.sort(reverse=True)
    N = sum(counts)
    H = entropy_from_counts(counts)
    worst = counts[0]
    num_buckets = len(counts)
    num_singletons = sum(1 for c in counts if c == 1)
    perplexity = 2.0 ** H
    # concentration (Herfindahl-Hirschman Index over buckets)
    hhi = sum((c / N) ** 2 for c in counts)
    eff_num_buckets_hhi = (1.0 / hhi) if hhi > 0 else float("inf")
    avg_bucket_size = N / num_buckets
    return GuessMetrics(
        guess="",
        H=H,
        num_buckets=num_buckets,
        num_singletons=num_singletons,
        worst_bucket_size=worst,
        worst_bucket_frac=worst / N,
        perplexity=perplexity,
        hhi=hhi,
        eff_num_buckets_hhi=eff_num_buckets_hhi,
        avg_bucket_size=avg_bucket_size,
    )

def evaluate_guess(g: str, S: List[str]) -> Tuple[GuessMetrics, Dict[str, int]]:
    buckets = bucket_counts_for_guess(g, S)
    m = summarize_buckets(buckets)
    m.guess = g
    return m, buckets

def rank_guesses(words: List[str], S: List[str], limit_guesses: int = 0) -> List[GuessMetrics]:
    # Optionally restrict the evaluated guesses to speed things up
    G = words if limit_guesses <= 0 or limit_guesses >= len(words) else words[:limit_guesses]
    results: List[GuessMetrics] = []
    bestH = -1.0
    for i, g in enumerate(G, 1):
        buckets = bucket_counts_for_guess(g, S)
        m = summarize_buckets(buckets)
        m.guess = g
        results.append(m)
        if m.H > bestH:
            bestH = m.H
        if i % 250 == 0:
            print(f"[info] evaluated {i}/{len(G)} guesses; current best H={bestH:.4f}", file=sys.stderr)
    # sort by entropy desc, then by smaller worst-case bucket, then by lexicographic guess
    results.sort(key=lambda x: (-x.H, x.worst_bucket_size, x.guess))
    return results

def deep_dive_on_guess(g: str, S: List[str], top_k_buckets: int = 12, reps_per_bucket: int = 6) -> None:
    buckets = bucket_counts_for_guess(g, S)
    metrics = summarize_buckets(buckets)
    metrics.guess = g
    N = len(S)
    print("\n=== Deep dive on best guess ===")
    print(f"Guess: {g}")
    print(f"Candidates (|S|): {N}")
    print(f"Entropy H(g): {metrics.H:.6f} bits   (max possible = log2|S| = {math.log2(N):.6f})")
    print(f"Perplexity (2^H): {metrics.perplexity:.2f}")
    print(f"Number of feedback buckets: {metrics.num_buckets}")
    print(f"Singleton buckets: {metrics.num_singletons}")
    print(f"Worst-case bucket size: {metrics.worst_bucket_size}  ({100*metrics.worst_bucket_frac:.2f}%)")
    print(f"Avg bucket size: {metrics.avg_bucket_size:.2f}")
    print(f"Bucket concentration HHI: {metrics.hhi:.5f}  (effective #buckets ≈ {metrics.eff_num_buckets_hhi:.2f})")

    # Top-K buckets by size with representative targets
    print("\nTop buckets (by size):")
    # invert buckets -> targets list for sampling representatives
    key_to_targets: Dict[str, List[str]] = defaultdict(list)
    for t in S:
        key_to_targets[hint_key(g, t)].append(t)

    largest = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:top_k_buckets]
    for idx, (k, c) in enumerate(largest, 1):
        p = c / N
        reps = key_to_targets[k][:reps_per_bucket]
        # shorten long keys for readability
        k_short = k if len(k) <= 120 else (k[:117] + "...")
        print(f"{idx:2d}. count={c:4d}  p={p:7.4f}  key={k_short}")
        if reps:
            print("    e.g.:", ", ".join(reps))

def filter_by_observed(S: List[str], guess: str, observed_key: str) -> List[str]:
    # Keep targets consistent with an observed feedback for guess
    return [t for t in S if hint_key(guess, t) == observed_key]

# ---------- (Optional) simulation to estimate expected #guesses ----------
def greedy_entropy_guess(words: List[str], S: List[str]) -> GuessMetrics:
    # pick the guess with maximum entropy over current S (ties -> smaller worst-case -> lexicographic)
    ranked = rank_guesses(words, S, limit_guesses=0)
    return ranked[0]

def simulate(words: List[str], all_targets: List[str], trials: int = 200, max_steps: int = 15, seed: int = 0) -> Tuple[float, List[int]]:
    random.seed(seed)
    lengths = []
    for _ in range(trials):
        t = random.choice(all_targets)
        S = all_targets[:]  # fresh candidate set
        steps = 0
        solved = False
        while steps < max_steps and len(S) > 1:
            gm = greedy_entropy_guess(words, S)
            g = gm.guess
            obs = hint_key(g, t)
            S = filter_by_observed(S, g, obs)
            steps += 1
            if len(S) == 1:
                solved = True
                break
        # If one remains, you would guess it next turn.
        if solved:
            lengths.append(steps + 0)  # already at singleton; you may or may not count the final reveal
        else:
            lengths.append(max_steps)
    avg = sum(lengths) / len(lengths) if lengths else float("nan")
    return avg, lengths

# ---------- CSV helpers (optional) ----------
def write_csv_per_guess(path: str, metrics_list: List[GuessMetrics]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("guess,entropy_bits,num_buckets,num_singletons,worst_bucket_size,worst_bucket_frac,perplexity,hhi,eff_num_buckets_hhi,avg_bucket_size\n")
        for m in metrics_list:
            f.write(f"{m.guess},{m.H:.6f},{m.num_buckets},{m.num_singletons},{m.worst_bucket_size},{m.worst_bucket_frac:.6f},{m.perplexity:.6f},{m.hhi:.8f},{m.eff_num_buckets_hhi:.6f},{m.avg_bucket_size:.6f}\n")

def write_csv_buckets(path: str, guess: str, S: List[str]) -> None:
    buckets = bucket_counts_for_guess(guess, S)
    with open(path, "w", encoding="utf-8") as f:
        f.write("bucket_key,count\n")
        for k, c in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
            # Wrap key in quotes to preserve spaces/commas
            f.write(f"\"{k}\",{c}\n")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Rich entropy analysis for canonical-hint Wordle-like game.")
    ap.add_argument("--words", default="words.txt", help="path to words list (one per line)")
    ap.add_argument("--topk", type=int, default=20, help="print top-K guesses by entropy")
    ap.add_argument("--limit-guesses", type=int, default=0, help="limit how many guesses to evaluate (0 = all)")
    ap.add_argument("--export-per-guess", default="", help="CSV path to export per-guess metrics")
    ap.add_argument("--export-buckets", default="", help="CSV path to export bucket histogram for the best guess")
    ap.add_argument("--simulate", type=int, default=0, help="run greedy-entropy simulation with N random targets (0=skip)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for simulation")
    ap.add_argument("--deep-buckets", type=int, default=12, help="how many largest buckets to display in deep dive")
    args = ap.parse_args()

    words = load_words(args.words)
    S = words[:]  # initial candidate set
    N = len(S)
    print(f"[info] Loaded {N} words.")

    # Rank guesses by entropy (optionally restricted)
    ranked = rank_guesses(words, S, limit_guesses=args.limit_guesses)

    # Print top-K table
    K = min(args.topk, len(ranked))
    print("\n=== Top guesses by expected information (bits) ===")
    print(f"{'rank':>4}  {'guess':<20}  {'H(bits)':>9}  {'perplex.':>8}  {'#bkt':>5}  {'singl.':>6}  {'worst':>6}  {'w%':>6}")
    for i in range(K):
        r = ranked[i]
        print(f"{i+1:4d}  {r.guess:<20}  {r.H:9.6f}  {r.perplexity:8.2f}  {r.num_buckets:5d}  {r.num_singletons:6d}  {r.worst_bucket_size:6d}  {100*r.worst_bucket_frac:5.2f}%")

    # Sanity check
    Hmax = math.log2(N)
    best = ranked[0]
    print("\n=== Sanity checks ===")
    print(f"Max possible entropy log2(|S|) = {Hmax:.6f}")
    if best.H <= Hmax + 1e-9:
        print(f"OK: best H({best.guess}) = {best.H:.6f} ≤ {Hmax:.6f}")
    else:
        print(f"[WARN] best H exceeds log2(|S|)? H={best.H:.6f}, log2N={Hmax:.6f}")

    # Deep dive on best guess
    deep_dive_on_guess(best.guess, S, top_k_buckets=args.deep_buckets)

    # CSV exports
    if args.export_per_guess:
        write_csv_per_guess(args.export_per_guess, ranked)
        print(f"\n[info] Wrote per-guess metrics to {args.export_per_guess}")
    if args.export_buckets:
        write_csv_buckets(args.export_buckets, best.guess, S)
        print(f"[info] Wrote bucket histogram for best guess to {args.export_buckets}")

    # Optional simulation of greedy-entropy gameplay
    if args.simulate > 0:
        print("\n=== Greedy-entropy simulation ===")
        avg, lengths = simulate(words, S, trials=args.simulate, seed=args.seed)
        print(f"Trials: {args.simulate}   Avg steps (to singleton set): {avg:.3f}")
        # quick histogram
        hist = Counter(lengths)
        print("Steps histogram:", dict(sorted(hist.items())))

if __name__ == "__main__":
    main()
