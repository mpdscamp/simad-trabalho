import argparse
import csv
import json
import math
import os
import random
import sys
import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------- import your canonical hint -----------
from hint import hint


# ---- Worker globals & helpers for multiprocessing ----
_worker_words = None  # set once per worker process

def _init_worker_words(words_list):
    # words_list is pickled once to each worker at start
    global _worker_words
    _worker_words = words_list

def _row_for_guess_worker(g: str) -> list[str]:
    # Build one row: hint_key for (g, t) over all targets t
    # Uses the per-worker global _worker_words to avoid re-sending large data.
    global _worker_words
    return [hint_key(g, t) for t in _worker_words]


# ---------- utilities ---------------------------
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
    counts = sorted(buckets.values(), reverse=True)
    N = sum(counts)
    H = entropy_from_counts(counts)
    worst = counts[0]
    num_buckets = len(counts)
    num_singletons = sum(1 for c in counts if c == 1)
    perplexity = 2.0 ** H
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

# ---------- fast path: reuse per_guess_metrics.csv if present ----------
def read_per_guess_metrics_csv(path: str) -> Optional[List[GuessMetrics]]:
    if not os.path.exists(path):
        return None
    out: List[GuessMetrics] = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            out.append(GuessMetrics(
                guess=row["guess"],
                H=float(row["entropy_bits"]),
                num_buckets=int(row["num_buckets"]),
                num_singletons=int(row["num_singletons"]),
                worst_bucket_size=int(row["worst_bucket_size"]),
                worst_bucket_frac=float(row["worst_bucket_frac"]),
                perplexity=float(row["perplexity"]),
                hhi=float(row["hhi"]),
                eff_num_buckets_hhi=float(row["eff_num_buckets_hhi"]),
                avg_bucket_size=float(row["avg_bucket_size"]),
            ))
    return out

def write_per_guess_metrics_csv(path: str, metrics_list: List[GuessMetrics]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["guess","entropy_bits","num_buckets","num_singletons","worst_bucket_size","worst_bucket_frac","perplexity","hhi","eff_num_buckets_hhi","avg_bucket_size"])
        for m in metrics_list:
            w.writerow([m.guess,f"{m.H:.6f}",m.num_buckets,m.num_singletons,m.worst_bucket_size,f"{m.worst_bucket_frac:.6f}",f"{m.perplexity:.6f}",f"{m.hhi:.8f}",f"{m.eff_num_buckets_hhi:.6f}",f"{m.avg_bucket_size:.6f}"])

def write_best_guess_buckets_csv(path: str, guess: str, S: List[str]) -> None:
    buckets = bucket_counts_for_guess(guess, S)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket_key","count"])
        for k, c in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
            w.writerow([k, c])

# ---------- parallel entropy for all guesses (first pass) ----------
def _counts_for_guess(g: str, words: List[str]) -> Tuple[str, Dict[str,int]]:
    buckets: Dict[str, int] = defaultdict(int)
    for t in words:
        buckets[hint_key(g, t)] += 1
    return g, buckets

def rank_guesses_parallel(words: List[str], max_workers: int = None, limit_guesses: int = 0) -> List[GuessMetrics]:
    # Optionally limit guesses to the first N (useful for a quick scan)
    G = words if limit_guesses <= 0 or limit_guesses >= len(words) else words[:limit_guesses]
    results: List[GuessMetrics] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_counts_for_guess, g, words): g for g in G}
        done = 0
        for fut in as_completed(futures):
            g, buckets = fut.result()
            m = summarize_buckets(buckets)
            m.guess = g
            results.append(m)
            done += 1
            if done % 100 == 0:
                print(f"[info] evaluated {done}/{len(G)} guesses", file=sys.stderr)

    results.sort(key=lambda x: (-x.H, x.worst_bucket_size, x.guess))
    return results

# ---------- Precompute hint keys matrix for top-K guesses ----------
def precompute_keys_matrix(top_guesses: List[str], words: List[str], out_path: str, max_workers: int = None) -> List[List[str]]:
    """
    Returns keys_mat where keys_mat[i][j] = hint_key(top_guesses[i], words[j]).
    Saves a .npz file for reuse.
    """
    import numpy as np

    if os.path.exists(out_path):
        npz = np.load(out_path, allow_pickle=True)
        arr = npz["keys"]
        return [list(row) for row in arr]

    # Compute rows in parallel; each worker gets 'words' once in initializer.
    rows: list[Optional[list[str]]] = [None] * len(top_guesses)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_words, initargs=(words,)) as ex:
        futures = {ex.submit(_row_for_guess_worker, g): i for i, g in enumerate(top_guesses)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            rows[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(top_guesses):
                print(f"[info] precomputed rows {done}/{len(top_guesses)}", file=sys.stderr)

    # Type ignore since we filled all positions
    keys_mat: List[List[str]] = rows  # type: ignore

    np.savez_compressed(out_path, keys=np.array(keys_mat, dtype=object))
    return keys_mat

# ---------- Greedy entropy policy using precomputed keys ----------
def pick_best_guess_from_allowed(allowed_idx: List[int], S_idx: List[int], keys_mat: List[List[str]]) -> Tuple[int, Dict[str,int], float]:
    """
    allowed_idx: indices into top_guesses (rows of keys_mat)
    S_idx: indices into words (columns of keys_mat)
    Returns (best_guess_row_idx, bucket_counts_for_that_guess, entropy)
    """
    best_row = -1
    best_H = -1.0
    best_buckets: Dict[str,int] = {}
    for row in allowed_idx:
        buckets: Dict[str,int] = defaultdict(int)
        row_keys = keys_mat[row]
        for j in S_idx:
            buckets[row_keys[j]] += 1
        H = entropy_from_counts(buckets.values())
        if H > best_H or (H == best_H and (best_row == -1 or max(buckets.values()) < max(best_buckets.values()))):
            best_H = H
            best_row = row
            best_buckets = buckets
    return best_row, best_buckets, best_H

# ---------- Simulation (interrupt-safe, uses precomputed keys) ----------
def simulate_for_initial_guess(
    init_guess_row: int,
    words: List[str],
    keys_mat: List[List[str]],
    trials: int,
    seed: int,
    allowed_rows: List[int],
    allowed_words: List[str],
    max_steps: int = 15,
    simulate_all: bool = False,
) -> Dict:
    """
    Simulate greedy-entropy play using ONLY guesses from allowed_rows (top-K).
    allowed_words: the actual guess strings corresponding to allowed_rows (top_guesses).
    """
    rnd = random.Random(seed + init_guess_row * 1337)
    N = len(words)

    # Choose targets: exhaustive or sample-without-replacement
    if simulate_all:
        target_indices = list(range(N))
    else:
        sample_n = min(trials, N)
        target_indices = rnd.sample(range(N), sample_n)  # no repeats

    # We’ll report the actual number of trials we ran:
    trials = len(target_indices)

    steps_list: List[int] = []
    solved_count = 0
    hardest_steps = -1
    hardest_path = []
    hardest_target = None

    for t_idx in target_indices:
        S_idx = list(range(N))
        path = []
        steps = 0

        while steps < max_steps and len(S_idx) > 1:
            if steps == 0:
                g_row = init_guess_row
            else:
                g_row, _, _ = pick_best_guess_from_allowed(allowed_rows, S_idx, keys_mat)

            g_word = allowed_words[g_row]
            obs_key = keys_mat[g_row][t_idx]

            # compute filtered size without mutating yet
            row_keys = keys_mat[g_row]
            new_size = sum(1 for j in S_idx if row_keys[j] == obs_key)

            # SAFEGUARD: if guess does NOT shrink S, or S is tiny, guess a candidate directly
            if new_size == len(S_idx) or len(S_idx) <= 8:
                # pick a candidate to guess (first in S_idx)
                direct_idx = S_idx[0]
                g_word = words[direct_idx]

                # compute observed key on the fly (we may be outside top-K)
                obs_key = hint_key(g_word, words[t_idx])

                # log the step
                path.append((g_word, obs_key, len(S_idx)))

                # filter S_idx using on-the-fly keys
                S_idx = [j for j in S_idx if hint_key(g_word, words[j]) == obs_key]
                steps += 1
                if len(S_idx) <= 1:
                    break
                else:
                    continue  # next step

            # NORMAL PATH (shrank S): log & filter using precomputed row
            path.append((g_word, obs_key, len(S_idx)))
            S_idx = [j for j in S_idx if row_keys[j] == obs_key]
            steps += 1
            if len(S_idx) <= 1:
                break

        if len(S_idx) == 1:
            final_idx = S_idx[0]
            final_word = words[final_idx]
            # the observed key against the true target:
            final_key = hint_key(final_word, words[t_idx])
            path.append((final_word, final_key, 1))
        
        if len(S_idx) <= 1:
            solved_count += 1
            steps_list.append(steps)
        else:
            steps_list.append(max_steps)

        if steps_list[-1] > hardest_steps:
            hardest_steps = steps_list[-1]
            hardest_path = path
            hardest_target = words[t_idx]

    steps_list_sorted = sorted(steps_list)
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else float("nan")
    median_steps = steps_list_sorted[len(steps_list)//2] if steps_list else float("nan")
    pct_solved = 100.0 * solved_count / trials if trials else 0.0

    return {
        "init_guess": allowed_words[init_guess_row],
        "trials_run": trials,
        "pct_solved": pct_solved,
        "avg_steps": avg_steps,
        "median_steps": median_steps,
        "max_steps": max(steps_list) if steps_list else 0,
        "hardest_target": hardest_target,
        "hardest_path": [{"guess": g, "key": k, "candidates_before": n} for (g, k, n) in hardest_path],
        "steps_hist": dict(Counter(steps_list)),
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Fast, interrupt-safe analysis & simulation.")
    ap.add_argument("--words", default="words.txt", help="path to words list (one per line)")
    ap.add_argument("--limit-guesses", type=int, default=0, help="limit guesses to first N for entropy scan (0 = all)")
    ap.add_argument("--top-k", type=int, default=100, help="top-K guesses to consider in simulation (100..1000)")
    ap.add_argument("--workers", type=int, default=0, help="process workers (0=auto)")
    ap.add_argument("--simulate", type=int, default=1000, help="trials per initial guess (0=skip simulation)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--per-guess-csv", default="per_guess_metrics.csv", help="metrics CSV path")
    ap.add_argument("--best-buckets-csv", default="best_guess_buckets.csv", help="best guess buckets CSV path")
    ap.add_argument("--keys-cache", default="precomputed_keys_topK.npz", help="cache for keys matrix")
    ap.add_argument("--resume", action="store_true", help="resume simulation_summary.csv if partially written")
    ap.add_argument("--simulate-all", action="store_true", help="if set, simulate each initial guess against all targets exactly once (ignores --simulate)")
    args = ap.parse_args()

    global words_allowed  # shared mapping for simulation function
    words = load_words(args.words)
    N = len(words)
    print(f"[info] Loaded {N} words.")

    workers = None if args.workers <= 0 else args.workers

    # === 1) Ensure per-guess metrics exist (or compute in parallel)
    metrics = read_per_guess_metrics_csv(args.per_guess_csv)
    if metrics is None:
        print("[info] per_guess_metrics.csv not found; computing...", file=sys.stderr)
        ranked = rank_guesses_parallel(words, max_workers=workers, limit_guesses=args.limit_guesses)
        write_per_guess_metrics_csv(args.per_guess_csv, ranked)
        metrics = ranked
        # also write best guess bucket histogram
        best = ranked[0]
        print(f"[info] Best guess by entropy: {best.guess} (H={best.H:.6f})", file=sys.stderr)
        write_best_guess_buckets_csv(args.best_buckets_csv, best.guess, words)
    else:
        metrics.sort(key=lambda x: (-x.H, x.worst_bucket_size, x.guess))
        print(f"[info] Loaded per-guess metrics (top1={metrics[0].guess}, H={metrics[0].H:.6f})", file=sys.stderr)

    # === 2) Select top-K guesses
    K = max(1, min(args.top_k, len(metrics)))
    top_guesses = [m.guess for m in metrics[:K]]
    print(f"[info] Using top-K={K} guesses for simulation.", file=sys.stderr)

    # === 3) Precompute keys matrix for (topK guesses x targets)
    keys_mat = precompute_keys_matrix(top_guesses, words, out_path=args.keys_cache, max_workers=workers)

    # === 4) Simulation (optional)
    if not args.simulate_all and args.simulate <= 0:
        print("[info] Simulation skipped (--simulate=0 and --simulate-all not set). Done.")
        return

    # Prepare output sinks
    summary_path = "simulation_summary.csv"
    paths_path = "simulation_paths.jsonl"
    summary_exists = os.path.exists(summary_path) and args.resume

    # open files append-mode for incremental saves
    summary_f = open(summary_path, "a" if summary_exists else "w", encoding="utf-8", newline="")
    paths_f = open(paths_path, "a" if args.resume and os.path.exists(paths_path) else "w", encoding="utf-8")

    summary_writer = csv.writer(summary_f)
    if not summary_exists:
        summary_writer.writerow([
            "init_guess","trials_run","pct_solved","avg_steps","median_steps","max_steps",
            "hardest_target","hardest_path_compact","worst_bucket_size_initial"
        ])

    # figure out which initial guesses already done (resume)
    done_inits = set()
    if args.resume and summary_exists:
        summary_f.flush()
        with open(summary_path, "r", encoding="utf-8") as sf:
            rdr = csv.DictReader(sf)
            for row in rdr:
                done_inits.add(row["init_guess"])

    try:
        # we parallelize across initial guesses (each aggregates its own trials)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {}
            allowed_rows = list(range(K))
            # initial worst bucket sizes (first split) for info
            init_worst = { top_guesses[row]: max(Counter(keys_mat[row]).values()) for row in allowed_rows }

            for row in allowed_rows:
                init_guess = top_guesses[row]
                if init_guess in done_inits:
                    continue
                fut = ex.submit(
                    simulate_for_initial_guess,
                    row,              # init_guess_row
                    words,            # all targets
                    keys_mat,
                    args.simulate,    # trials (ignored if simulate_all)
                    args.seed,
                    allowed_rows,     # policy rows
                    top_guesses,      # allowed_words (strings)
                    15,               # max_steps
                    args.simulate_all
                )

                futures[fut] = row

            done = 0
            for fut in as_completed(futures):
                row = futures[fut]
                res = fut.result()
                # write JSONL (detailed path for hardest trial)
                paths_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                paths_f.flush()

                # compact path for CSV (truncate for readability)
                compact = " | ".join([f"{p['guess']} → {p['key']} (|S|={p['candidates_before']})" for p in res["hardest_path"]])
                if len(compact) > 300:
                    compact = compact[:297] + "..."

                summary_writer.writerow([
                    res["init_guess"], res["trials_run"], f"{res['pct_solved']:.2f}",
                    f"{res['avg_steps']:.3f}", res["median_steps"], res["max_steps"],
                    res["hardest_target"] or "",
                    compact,
                    init_worst.get(res["init_guess"], 0)
                ])
                summary_f.flush()

                done += 1
                if done % 10 == 0 or done == len(futures):
                    print(f"[info] simulation progress: {done}/{len(futures)} initial guesses", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n[info] CTRL+C received, flushing partial results...", file=sys.stderr)

    finally:
        summary_f.close()
        paths_f.close()

    print("[info] Done.", file=sys.stderr)

if __name__ == "__main__":
    # Make CTRL+C responsive on some platforms
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
