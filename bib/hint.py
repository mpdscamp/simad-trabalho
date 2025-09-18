from typing import List, Tuple, Optional
from collections import Counter

def _lcsstr_preferring_blues(g: str, t: str, blue_pairs: set[tuple[int,int]]) -> Optional[tuple[int,int,int,int]]:
    """
    Longest common substring, but prefer any candidate that overlaps a blue-fixed (gi,tj).
    Tie-break inside each preference class by smallest (t_start, g_start).
    Returns (g_start, g_end, t_start, t_end) or None.
    """
    m, n = len(g), len(t)
    if m == 0 or n == 0:
        return None

    # dp[i][j] = LCSstr ending at g[i-1], t[j-1]
    dp = [[0]*(n+1) for _ in range(m+1)]
    best_overlap = None   # (L, ts, gs, ge, te)
    best_plain   = None   # (L, ts, gs, ge, te)

    def upd(slot, L, gs, ge, ts, te):
        if slot is None:
            return (L, ts, gs, ge, te)
        L0, ts0, gs0, ge0, te0 = slot
        if L > L0 or (L == L0 and (ts, gs) < (ts0, gs0)):
            return (L, ts, gs, ge, te)
        return slot

    for i in range(1, m+1):
        gi = g[i-1]
        for j in range(1, n+1):
            if gi == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                L = dp[i][j]
                gs = i - L; ge = i - 1
                ts = j - L; te = j - 1
                # does this substring overlap any blue-fixed (gi,tj)?
                overlaps_blue = any((gs <= bgi <= ge) and (ts <= btj <= te) for (bgi, btj) in blue_pairs)
                if overlaps_blue:
                    best_overlap = upd(best_overlap, L, gs, ge, ts, te)
                else:
                    best_plain   = upd(best_plain,   L, gs, ge, ts, te)

    pick = best_overlap if best_overlap is not None else best_plain
    if pick is None:
        return None
    _, ts, gs, ge, te = pick
    return (gs, ge, ts, te)

def _lcsstr_canonical(g: str, t: str) -> Optional[Tuple[int,int,int,int]]:
    """
    Longest common substring with canonical tie-break:
    choose the match with smallest (t_start, g_start).
    Returns (g_start, g_end, t_start, t_end) or None.
    """
    m, n = len(g), len(t)
    if m == 0 or n == 0:
        return None
    dp = [[0]*(n+1) for _ in range(m+1)]
    best = None  # (g_start, g_end, t_start, t_end)
    bestL = 0
    for i in range(1, m+1):
        gi = g[i-1]
        for j in range(1, n+1):
            if gi == t[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                L = dp[i][j]
                gs = i - L
                ts = j - L
                # prefer larger L; for ties pick smallest (ts, gs)
                if (L > bestL) or (L == bestL and (best is None or (ts, gs) < (best[2], best[0]))):
                    bestL = L
                    best = (gs, i-1, ts, j-1)
    return best if bestL > 0 else None

def _greedy_match_side(g: str, t: str, gi_range: range, tj_lo: int, tj_hi: int,
                       used_t: List[bool], g2t: dict) -> None:
    """
    Greedy earliest feasible subsequence on one side:
    for gi in gi_range, match g[gi] to the earliest tj in (tj_lo, tj_hi)
    with same letter and tj > last_t, not used.
    """
    last_t = tj_lo
    for gi in gi_range:
        if gi in g2t:  # already fixed (e.g., blue or part of main block)
            last_t = max(last_t, g2t[gi])
            continue
        c = g[gi]
        tj = last_t + 1
        while tj < tj_hi and (used_t[tj] or t[tj] != c):
            tj += 1
        if tj < tj_hi:
            g2t[gi] = tj
            used_t[tj] = True
            last_t = tj

def hint(guess: str, target: str) -> List[str]:
    g, t = guess, target
    m, n = len(g), len(t)

    # --- Step 0: setup
    g2t = {}                 # mapping: guess index -> target index (greens)
    used_t = [False] * n     # consumed target positions

    # --- Step 1: blue anchors (consume positions)
    blue_start = (m > 0 and n > 0 and g[0] == t[0])
    blue_end   = (m > 0 and n > 0 and g[-1] == t[-1])
    if blue_start:
        g2t[0] = 0; used_t[0] = True
    if blue_end and (m-1 not in g2t):
        g2t[m-1] = n-1; used_t[n-1] = True

    # collect blue pairs for preference
    blue_pairs = {(gi, tj) for gi, tj in g2t.items()}

    # --- Step 2: main block (LCSstr preferring overlaps with blue)
    blk = _lcsstr_preferring_blues(g, t, blue_pairs)
    if blk is not None:
        gs, ge, ts, te = blk
        # place it (consuming target positions), consistent with any blue already set
        for k in range(ge - gs + 1):
            gi = gs + k
            tj = ts + k
            if gi not in g2t and not used_t[tj]:  # avoid conflicting with blue
                g2t[gi] = tj
                used_t[tj] = True
    else:
        gs = ge = ts = te = -1  # sentinel when no main block exists

    # --- Step 3: greedy subsequence matches on each side (unique)
    if gs != -1:
        # left side: target indices in [0, ts)
        _greedy_match_side(g, t, range(0, gs), -1, ts, used_t, g2t)
        # right side: target indices in (te, n)
        _greedy_match_side(g, t, range(ge+1, m), te, n, used_t, g2t)
    else:
        # no main block: match across whole string
        _greedy_match_side(g, t, range(0, m), -1, n, used_t, g2t)

    # --- Step 4: split greens into contiguous runs (blocks)
    runs = []
    cur = []
    prev_g = prev_t = None
    for gi in sorted(g2t.keys()):
        tj = g2t[gi]
        if cur and gi == prev_g + 1 and tj == prev_t + 1:
            cur.append(gi)
        else:
            if cur:
                runs.append(cur)
            cur = [gi]
        prev_g, prev_t = gi, tj
    if cur:
        runs.append(cur)

    # --- Step 5: assign canonical block numbers (left->right in guess)
    gi_to_block = {}
    for block_id, run in enumerate(runs, start=1):
        for gi in run:
            gi_to_block[gi] = block_id

    # --- Step 6: prepare counts for Y / .
    t_counts = Counter(t)
    for gi in gi_to_block:
        t_counts[g[gi]] -= 1

    # --- Step 7: compose tokens with blue overlay
    tokens = [""] * m
    for gi, tj in g2t.items():
        b = gi_to_block[gi]
        tok = str(b)
        if (gi == 0 and blue_start and tj == 0) or (gi == m-1 and blue_end and tj == n-1):
            tok = "B" + tok
        tokens[gi] = tok

    # --- Step 8: fill remaining with Y / .
    for i in range(m):
        if tokens[i]:
            continue
        c = g[i]
        if t_counts.get(c, 0) > 0:
            tokens[i] = "Y"
            t_counts[c] -= 1
        else:
            tokens[i] = "."

    return tokens
