from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import Counter

@dataclass(frozen=True)
class Cand:
    gi0: int
    gi1: int
    tj0: int
    tj1: int
    L: int
    # weight favors length, then earlier target/guess starts for tie-breaks
    def weight(self) -> int:
        return self.L * 1000 - self.tj0 * 10 - self.gi0

def all_common_substrings(g: str, t: str) -> List[Cand]:
    m, n = len(g), len(t)
    cands: List[Cand] = []
    # Find all matching runs and emit every length from 1..run_len
    for i in range(m):
        for j in range(n):
            if g[i] != t[j]:
                continue
            L = 0
            while i + L < m and j + L < n and g[i + L] == t[j + L]:
                L += 1
            for k in range(1, L + 1):
                cands.append(Cand(i, i + k - 1, j, j + k - 1, k))
    return cands

def select_blocks_with_anchors(
    cands: List[Cand],
    m: int, n: int,
    anchor_start: Optional[Cand],  # a chosen candidate covering (0,0), or None
    anchor_end: Optional[Cand]     # a chosen candidate covering (m-1,n-1), or None
) -> List[Cand]:
    """Pick max-weight set of non-overlapping candidates that includes the given anchors (if any)."""
    # Build segment boundaries from anchors (in guess & target)
    bounds: List[Tuple[int, int]] = [(-1, -1)]
    if anchor_start is not None:
        bounds.append((anchor_start.gi1, anchor_start.tj1))
    if anchor_end is not None:
        bounds.insert(len(bounds), (anchor_end.gi0 - 1, anchor_end.tj0 - 1))  # place before final bound
    bounds.append((m, n))

    # Filter cands to avoid overlapping the anchors
    used_g = [False] * m
    used_t = [False] * n
    anchors: List[Cand] = []
    if anchor_start is not None:
        anchors.append(anchor_start)
        for gi in range(anchor_start.gi0, anchor_start.gi1 + 1):
            used_g[gi] = True
        for tj in range(anchor_start.tj0, anchor_start.tj1 + 1):
            used_t[tj] = True
    if anchor_end is not None:
        anchors.append(anchor_end)
        for gi in range(anchor_end.gi0, anchor_end.gi1 + 1):
            used_g[gi] = True
        for tj in range(anchor_end.tj0, anchor_end.tj1 + 1):
            used_t[tj] = True

    chosen: List[Cand] = anchors[:]  # we must include anchors

    # For each consecutive bound pair, run a simple O(K^2) DP on candidates inside the window
    for b in range(len(bounds) - 1):
        gL, tL = bounds[b]
        gR, tR = bounds[b + 1]
        seg = [
            c for c in cands
            if c.gi0 > gL and c.gi1 < gR and c.tj0 > tL and c.tj1 < tR
            and not any(used_g[gi] for gi in range(c.gi0, c.gi1 + 1))
            and not any(used_t[tj] for tj in range(c.tj0, c.tj1 + 1))
        ]
        # sort by (end positions) so compatibility check is simple; small tie drift
        seg.sort(key=lambda c: (c.gi1, c.tj1, -c.L, c.tj0, c.gi0))
        K = len(seg)
        if K == 0:
            continue
        dp = [c.weight() for c in seg]
        prev = [-1] * K

        def compat(a: Cand, b: Cand) -> bool:
            return a.gi1 < b.gi0 and a.tj1 < b.tj0

        for i in range(K):
            for j in range(i):
                if compat(seg[j], seg[i]):
                    cand_score = dp[j] + seg[i].weight()
                    if cand_score > dp[i] or (cand_score == dp[i] and
                                              (seg[j].tj0, seg[j].gi0) < (seg[prev[i]].tj0, seg[prev[i]].gi0) if prev[i] != -1 else True):
                        dp[i] = cand_score
                        prev[i] = j
        # reconstruct
        idx = max(range(K), key=lambda i: dp[i])
        chain = []
        while idx != -1:
            chain.append(seg[idx])
            idx = prev[idx]
        chain.reverse()
        # mark used and append
        for c in chain:
            for gi in range(c.gi0, c.gi1 + 1):
                used_g[gi] = True
            for tj in range(c.tj0, c.tj1 + 1):
                used_t[tj] = True
        chosen.extend(chain)

    # sort chosen by (gi0, tj0) for consistent downstream numbering
    chosen.sort(key=lambda c: (c.gi0, c.tj0))
    return chosen

def hint(guess: str, target: str) -> List[str]:
    g, t = guess, target
    m, n = len(g), len(t)

    # Step 1: enumerate all common substrings (candidates)
    cands = all_common_substrings(g, t)

    # Step 2: find possible anchors covering blues (start/end)
    start_cover = [c for c in cands if c.gi0 == 0 and c.tj0 == 0]
    end_cover   = [c for c in cands if c.gi1 == m - 1 and c.tj1 == n - 1]

    # If not blue, the list is empty (no constraint).
    # Choose anchors by trying all feasible combos (small search since m,n<=10)
    best_blocks: List[Cand] = []
    best_score = -10**9

    # if no blue at edge, pretend anchor list = [None]
    start_opts = start_cover if (m > 0 and n > 0 and g[0] == t[0]) else [None]
    end_opts   = end_cover   if (m > 0 and n > 0 and g[-1] == t[-1]) else [None]

    for a_start in start_opts:
        for a_end in end_opts:
            # Ensure anchors (if both present) don't conflict and are ordered
            if a_start and a_end:
                if not (a_start.gi1 < a_end.gi0 and a_start.tj1 < a_end.tj0):
                    continue
            blocks = select_blocks_with_anchors(cands, m, n, a_start, a_end)
            score = sum(c.weight() for c in blocks)
            if score > best_score or (score == best_score and blocks < best_blocks):
                best_score = score
                best_blocks = blocks

    # Step 3: build mapping g_idx -> t_idx from chosen blocks (greens)
    g2t: Dict[int, int] = {}
    used_t = [False] * n
    for c in best_blocks:
        for k in range(c.L):
            gi = c.gi0 + k
            tj = c.tj0 + k
            g2t[gi] = tj
            used_t[tj] = True

    # Step 4: form runs (contiguous in guess and target) and number them left->right
    runs: List[List[int]] = []
    sorted_g = sorted(g2t.keys())
    cur = []
    prev_g = prev_t = None
    for gi in sorted_g:
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

    # Number blocks by run order from left to right (in guess)
    gi_to_block: Dict[int, int] = {}
    for block_id, run in enumerate(runs, start=1):
        for gi in run:
            gi_to_block[gi] = block_id

    # Step 5: build tokens; prefix blue positions with 'B' if applicable
    tokens = [""] * m

    blue_start = (m > 0 and n > 0 and g[0] == t[0])
    blue_end   = (m > 0 and n > 0 and g[-1] == t[-1])

    for gi, tj in g2t.items():
        bid = gi_to_block[gi]
        tok = str(bid)
        if (gi == 0 and blue_start and tj == 0) or (gi == m - 1 and blue_end and tj == n - 1):
            tok = "B" + tok
        tokens[gi] = tok

    # Step 6: assign Y / . by multiplicities for non-green positions
    t_counts = Counter(t)
    for gi in g2t:
        t_counts[g[gi]] -= 1
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
