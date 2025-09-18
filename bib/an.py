from hint import hint
import math
from collections import defaultdict

with open("words.txt") as f:
    words = [line.strip() for line in f if line.strip()]

def hint_key(guess: str, target: str) -> str:
    return ' '.join(hint(guess, target))

def expected_entropy(g: str, S: list[str]) -> float:
    buckets = defaultdict(int)
    for t in S:
        k = hint_key(g, t)
        buckets[k] += 1
    N = len(S)
    H = 0.0
    for c in buckets.values():
        p = c / N
        H -= p * math.log2(p)
    return H

best_g = None
best_H = -1
for g in words:
    H = expected_entropy(g, words)
    if H > best_H:
        best_H, best_g = H, g
print("Best guess:", best_g, "with expected entropy:", best_H)

observed = "B1 1 . 2 3 Y . ."  # example
words = [t for t in words if hint_key(best_g, t) == observed]
