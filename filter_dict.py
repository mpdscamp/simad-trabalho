#!/usr/bin/env python3
# filter_dict.py
import argparse
import sys
import unicodedata
import re
from pathlib import Path

# Unicode-aware: only letters, length 4..10
LETTER_ONLY_4_10 = re.compile(r"^[^\W\d_]{4,10}$", re.UNICODE)

# Common hyphen and apostrophe variants to exclude explicitly
DISALLOWED = set([
    "-", "–", "—", "‒", "―",              # hyphen/dash variants
    "'", "’", "‘", "ʼ", "ʹ"               # apostrophes
])

def has_disallowed_chars(s: str) -> bool:
    return any(ch in DISALLOWED for ch in s)

def main():
    ap = argparse.ArgumentParser(description="Filter Portuguese word list.")
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input word list (one word per line).")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="Output file (default: <input> filtered to *_4_10.txt).")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out_path) if args.out_path else in_path.with_name(
        in_path.stem + "_4_10.txt"
    )

    kept = 0
    skipped = 0
    seen = set()

    with in_path.open("r", encoding="utf-8", errors="strict") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for raw in fin:
            w = raw.strip()
            if not w:
                continue

            # Normalize for consistent diacritics
            w = unicodedata.normalize("NFC", w)

            # Skip if starts with uppercase letter (proper name)
            if w[0].isupper():
                skipped += 1
                continue

            # Skip if contains hyphen/apostrophe variants
            if has_disallowed_chars(w):
                skipped += 1
                continue

            # Keep only pure letters, length 4..10
            if LETTER_ONLY_4_10.match(w):
                if w not in seen:
                    fout.write(w + "\n")
                    seen.add(w)
                    kept += 1
            else:
                skipped += 1

    print(f"Done. Kept: {kept} | Skipped: {skipped} | Output: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
