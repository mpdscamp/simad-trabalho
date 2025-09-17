#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Portuguese word list for a Wordle-like game.

What's new vs. previous version:
- Whitelist with Hunspell pt_BR + pt_PT (keeps only words recognized by the lexicon)
- Optionally ban foreign letters k/w/y (default on) to avoid names/loanwords
- Still merges frequency lists (pt + pt_BR), removes stopwords, filters by length,
  ranks by combined frequency, and caps the list (default 10,000)

Output: words.txt
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script requires 'requests'. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

# Frequency lists (HermitDave)
FREQ_SOURCES = {
    "pt":    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/pt/pt_full.txt",
    "pt_br": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/pt_br/pt_br_full.txt",
}

# Stopwords (stopwords-iso)
STOPWORDS_SOURCE = "https://raw.githubusercontent.com/stopwords-iso/stopwords-pt/master/stopwords-pt.json"

# Hunspell dictionaries (word lists). We’ll parse only plain entries before slashes/flags.
# These mirrors are commonly available; if one fails, the script continues with the other.
HUNSPELL_SOURCES = [
    # LibreOffice pt_BR
    "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_BR/pt_BR.dic",
    # LibreOffice pt_PT
    "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_PT/pt_PT.dic",
]

EXTRA_STOPWORDS = {
    "pro", "pra", "pras", "pros", "tá", "tão", "num", "nuns", "numa", "numas",
    "cadê", "tô", "cê", "né", "ah", "eh", "ô", "hã",
}

def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def load_stopwords(strip: bool) -> set:
    resp = requests.get(STOPWORDS_SOURCE, timeout=60)
    resp.raise_for_status()
    sw = set(json.loads(resp.text))
    sw |= EXTRA_STOPWORDS
    expanded = set()
    for w in sw:
        wl = w.lower()
        wl = strip_accents(wl) if strip else wl
        expanded.add(wl)
        expanded.add(re.sub(r"[^a-zà-ÿ]", "", wl))
    return {w for w in expanded if w}

def is_alpha_word(w: str) -> bool:
    return len(w) > 0 and all(c.isalpha() for c in w)

def download_text(url: str, timeout=120) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text

def parse_freq(text: str) -> dict:
    freq = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        try:
            word, count = line.rsplit(" ", 1)
            count = int(float(count))
        except Exception:
            continue
        word = word.strip()
        if not word:
            continue
        freq[word] = freq.get(word, 0) + max(count, 0)
    return freq

def load_hunspell_wordset(strip: bool) -> set:
    lex = set()
    for url in HUNSPELL_SOURCES:
        try:
            txt = download_text(url)
        except Exception as e:
            print(f"Warning: failed to load Hunspell from {url}: {e}", file=sys.stderr)
            continue
        lines = txt.splitlines()
        # First line might be an integer count in Hunspell .dic
        start_idx = 1 if lines and lines[0].strip().isdigit() else 0
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            # Entries may be like 'coração/SM' -> take token before slash
            token = line.split("/")[0].strip()
            token = token.lower()
            token = strip_accents(token) if strip else token
            # keep only alphabetic words (after optional accent strip)
            if strip:
                if not re.fullmatch(r"[a-z]+", token):
                    continue
            else:
                if not is_alpha_word(token):
                    continue
            lex.add(token)
        print(f"Loaded {len(lex):,} lexicon entries (cumulative) from {url}", file=sys.stderr)
    return lex

def build_vocab(args):
    # Load lexicon whitelist
    lexicon = load_hunspell_wordset(strip=args.strip_accents)
    if not lexicon:
        print("Warning: Hunspell lexicon unavailable; falling back to frequency-only (not recommended).", file=sys.stderr)

    # Load frequency lists (pt + pt_BR)
    combined = defaultdict(int)
    for name, url in FREQ_SOURCES.items():
        try:
            part = parse_freq(download_text(url))
            for w, c in part.items():
                combined[w] += c
            print(f"Loaded {len(part):,} frequency entries from {name}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: failed to load '{name}' ({url}): {e}", file=sys.stderr)

    if not combined:
        raise RuntimeError("No frequency data retrieved. Check your network or sources.")

    # Prepare stopwords
    stopwords = load_stopwords(strip=args.strip_accents)

    kept = {}
    for raw_w, cnt in combined.items():
        w = raw_w.strip().lower()
        w = strip_accents(w) if args.strip_accents else w

        # Letters-only
        if args.strip_accents:
            if not re.fullmatch(r"[a-z]+", w):
                continue
        else:
            if not is_alpha_word(w):
                continue

        # Length
        if not (args.min_len <= len(w) <= args.max_len):
            continue

        # Optional ban on k/w/y to avoid foreign names/loanwords
        if args.ban_foreign_letters and re.search(r"[kwy]", w):
            continue

        # Stopwords
        if w in stopwords:
            continue

        # Whitelist with Hunspell (if available)
        if lexicon and w not in lexicon:
            continue

        kept[w] = kept.get(w, 0) + cnt

    if not kept:
        raise RuntimeError("No words survived filtering. Try relaxing filters or disable some options.")

    # Sort by frequency and cap
    sorted_words = sorted(kept.items(), key=lambda kv: kv[1], reverse=True)
    if args.max_words > 0:
        sorted_words = sorted_words[: args.max_words]
    words = [w for w, _ in sorted_words]

    # Write
    out_path = Path(args.output).resolve()
    out_path.write_text("\n".join(words) + "\n", encoding="utf-8")

    print(f"\nCreated {len(words):,} words → {out_path}", file=sys.stderr)
    print("Top 25 preview:", ", ".join(words[:25]), file=sys.stderr)

def main():
    p = argparse.ArgumentParser(description="Build a Portuguese word list for Wordle-like games.")
    p.add_argument("--min-len", type=int, default=4, help="Minimum word length (default: 4)")
    p.add_argument("--max-len", type=int, default=10, help="Maximum word length (default: 10)")
    p.add_argument("--max-words", type=int, default=10000, help="Cap number of words (default: 10000)")
    p.add_argument("--output", type=str, default="words.txt", help="Output file (default: words.txt)")
    p.add_argument("--strip-accents", action="store_true", help="Strip diacritics (ú→u, ç→c)")
    p.add_argument("--ban-foreign-letters", action=argparse.BooleanOptionalAction, default=True,
                   help="Ban letters k/w/y (default: True; use --no-ban-foreign-letters to allow)")
    args = p.parse_args()

    if args.min_len < 2 or args.max_len < args.min_len:
        print("Invalid length bounds.", file=sys.stderr)
        sys.exit(2)

    try:
        build_vocab(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
