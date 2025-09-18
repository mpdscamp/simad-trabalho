from typing import List, Tuple
from collections import Counter

class Player:
    hints: List[Tuple[str, List[str]]]
    words: List[str]
    unused_consonants: set[str]
    consonant_frequency = {
        'b' : 1.04, 'c' : 3.88, 'd' : 5.01, 'f' : 1.02, 'g' : 1.30,
        'h' : 1.28, 'j' : 0.40,
        'k' : 0.02, 'l' : 2.78, 'm' : 4.74, 'n' : 5.05, 'p' : 2.52,
        'q' : 1.20, 'r' : 6.53, 's' : 7.81, 't' : 4.34, 'v' : 1.67,
        'w' : 0.01, 'x' : 0.21, 'y' : 0.01, 'z' : 0.47
    }
    def __init__(self, words: List[str]):
        self.words = words
        self.hints = []
        self.unused_consonants: set[str] = set("bcdfghjklmnpqrstvwxyz")
    def filter_all_vowels(self):
        vowels = "aeiou"
        return [word for word in self.words if all(c in word for c in vowels)]
    def bestword(self, worlist):
        # return the word with best score
        # score is the sum of the frequency of the consonants in the word that are in unused_consonants
        def score(word):
            return sum(self.consonant_frequency[c] for c in set(word) if c in self.unused_consonants)
        return max(worlist, key=score)
    def first_word(self):
        all_vowels =  self.filter_all_vowels()
        return self.bestword(all_vowels)
    def guess(self, round : int):
        if round == 1:
            ans = self.first_word()
        else:
            ans = self.bestword(self.words)
        for c in ans:
            if c in self.unused_consonants:
                self.unused_consonants.remove(c)
        return ans
    def feedback(self, hint: Tuple[str, List[str]]):
        # hint.first is the word guessed
        # hint.second is a list of strings, each is a hint for the corresponding letter in the guessed word
        # if the hint is a number, the letter is correct and it will correspond to which correct substring component that letter belongs to (1 indexed)
        # if the hint is '.', the letter is not in the word
        # if it is correct and has a prefix B, the letter is correct and belongs either to start or end position of the answer 
        # if the hint is 'Y', it means that the letter is in the wrong order relative to the other correct substrings 
        guess, hints = hint
        def check_win():
            win = 1
            start = 0
            end = 0
            for i, h in enumerate(hints):
                letter = h
                if h.startswith('B'):
                    start = start or i == 0
                    end = end or i == len(hints) - 1
                    letter = h[1:]
                win = win and letter.isdigit() and int(letter) == 1
            win = win and (start and end)
            return win
        def _extract_components() -> Tuple[List[str], List[int]]:
            components: List[str] = []
            positions: List[int] = []
            for i, h in enumerate(hints):
                letter = h
                if h.startswith('B'):
                    letter = h[1:]
                if letter.isdigit():
                    comp_id = int(letter)
                    if len(components) < comp_id:
                        components.extend([""])
                        positions.extend([i])
                    components[comp_id - 1] += guess[i]
            return components, positions
        def filter_dot(cands: List[str]) -> List[str]:
            min_required = Counter()
            has_gray = Counter()
            for i, h in enumerate(hints):
                ch = guess[i]
                if h == '.':
                    has_gray[ch] += 1
                else:
                    min_required[ch] += 1
            def check(word: str) -> bool:
                wc = Counter(word)
                for ch, need in min_required.items():
                    if wc[ch] < need:
                        return False
                for ch, _ in has_gray.items():
                    max_allowed = min_required.get(ch, 0)
                    if wc[ch] > max_allowed:
                        return False
                return True

            return [w for w in cands if check(w)]

        def filter_b_prefix(cands: List[str]) -> List[str]:
            components, _ = _extract_components()
            if not components:
                return cands

            def starts_with(word: str) -> bool:
                return word.startswith(components[0])

            def ends_with(word: str) -> bool:
                return word.endswith(components[-1])

            if hints[0].startswith('B'):
                cands = [w for w in cands if starts_with(w)]
            if hints[-1].startswith('B'):
                cands = [w for w in cands if ends_with(w)]
            return cands

        def filter_correct_components(cands: List[str]) -> List[str]:
            components, positions = _extract_components()
            def check(word: str) -> bool:
                pos = 0
                for ind, comp in enumerate(components):
                    if not comp:
                        continue
                    idx = word.find(comp, pos)
                    if idx == -1:
                        return True
                    pos = idx + len(comp)
                return True
            filtered = [w for w in cands if check(w)]
            return filtered

        def filter_yellows(cands: List[str]) -> List[str]:
            components, positions = _extract_components()
            if not components:
                return cands

            guess_positions: List[tuple[int, int]] = []
            pos = 0
            for comp in components:
                if not comp:
                    continue
                idx = guess.find(comp, pos)
                if idx == -1:
                    return cands
                guess_positions.append((idx, idx + len(comp)))
                pos = idx + len(comp)

            def check(word: str) -> bool:
                word_positions: List[tuple[int, int]] = []
                pos = 0
                for comp in components:
                    idx = word.find(comp, pos)
                    if idx == -1:
                        return True
                    word_positions.append((idx, idx + len(comp)))
                    pos = idx + len(comp)

                guess_bounds = [(0, 0)] + guess_positions + [(len(guess), len(guess))]
                word_bounds = [(0, 0)] + word_positions + [(len(word), len(word))]

                for (g_prev, g_next), (w_prev, w_next) in zip(zip(guess_bounds, guess_bounds[1:]),
                                                     zip(word_bounds, word_bounds[1:])):
                    g_start, g_end = g_prev[1], g_next[0]
                    w_start, w_end = w_prev[1], w_next[0]
                    yellow_here = {guess[i] for i in range(g_start, g_end) if hints[i] == "Y"}
                    segment = word[w_start:w_end]
                    if any(ch in yellow_here for ch in segment):
                        return False

                return True

            return [w for w in cands if check(w)]
        if check_win():
            self.words = [guess]
            return
        candidates = [w for w in self.words if w != guess]
        candidates = filter_b_prefix(candidates)
        candidates = filter_dot(candidates)
        candidates = filter_correct_components(candidates)
        candidates = filter_yellows(candidates)
        self.words = candidates
    def answer(self) -> str:
        if len(self.words) == 1:
            return self.words[0]
        if len(self.words) == 0:
            return "Failed"
        return ""
    def list_words(self) -> List[str]:
        return self.words

    