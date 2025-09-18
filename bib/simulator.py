import random
from player import Player
from hint import hint

if __name__ == "__main__":
    with open("words.txt", "r") as file:
        words = [line.strip() for line in file.readlines()]
    with open("player.txt", "w") as f:
        pass
    for cur_word in words[:10000]:
    # for cur_word in ["acaba"]:
        player = Player(words)
        answer = cur_word
        # print(f"Answer: {answer}")
        round = 1
        while player.answer() == "":
            guess = player.guess(round)
            # print(f"Guess {round}: {guess} (answer: {answer})")
            hints = hint(guess, answer)
            # print(f"Hint: {hints}")
            player.feedback((guess, hints))
            round += 1
            if round > 50:
                print(f"guess {guess}, answer {answer}, remaining {player.list_words()}")
                break
            # print(f"Remaining candidates: {player.list_words()}")
        with open("player.txt", "a") as f:
            f.write(f"{answer},{player.answer()},{round}\n")
