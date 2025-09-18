avg_rounds = 0
total = 0
worst_case = [0, ""]
with open("player.txt", "r") as f:
    lines = [line.strip() for line in f.readlines()]
    for line in lines:
        answer, player_answer, rounds = line.split(",")
        if answer == player_answer:
            avg_rounds += int(rounds)
            total += 1
            if int(rounds) > worst_case[0]:
                worst_case = [int(rounds), answer]
        else:
            print(f"Failed to guess {answer}, got {player_answer} in {rounds} rounds")
avg_rounds /= total
print(f"Average rounds: {avg_rounds}, total: {total}, worst case: {worst_case[0]} (word: {worst_case[1]})")