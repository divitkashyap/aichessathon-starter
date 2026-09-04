"""Play a candidate/champion match with every opening used in both directions."""

import argparse
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local
from tools.openings import opening_fens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=2_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--openings", type=int, default=12)
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    champion = arguments.champion.resolve()
    wins = draws = losses = 0
    failures: dict[str, int] = {}

    games: list[tuple[Path, Path, bool, str]] = []
    for fen in opening_fens()[: arguments.openings]:
        games.append((candidate, champion, True, fen))
        games.append((champion, candidate, False, fen))

    for index, (white, black, candidate_white, fen) in enumerate(games, start=1):
        outcome = play_match(
            local(white),
            local(black),
            arguments.base_ms,
            arguments.increment_ms,
            start_fen=fen,
        )
        if outcome.termination in FAILED_TERMINATIONS:
            failures[outcome.termination] = failures.get(outcome.termination, 0) + 1
        if outcome.result == "draw":
            draws += 1
        elif (outcome.result == "white") == candidate_white:
            wins += 1
        else:
            losses += 1
        print(
            f"game {index}/{len(games)}: {outcome.result} by {outcome.termination} "
            f"(candidate {'white' if candidate_white else 'black'})",
            flush=True,
        )

    score = (wins + draws / 2) / len(games)
    print(f"candidate +{wins} ={draws} -{losses}, score {score:.1%}")
    if failures:
        names = ", ".join(f"{name} {count}" for name, count in failures.items())
        raise SystemExit(f"technical failures: {names}")


if __name__ == "__main__":
    main()
