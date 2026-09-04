"""Play a candidate/champion match with every opening used in both directions."""

import argparse
import hashlib
import json
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
    parser.add_argument("--opening-start", type=int, default=0)
    parser.add_argument("--output", type=Path, help="New directory for reproducible game records")
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    champion = arguments.champion.resolve()
    if not 1 <= arguments.openings <= len(opening_fens()):
        parser.error(f"--openings must be between 1 and {len(opening_fens())}")
    if arguments.opening_start < 0 or arguments.opening_start + arguments.openings > len(
        opening_fens()
    ):
        parser.error("opening range is outside the available suite")
    if arguments.output is not None and arguments.output.exists():
        parser.error("refusing to overwrite an existing match directory")
    for role, path in (("candidate", candidate), ("champion", champion)):
        if not path.exists():
            parser.error(f"{role} does not exist: {path}")
        print(f"{role}={path}", flush=True)
        if path.is_file():
            print(f"{role}_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(
        f"clock={arguments.base_ms}+{arguments.increment_ms}ms "
        f"paired_openings={arguments.openings}",
        flush=True,
    )
    wins = draws = losses = 0
    failures: dict[str, int] = {}
    if arguments.output is not None:
        arguments.output.mkdir(parents=True)
        manifest = dict(
            candidate=str(candidate),
            champion=str(champion),
            base_ms=arguments.base_ms,
            increment_ms=arguments.increment_ms,
            opening_start=arguments.opening_start,
            openings=arguments.openings,
        )
        # Snapshot source hashes, including directory-based experimental agents.
        for role, path in (("candidate", candidate), ("champion", champion)):
            files = (
                [path] if path.is_file() else sorted(path.glob("*.py")) + sorted(path.glob("*.npz"))
            )
            manifest[role + "_hashes"] = {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files
            }
        with (arguments.output / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)

    games: list[tuple[Path, Path, bool, str]] = []
    for fen in opening_fens()[
        arguments.opening_start : arguments.opening_start + arguments.openings
    ]:
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
        if arguments.output is not None:
            with (arguments.output / f"game-{index:03d}.pgn").open("x") as stream:
                stream.write(outcome.pgn)
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
    if arguments.output is not None:
        with (arguments.output / "result.json").open("x") as stream:
            json.dump(
                dict(wins=wins, draws=draws, losses=losses, score=score, failures=failures),
                stream,
                indent=2,
            )
    if failures:
        names = ", ".join(f"{name} {count}" for name, count in failures.items())
        raise SystemExit(f"technical failures: {names}")


if __name__ == "__main__":
    main()
