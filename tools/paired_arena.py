"""Play a candidate/champion match with every opening used in both directions."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Final

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local
from tools.openings import DEFAULT_SUITE, opening_fens

SEARCH_LOG_PATTERN: Final = re.compile(
    r"depth=(?P<depth>\d+)\s+score=-?\d+\s+nodes=(?P<nodes>\d+)\s+"
    r"time=(?P<time>\d+(?:\.\d+)?)ms"
)


def summarize_search_log(log: str) -> dict[str, int | float | str | None]:
    """Summarize the partial observed stderr tail from one agent log."""
    matches = list(SEARCH_LOG_PATTERN.finditer(log))
    if not matches:
        return {
            "coverage": "partial_observed",
            "moves_logged": 0,
            "mean_completed_depth": None,
            "total_reported_nodes": 0,
            "mean_move_ms": None,
        }
    depths = [int(match.group("depth")) for match in matches]
    nodes = [int(match.group("nodes")) for match in matches]
    times = [float(match.group("time")) for match in matches]
    return {
        "coverage": "partial_observed",
        "moves_logged": len(matches),
        "mean_completed_depth": sum(depths) / len(depths),
        "total_reported_nodes": sum(nodes),
        "mean_move_ms": sum(times) / len(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=2_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--suite", choices=("legacy", "validation"), default=DEFAULT_SUITE)
    parser.add_argument("--unbuffered-logs", action="store_true", help="Launch both local Python runners with -u for timely diagnostic logs")
    parser.add_argument("--openings", type=int, default=12)
    parser.add_argument("--opening-start", type=int, default=0)
    parser.add_argument("--output", type=Path, help="New directory for reproducible game records")
    arguments = parser.parse_args()

    suite_fens = opening_fens(arguments.suite)
    candidate = arguments.candidate.resolve()
    champion = arguments.champion.resolve()
    if not 1 <= arguments.openings <= len(suite_fens):
        parser.error(f"--openings must be between 1 and {len(suite_fens)}")
    if arguments.opening_start < 0 or arguments.opening_start + arguments.openings > len(suite_fens):
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
        f"suite={arguments.suite} paired_openings={arguments.openings}",
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
            suite=arguments.suite,
            unbuffered_logs=arguments.unbuffered_logs,
            opening_start=arguments.opening_start,
            openings=arguments.openings,
            starting_fens=list(
                suite_fens[arguments.opening_start : arguments.opening_start + arguments.openings]
            ),
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
    for fen in suite_fens[
        arguments.opening_start : arguments.opening_start + arguments.openings
    ]:
        games.append((candidate, champion, True, fen))
        games.append((champion, candidate, False, fen))

    for index, (white, black, candidate_white, fen) in enumerate(games, start=1):
        white_agent = local(white)
        black_agent = local(black)
        if arguments.unbuffered_logs:
            # Opt-in local instrumentation; neither agent source nor referee
            # rules change. Record the launch mode in the match manifest.
            white_agent.command.insert(1, "-u")
            black_agent.command.insert(1, "-u")
        outcome = play_match(
            white_agent,
            black_agent,
            arguments.base_ms,
            arguments.increment_ms,
            start_fen=fen,
        )
        if arguments.output is not None:
            with (arguments.output / f"game-{index:03d}.pgn").open("x") as stream:
                stream.write(outcome.pgn)
            with (arguments.output / f"game-{index:03d}-white.log").open("x") as stream:
                stream.write(white_agent.stderr_tail)
            with (arguments.output / f"game-{index:03d}-black.log").open("x") as stream:
                stream.write(black_agent.stderr_tail)
            summary = {
                "game": index,
                "notice": (
                    "Agent stderr tails are partial observed logs; buffered final lines may be "
                    "missing after process termination, and moves_logged is not total moves."
                ),
                "white": {
                    "color": "white",
                    "role": "candidate" if candidate_white else "champion",
                    **summarize_search_log(white_agent.stderr_tail),
                },
                "black": {
                    "color": "black",
                    "role": "champion" if candidate_white else "candidate",
                    **summarize_search_log(black_agent.stderr_tail),
                },
            }
            with (arguments.output / f"game-{index:03d}-summary.json").open("x") as stream:
                json.dump(summary, stream, indent=2)
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
