"""Build the flat compiled-search submission archive reproducibly."""

from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path

from harness.rules import MAX_UNZIPPED_BYTES

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    (ROOT / "challengers" / "fast" / "agent.py", "agent.py"),
    (ROOT / "fastcore.py", "fastcore.py"),
    (ROOT / "fastsearch.py", "fastsearch.py"),
)


def build(destination: Path) -> None:
    missing = [str(source) for source, _ in SOURCES if not source.is_file()]
    if missing:
        raise SystemExit(f"missing submission files: {', '.join(missing)}")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {destination}")

    agent_source = SOURCES[0][0].read_text()
    ast.parse(agent_source)
    if "parents[" in agent_source:
        raise SystemExit("submission agent must not depend on the repository directory layout")

    uncompressed = sum(source.stat().st_size for source, _ in SOURCES)
    if uncompressed > MAX_UNZIPPED_BYTES:
        raise SystemExit(
            f"submission expands to {uncompressed:,} bytes, above "
            f"the {MAX_UNZIPPED_BYTES:,}-byte limit"
        )

    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for source, member in SOURCES:
            archive.write(source, member)

    print(
        f"built={destination} compressed={destination.stat().st_size:,} "
        f"uncompressed={uncompressed:,}"
    )
    for source, member in SOURCES:
        print(f"{member} <- {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    build(arguments.out)


if __name__ == "__main__":
    main()
