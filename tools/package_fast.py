"""Build the isolated compiled-search challenger as a submission-shaped zip."""

from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path

from harness.rules import MAX_UNZIPPED_BYTES

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = (
    (ROOT / "challengers" / "fast" / "agent.py", "agent.py"),
    (ROOT / "fastcore.py", "fastcore.py"),
    (ROOT / "fastsearch.py", "fastsearch.py"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("candidate-fast-v2.zip"))
    arguments = parser.parse_args()

    agent_source = MEMBERS[0][0].read_text()
    ast.parse(agent_source)
    if "parents[" in agent_source or "sys.path" in agent_source:
        raise SystemExit("submission agent must not depend on its parent directory layout")

    with zipfile.ZipFile(arguments.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, destination in MEMBERS:
            archive.write(source, destination)
    unzipped = sum(source.stat().st_size for source, _ in MEMBERS)
    if unzipped > MAX_UNZIPPED_BYTES:
        raise SystemExit(f"package is too large: {unzipped:,} bytes")
    print(f"{arguments.out} ({arguments.out.stat().st_size:,} bytes, {unzipped:,} unzipped)")
    for _, destination in MEMBERS:
        print(f"  {destination}")


if __name__ == "__main__":
    main()
