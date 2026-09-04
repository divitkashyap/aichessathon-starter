"""Package an explicit flat-file challenger without replacing any archive."""

import argparse
import ast
import hashlib
import zipfile
from pathlib import Path


def build(source: Path, destination: Path, names: list[str]) -> str:
    if destination.exists():
        raise FileExistsError(f"refusing to replace {destination}")
    if "agent.py" not in names or len(set(names)) != len(names):
        raise ValueError("members must be unique and include agent.py")
    for name in names:
        if Path(name).name != name or Path(name).suffix not in (".py", ".npz"):
            raise ValueError(f"unsupported flat member: {name}")
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
        if name.endswith(".py"):
            ast.parse((source / name).read_text())
    size = sum((source / name).stat().st_size for name in names)
    if size > 50_000_000:
        raise ValueError("uncompressed archive exceeds 50 MB")
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (source / name).read_bytes())
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"archive={destination} uncompressed_bytes={size} sha256={digest}")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--files", required=True, nargs="+")
    args = parser.parse_args()
    build(args.source, args.out, args.files)


if __name__ == "__main__":
    main()
