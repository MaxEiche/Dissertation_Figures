"""Verify all required input files against the repository's SHA-256 manifest."""

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "DATA_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    directory_names = {}
    failures = []
    for record in manifest["records"]:
        relative = Path(record["path"])
        path = root
        for part in relative.parts:
            if path not in directory_names:
                directory_names[path] = {entry.name for entry in path.iterdir()} if path.is_dir() else set()
            if part not in directory_names[path]:
                failures.append(f"Missing file or capitalization mismatch: {record['path']}")
                break
            path /= part
        else:
            if path.stat().st_size != record["bytes"]:
                failures.append(f"Size mismatch: {record['path']}")
                continue
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != record["sha256"]:
                failures.append(f"SHA-256 mismatch: {record['path']}")
    for failure in failures:
        print(failure)
    print(f"Verified {len(manifest['records']) - len(failures)}/{len(manifest['records'])} required input files.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
