from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "corpus_sources" / "zimu_20261016.json"
DEFAULT_OUTPUT = ROOT / "data" / "transcripts" / "_candidates_20261016"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve public-source SRT files for local use; never commits transcripts.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--speaker", choices=["nailv", "nana7mi", "azi"])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["entries"]
    root = args.outdir.resolve()
    successes = 0
    failures = 0
    for row in records:
        if args.speaker and row["speaker_id"] != args.speaker:
            continue
        target = (root / row["relative_path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Unsafe relative path in manifest")
        expected_url = f"https://api.zimu.live:7443/clips/{int(row['clip_id'])}/srt"
        if row["source_url"] != expected_url:
            raise ValueError(f"Unexpected source URL: {row['source_url']}")
        try:
            if target.exists():
                contents = target.read_bytes()
            elif args.verify_only:
                raise FileNotFoundError(target)
            else:
                request = urllib.request.Request(expected_url, headers={"User-Agent": "StreamRAG provenance fetcher/1.0"})
                with urllib.request.urlopen(request, timeout=25) as response:
                    contents = response.read()
                if len(contents) > 5_000_000:
                    raise ValueError("Unexpected oversized SRT")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
            actual = hashlib.sha256(contents).hexdigest()
            if actual != row["sha256"]:
                raise ValueError("SHA256 does not match archived source snapshot; do not silently replace.")
            if b"-->" not in contents or len(contents) < 100:
                raise ValueError("File does not look like a valid SRT")
            successes += 1
            print(f"OK {row['relative_path']} ({len(contents)} bytes)")
        except Exception as exc:
            failures += 1
            print(f"FAIL {row['relative_path']}: {exc}", file=sys.stderr)
    print(f"verified={successes} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
