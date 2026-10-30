"""Compare protected research files against the pre-modernization SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    research = {k:v for k,v in expected.items() if k.startswith(("storage/", "data/", "config/"))}
    changed, missing = [], []
    for relative, digest in research.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest contains a path outside the project")
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    print(json.dumps({"checked":len(research), "changed":changed, "missing":missing,
                      "experiments":len(list((root/"storage/experiments").glob("*.json"))),
                      "models":len(list((root/"storage/models").glob("*/model.joblib")))}, indent=2))
    return int(bool(changed or missing))


if __name__ == "__main__":
    raise SystemExit(main())
