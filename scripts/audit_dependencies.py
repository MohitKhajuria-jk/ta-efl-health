"""Query OSV for the exact Python lockfile; failed requests never count as clean audits."""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://api.osv.dev/v1/querybatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional report file")
    args = parser.parse_args()
    packages = []
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            packages.append({"package": {"name": match[1], "ecosystem": "PyPI"}, "version": match[2]})
    if not packages:
        raise ValueError("No pinned dependencies found")
    request = Request(SOURCE, json.dumps({"queries": packages}).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "source": SOURCE,
              "checked": len(packages), "complete": False}
    try:
        with urlopen(request, timeout=30) as response:
            results = json.load(response)["results"]
        if len(results) != len(packages):
            raise ValueError("Incomplete advisory response")
        findings = [{"package": package["package"]["name"], "version": package["version"],
                     "vulnerabilities": result["vulns"]}
                    for package, result in zip(packages, results) if result.get("vulns")]
        report.update(complete=True, findings=findings)
        code = 1 if findings else 0
    except (URLError, TimeoutError, ValueError, KeyError) as exc:
        report["error"] = str(exc)
        code = 2
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return code


if __name__ == "__main__":
    sys.exit(main())
