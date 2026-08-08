"""Render the project monitor into a single self-contained HTML file.

Reads the normalised store produced by `collect_metrics.py` and injects it into
`dashboard_template.html`. No external assets, no CDN, no server: the output can be
opened by double click on Windows as well as macOS.

Usage:
    python3 tools/monitor/build_dashboard.py --store artifacts/71_monitor
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_registry import METRICS, PIPELINE_STEPS, STATUS_LEVELS  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
PLACEHOLDER = "/*__DATA__*/"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact(records: List[Dict[str, Any]], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dictionary-encode the records so the payload stays small."""
    run_index = {run["run_id"]: i for i, run in enumerate(runs)}
    dicts: Dict[str, List[str]] = {"steps": [], "metrics": [], "vendors": [], "splits": [], "kinds": [], "scopes": []}
    lookup: Dict[str, Dict[str, int]] = {key: {} for key in dicts}

    def idx(kind: str, value: Optional[str]) -> int:
        if value is None:
            return -1
        table = lookup[kind]
        if value not in table:
            table[value] = len(dicts[kind])
            dicts[kind].append(value)
        return table[value]

    rows: List[List[Any]] = []
    for record in records:
        run_id = record["run_id"]
        if run_id not in run_index:
            continue
        rows.append(
            [
                idx("steps", record["step_id"]),
                run_index[run_id],
                idx("metrics", record["metric"]),
                idx("vendors", record.get("vendor")),
                round(float(record["value"]), 6),
                record.get("n"),
                idx("splits", record.get("split") or "test"),
                idx("kinds", record.get("run_kind") or "eval"),
                idx("scopes", record.get("scope") or "full"),
            ]
        )
    return {"dict": dicts, "records": rows}


def build_payload(store: Path, artifacts_root: str) -> Dict[str, Any]:
    records = read_jsonl(store / "metrics.jsonl")
    runs = read_jsonl(store / "runs.jsonl")
    reasons = read_jsonl(store / "review_reasons.jsonl")
    report_path = store / "collect_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    packed = compact(records, runs)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "artifacts_root": artifacts_root,
        "steps": PIPELINE_STEPS,
        "metrics_meta": METRICS,
        "status_levels": STATUS_LEVELS,
        "runs": runs,
        "reasons": reasons,
        "report": {
            "records": report.get("records", len(records)),
            "runs": report.get("runs", len(runs)),
            "warnings": report.get("warnings", []),
        },
        "dict": packed["dict"],
        "records": packed["records"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Genera la dashboard HTML del monitor di progetto")
    parser.add_argument("--store", default="artifacts/71_monitor", help="cartella con metrics.jsonl / runs.jsonl")
    parser.add_argument("--out", default=None, help="file HTML di output (default: <store>/index.html)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    store = Path(args.store).resolve()
    if not (store / "metrics.jsonl").exists():
        parser.error(f"store senza metrics.jsonl: {store}. Lancia prima collect_metrics.py")
    out = Path(args.out).resolve() if args.out else store / "index.html"

    report_path = store / "collect_report.json"
    artifacts_root = "artifacts"
    if report_path.exists():
        artifacts_root = json.loads(report_path.read_text(encoding="utf-8")).get("artifacts_root", artifacts_root)

    payload = build_payload(store, artifacts_root)
    # `</script>` inside a JSON string would close the host script tag early.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise RuntimeError(f"placeholder {PLACEHOLDER} non trovato in {TEMPLATE}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, blob), encoding="utf-8")

    if not args.quiet:
        size_kb = out.stat().st_size / 1024
        print(f"dashboard: {out}  ({size_kb:.0f} KB, {len(payload['records'])} misure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
