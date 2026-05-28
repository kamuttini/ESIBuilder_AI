#!/usr/bin/env python3
"""Route line #04 PROBETYPE from line #03 ID_PROBE.

Rule requested by workflow:
- probe type 1/2 stay univocal
- probe type 3/4 are grouped as "3-4" (requires second-stage model to split L vs T)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROBE_TYPE_LINEAR = 1
PROBE_TYPE_CONVEX = 2
PROBE_TYPE_TRANS_L = 3
PROBE_TYPE_TRANS_T = 4


def _normalize_probe_id(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        # Handles tokens like "1.0" from csv exports.
        return str(int(float(raw)))
    except ValueError:
        return raw


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(frozen=True)
class ProbeTypeDecision:
    probe_id: str
    probe_name: str
    type_value: str
    resolved_type_int: Optional[int]
    candidate_type_ints: Tuple[int, ...]
    needs_secondary_model: bool
    strategy: str
    source: str
    observed_type_shares: Dict[int, float]


def _decide_probe_type(
    probe_id: str,
    probe_name: str,
    observed_shares: Dict[int, float],
    summary_primary_type: Optional[int],
) -> ProbeTypeDecision:
    observed = {k: float(v) for k, v in observed_shares.items() if float(v) > 0.0}
    observed_types = sorted(observed.keys())

    # 3/4 always grouped by design.
    if (PROBE_TYPE_TRANS_L in observed) or (PROBE_TYPE_TRANS_T in observed):
        return ProbeTypeDecision(
            probe_id=probe_id,
            probe_name=probe_name,
            type_value="3-4",
            resolved_type_int=None,
            candidate_type_ints=(PROBE_TYPE_TRANS_L, PROBE_TYPE_TRANS_T),
            needs_secondary_model=True,
            strategy="biplane_group",
            source="evidence" if observed else "summary",
            observed_type_shares=observed,
        )

    # Single-stage univocal classes.
    if PROBE_TYPE_LINEAR in observed and PROBE_TYPE_CONVEX not in observed:
        return ProbeTypeDecision(
            probe_id=probe_id,
            probe_name=probe_name,
            type_value="1",
            resolved_type_int=PROBE_TYPE_LINEAR,
            candidate_type_ints=(PROBE_TYPE_LINEAR,),
            needs_secondary_model=False,
            strategy="univocal_linear",
            source="evidence",
            observed_type_shares=observed,
        )

    if PROBE_TYPE_CONVEX in observed and PROBE_TYPE_LINEAR not in observed:
        return ProbeTypeDecision(
            probe_id=probe_id,
            probe_name=probe_name,
            type_value="2",
            resolved_type_int=PROBE_TYPE_CONVEX,
            candidate_type_ints=(PROBE_TYPE_CONVEX,),
            needs_secondary_model=False,
            strategy="univocal_convex",
            source="evidence",
            observed_type_shares=observed,
        )

    # Rare/incoherent case (1 and 2 both present in evidence).
    if PROBE_TYPE_LINEAR in observed and PROBE_TYPE_CONVEX in observed:
        winner = PROBE_TYPE_LINEAR
        if observed.get(PROBE_TYPE_CONVEX, 0.0) > observed.get(PROBE_TYPE_LINEAR, 0.0):
            winner = PROBE_TYPE_CONVEX
        return ProbeTypeDecision(
            probe_id=probe_id,
            probe_name=probe_name,
            type_value=str(winner),
            resolved_type_int=winner,
            candidate_type_ints=(PROBE_TYPE_LINEAR, PROBE_TYPE_CONVEX),
            needs_secondary_model=False,
            strategy="incoherent_1_2_fallback_majority",
            source="evidence",
            observed_type_shares=observed,
        )

    # Fallback from summary primary.
    if summary_primary_type in {PROBE_TYPE_LINEAR, PROBE_TYPE_CONVEX}:
        return ProbeTypeDecision(
            probe_id=probe_id,
            probe_name=probe_name,
            type_value=str(summary_primary_type),
            resolved_type_int=summary_primary_type,
            candidate_type_ints=(summary_primary_type,),
            needs_secondary_model=False,
            strategy="summary_primary_fallback",
            source="summary",
            observed_type_shares=observed,
        )
    if summary_primary_type in {PROBE_TYPE_TRANS_L, PROBE_TYPE_TRANS_T}:
        return ProbeTypeDecision(
            probe_id=probe_id,
            probe_name=probe_name,
            type_value="3-4",
            resolved_type_int=None,
            candidate_type_ints=(PROBE_TYPE_TRANS_L, PROBE_TYPE_TRANS_T),
            needs_secondary_model=True,
            strategy="summary_biplane_group_fallback",
            source="summary",
            observed_type_shares=observed,
        )

    return ProbeTypeDecision(
        probe_id=probe_id,
        probe_name=probe_name,
        type_value="UNKNOWN",
        resolved_type_int=None,
        candidate_type_ints=tuple(observed_types),
        needs_secondary_model=False,
        strategy="unknown",
        source="none",
        observed_type_shares=observed,
    )


class ProbeTypeRouter:
    def __init__(self, decisions: Dict[str, ProbeTypeDecision]) -> None:
        self._decisions = decisions

    @classmethod
    def from_csvs(cls, probe_summary_csv: Path, probe_type_evidence_csv: Path) -> "ProbeTypeRouter":
        summary: Dict[str, Dict[str, object]] = {}
        if probe_summary_csv.is_file():
            with probe_summary_csv.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    pid = _normalize_probe_id(row.get("probe_id"))
                    if not pid:
                        continue
                    primary_raw = str(row.get("primary_probe_type_id") or "").strip()
                    primary_int: Optional[int] = None
                    if primary_raw:
                        try:
                            primary_int = int(float(primary_raw))
                        except ValueError:
                            primary_int = None
                    summary[pid] = {
                        "probe_name": str(row.get("probe_name") or "").strip(),
                        "summary_primary_type": primary_int,
                    }

        shares_by_probe: Dict[str, Dict[int, float]] = {}
        if probe_type_evidence_csv.is_file():
            with probe_type_evidence_csv.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    pid = _normalize_probe_id(row.get("probe_id"))
                    if not pid:
                        continue
                    try:
                        probe_type = int(float(str(row.get("probe_type_id") or "").strip()))
                    except ValueError:
                        continue
                    share = _safe_float(row.get("share_within_probe"), default=0.0)
                    if share <= 0.0:
                        # Fallback if share is missing.
                        share = _safe_float(row.get("samples"), default=0.0)
                    shares_by_probe.setdefault(pid, {})
                    shares_by_probe[pid][probe_type] = shares_by_probe[pid].get(probe_type, 0.0) + float(share)

        decisions: Dict[str, ProbeTypeDecision] = {}
        all_probe_ids = sorted(set(summary.keys()) | set(shares_by_probe.keys()), key=lambda x: (int(x) if x.isdigit() else 999999, x))
        for pid in all_probe_ids:
            s = summary.get(pid, {})
            decision = _decide_probe_type(
                probe_id=pid,
                probe_name=str(s.get("probe_name") or ""),
                observed_shares=shares_by_probe.get(pid, {}),
                summary_primary_type=s.get("summary_primary_type"),  # type: ignore[arg-type]
            )
            decisions[pid] = decision
        return cls(decisions=decisions)

    def resolve(self, probe_id: object) -> ProbeTypeDecision:
        pid = _normalize_probe_id(probe_id)
        if not pid:
            return ProbeTypeDecision(
                probe_id="",
                probe_name="",
                type_value="UNKNOWN",
                resolved_type_int=None,
                candidate_type_ints=tuple(),
                needs_secondary_model=False,
                strategy="missing_probe_id",
                source="none",
                observed_type_shares={},
            )
        if pid in self._decisions:
            return self._decisions[pid]
        return ProbeTypeDecision(
            probe_id=pid,
            probe_name="",
            type_value="UNKNOWN",
            resolved_type_int=None,
            candidate_type_ints=tuple(),
            needs_secondary_model=False,
            strategy="probe_id_not_in_router",
            source="none",
            observed_type_shares={},
        )

    def iter_decisions(self) -> Iterable[ProbeTypeDecision]:
        return self._decisions.values()


def _write_map_csv(path: Path, rows: Iterable[ProbeTypeDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "probe_id",
                "probe_name",
                "probe_type_value",
                "resolved_type_int",
                "candidate_type_ints",
                "needs_secondary_model",
                "strategy",
                "source",
                "observed_type_shares_json",
            ]
        )
        for d in rows:
            writer.writerow(
                [
                    d.probe_id,
                    d.probe_name,
                    d.type_value,
                    "" if d.resolved_type_int is None else str(d.resolved_type_int),
                    ",".join(str(x) for x in d.candidate_type_ints),
                    "1" if d.needs_secondary_model else "0",
                    d.strategy,
                    d.source,
                    json.dumps(d.observed_type_shares, ensure_ascii=False, sort_keys=True),
                ]
            )


def _annotate_csv(input_csv: Path, output_csv: Path, router: ProbeTypeRouter, probe_id_column: str) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        base_fields = list(rows[0].keys()) if rows else []

    extra_fields = [
        "probe_type_value",
        "probe_type_resolved_int",
        "probe_type_candidate_ints",
        "probe_type_needs_secondary_model",
        "probe_type_strategy",
        "probe_type_source",
    ]
    fields = base_fields + [f for f in extra_fields if f not in base_fields]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            decision = router.resolve(row.get(probe_id_column))
            out = dict(row)
            out["probe_type_value"] = decision.type_value
            out["probe_type_resolved_int"] = "" if decision.resolved_type_int is None else str(decision.resolved_type_int)
            out["probe_type_candidate_ints"] = ",".join(str(x) for x in decision.candidate_type_ints)
            out["probe_type_needs_secondary_model"] = "1" if decision.needs_secondary_model else "0"
            out["probe_type_strategy"] = decision.strategy
            out["probe_type_source"] = decision.source
            writer.writerow(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Route probe_id -> probe_type value/group for FSS line #04.")
    p.add_argument(
        "--probe-summary-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_model_to_type_summary_2026-03-31.csv"),
        help="CSV summary probe->type (primary/status).",
    )
    p.add_argument(
        "--probe-type-evidence-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_id_probe_type_evidence_from_line16_manifest_2026-03-30.csv"),
        help="CSV evidence probe_id+probe_type_id+share.",
    )
    p.add_argument("--probe-id", type=str, default="", help="Single probe_id to resolve.")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional input CSV to annotate with probe_type routing columns.",
    )
    p.add_argument(
        "--input-probe-column",
        type=str,
        default="predicted_probe_id",
        help="Probe id column name in --input-csv.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output annotated CSV for --input-csv.",
    )
    p.add_argument(
        "--output-map-csv",
        type=Path,
        default=Path("artifacts/60_metadata/probe_id_to_probe_type_router.csv"),
        help="Output full router map CSV.",
    )
    p.add_argument(
        "--print-json",
        action="store_true",
        help="Print decision in JSON for --probe-id.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    router = ProbeTypeRouter.from_csvs(
        probe_summary_csv=args.probe_summary_csv.expanduser().resolve(),
        probe_type_evidence_csv=args.probe_type_evidence_csv.expanduser().resolve(),
    )

    if args.output_map_csv:
        output_map_csv = args.output_map_csv.expanduser().resolve()
        _write_map_csv(output_map_csv, router.iter_decisions())
        print(f"Output map CSV: {output_map_csv}")

    if args.probe_id:
        decision = router.resolve(args.probe_id)
        if args.print_json:
            print(json.dumps(asdict(decision), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"probe_id={decision.probe_id}")
            print(f"probe_name={decision.probe_name}")
            print(f"probe_type_value={decision.type_value}")
            print(
                "resolved_type_int="
                + ("" if decision.resolved_type_int is None else str(decision.resolved_type_int))
            )
            print(f"candidate_type_ints={','.join(str(x) for x in decision.candidate_type_ints)}")
            print(f"needs_secondary_model={'1' if decision.needs_secondary_model else '0'}")
            print(f"strategy={decision.strategy}")
            print(f"source={decision.source}")

    if args.input_csv:
        input_csv = args.input_csv.expanduser().resolve()
        output_csv = args.output_csv.expanduser().resolve() if args.output_csv else input_csv.with_name(
            input_csv.stem + "_with_probe_type.csv"
        )
        _annotate_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            router=router,
            probe_id_column=args.input_probe_column,
        )
        print(f"Output annotated CSV: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
