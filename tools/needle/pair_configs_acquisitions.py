#!/usr/bin/env python3
"""Pair each legacy configuration (SSD_esi1_n1) with the acquisition it came from (SSD_esi1_n3).

No such mapping exists, so it is rebuilt here. Names alone are not enough: they are similar
enough to look right and different enough to be wrong -- a "FlexFocus 400" matches a
"Flex Focus 500" by string distance. So the name only proposes candidates, and each candidate
is then checked against something the name cannot fake: the frame size the configuration
declares in #09/#10 must actually occur among the acquisition's images.

Configurations the operators marked as wrong in the folder name are dropped (NON USARE,
SBAGLIATO, GRIGLIA STRANA, MODIFICA PER CAREGGI NO, FORSEEEEEEE): using one as ground truth
would mean learning its mistake.

Output: pairs.csv with one row per configuration, its best acquisition, the evidence behind
the choice and a confidence label, plus unmatched.csv for what is left over.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guides_geometry import read_setup  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information"}

# What the operators wrote on the folder when a configuration must not be used.
BAD_NAME = re.compile(
    r"non usare|sbagliat|griglia strana|modifica per careggi no|forseee|no esi builder",
    re.IGNORECASE,
)

FIELDS = [
    "config", "config_path", "setup_file", "probe_type", "kit_id", "n_angles",
    "video_w", "video_h", "acquisition", "confidence", "name_score",
    "size_match_pct", "images_checked", "candidates",
]


def normalise(name: str) -> str:
    name = re.sub(r"^\d+[.\s]*", "", name.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name).split())


def image_sizes(folder: Path, limit: int) -> Counter:
    """Frame sizes found in an acquisition, read from the headers only."""
    from PIL import Image

    sizes: Counter = Counter()
    seen = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if Path(name).suffix.lower() not in IMAGE_EXTS or name.startswith("._"):
                continue
            try:
                with Image.open(Path(dirpath) / name) as img:
                    sizes[img.size] += 1
            except Exception:
                continue
            seen += 1
            if seen >= limit:
                return sizes
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs-root", type=Path,
                        default=Path("/Volumes/SSD_esi1_n1"))
    parser.add_argument("--acquisitions-root", type=Path,
                        default=Path("/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--images-per-acquisition", type=int, default=60)
    parser.add_argument("--keep-flagged", action="store_true",
                        help="Keep the configurations the operators marked as wrong.")
    args = parser.parse_args()

    configs: List[Tuple[Path, Path]] = []
    dropped: List[str] = []
    for entry in sorted(args.configs_root.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        setups = sorted((entry / "DB_setup").glob("setup_*.fss"))
        if not setups:
            continue
        if BAD_NAME.search(entry.name) and not args.keep_flagged:
            dropped.append(entry.name)
            continue
        configs.append((entry, setups[0]))
    print(f"configurazioni: {len(configs)}  (scartate per nome: {len(dropped)})")

    acquisitions = [
        d for d in sorted(args.acquisitions_root.iterdir())
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")
    ]
    print(f"acquisizioni: {len(acquisitions)}")

    norm_acq = {normalise(a.name): a for a in acquisitions}
    keys = list(norm_acq)

    size_cache: Dict[str, Counter] = {}

    def sizes_of(folder: Path) -> Counter:
        if folder.name not in size_cache:
            size_cache[folder.name] = image_sizes(folder, args.images_per_acquisition)
        return size_cache[folder.name]

    rows: List[Dict[str, object]] = []
    unmatched: List[Dict[str, object]] = []

    for index, (config_dir, setup_file) in enumerate(configs, 1):
        setup = read_setup(setup_file)
        if setup is None:
            unmatched.append({"config": config_dir.name, "reason": "setup illeggibile"})
            continue
        target = tuple(setup.video_size)

        candidates = difflib.get_close_matches(
            normalise(config_dir.name), keys, n=args.candidates, cutoff=0.45
        )
        best: Optional[Tuple[float, float, int, Path]] = None
        for key in candidates:
            folder = norm_acq[key]
            sizes = sizes_of(folder)
            checked = sum(sizes.values())
            hit = sizes.get(target, 0)
            share = hit / checked if checked else 0.0
            name_score = difflib.SequenceMatcher(
                None, normalise(config_dir.name), key
            ).ratio()
            # the frame size decides, the name only breaks ties
            score = (share, name_score)
            if best is None or score > (best[0], best[1]):
                best = (share, name_score, checked, folder)

        if best is None or best[0] <= 0.0:
            unmatched.append({
                "config": config_dir.name,
                "reason": "nessun candidato con la risoluzione dichiarata",
                "video": f"{target[0]}x{target[1]}",
                "candidates": " | ".join(norm_acq[k].name for k in candidates),
            })
            continue

        share, name_score, checked, folder = best
        if share >= 0.5 and name_score >= 0.6:
            confidence = "alta"
        elif share >= 0.2 or name_score >= 0.6:
            confidence = "media"
        else:
            confidence = "bassa"

        rows.append({
            "config": config_dir.name,
            "config_path": str(config_dir),
            "setup_file": str(setup_file),
            "probe_type": setup.probe_type,
            "kit_id": setup.kit_id,
            "n_angles": setup.n_angles,
            "video_w": target[0], "video_h": target[1],
            "acquisition": str(folder),
            "confidence": confidence,
            "name_score": f"{name_score:.3f}",
            "size_match_pct": f"{share * 100:.1f}",
            "images_checked": checked,
            "candidates": " | ".join(norm_acq[k].name for k in candidates),
        })
        if index % 25 == 0:
            print(f"  {index}/{len(configs)}…", flush=True)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "unmatched.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["config", "reason", "video", "candidates"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(unmatched)
    (out / "dropped_by_name.txt").write_text("\n".join(dropped), encoding="utf-8")

    counts = Counter(str(r["confidence"]) for r in rows)
    print(f"\nappaiate: {len(rows)}  |  non appaiate: {len(unmatched)}")
    for level in ("alta", "media", "bassa"):
        print(f"  confidenza {level}: {counts.get(level, 0)}")
    print(f"output in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
