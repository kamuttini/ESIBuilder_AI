#!/usr/bin/env python3
"""Copy the legacy .fss and .ndg files off the external volume, once.

SSD_esi1_n1 unmounts by itself, repeatedly, and every analysis that touches it dies halfway
through with results that look like a finding until you notice the disk is gone. The files that
matter are a few megabytes of text and small binaries: there is no reason to read them from a
volume that disappears.

The acquisition frames stay where they are -- those are hundreds of gigabytes -- so this only
removes the dependency for the part that can be removed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs-root", type=Path, default=Path("/Volumes/SSD_esi1_n1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = []
    copied = {"fss": 0, "ndg": 0, "grid": 0}

    for entry in sorted(args.configs_root.iterdir()):
        setup_dir = entry / "DB_setup"
        if not entry.is_dir() or not setup_dir.is_dir():
            continue
        target = out / entry.name / "DB_setup"
        target.mkdir(parents=True, exist_ok=True)
        files = []
        for path in sorted(setup_dir.iterdir()):
            suffix = path.suffix.lower().lstrip(".")
            if suffix not in ("fss", "ndg", "grid"):
                continue
            try:
                shutil.copy2(path, target / path.name)
            except OSError:
                continue
            copied[suffix] = copied.get(suffix, 0) + 1
            files.append(path.name)
        index.append({"config": entry.name, "files": files,
                      "origin": str(entry)})

    (out / "index.json").write_text(
        json.dumps({"configs": index, "counts": copied}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"configurazioni copiate: {len(index)}")
    for kind, count in sorted(copied.items()):
        print(f"  .{kind}: {count}")
    print(f"destinazione: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
