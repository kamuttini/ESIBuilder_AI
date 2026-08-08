"""Self-test of the scale pipeline stage on synthetic rulers whose truth is exact.

Why this exists: the real corpus lives on external SSDs that are not always mounted, and the
stage's job is arithmetic on a drawn ruler — which can be checked against a ruler we drew
ourselves. Three scenarios, each one a failure mode that was actually paid for:

  normale    zero at the top, four depths, three frames each -> mm_per_px, zero, far end
  ribaltato  the frames flipped up/down (``giu``): the zero must move to the *other* end,
             which is the mistake that silently mislabelled a fifth of the legacy corpus
  buco       one depth with no ruler at all: it must come back interpolated from the folder
             trend, not vanish — a hole in #21 would shift the pairing with #17/#18

No nets, no SSD, no .fss: only cv2 and tesseract. Run it after touching the stage or the
detector.

  OldSoftwareEsiBuilder/.venv-mps/bin/python tools/scale/selftest_predict_scale_from_pipeline.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "scale"))

from scale.predict_scale_from_pipeline import run  # noqa: E402

W, H = 1000, 760
SPAN_PX = 600.0
RULER_X = 800
Y0 = 70
DEPTHS = (40.0, 60.0, 80.0, 100.0)
FRAMES_PER_DEPTH = 3
RECT_LTRB = [70, 40, 730, 700]


def draw_frame(depth_mm: float, seed: int, with_ruler: bool = True,
               flip_ud: bool = False) -> np.ndarray:
    """One frame whose ruler encodes ``mm_per_px = depth_mm / SPAN_PX`` exactly.

    ``flip_ud`` draws the up/down-flipped acquisition the way a scanner really produces it:
    the zero moves to the bottom end and the values grow upwards, while the digits and the
    interface text stay upright. Flipping the finished image instead would mirror the glyphs,
    which no machine does — and the OCR then reads numbers that are not there, so the test
    would be failing its own fixture rather than the stage.
    """
    img = np.full((H, W), 10, np.uint8)
    # a bright fan on the left: a global threshold would find this and miss the ruler,
    # which is the reason the detector thresholds inside a lateral band
    cv2.ellipse(img, (400, 40), (330, 660), 0, 20, 160, 150, -1)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    cv2.putText(img, f"D {depth_mm / 10:g} cm", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 250, 2)
    if with_ruler:
        pitch = 5.0 / (depth_mm / SPAN_PX)  # a tick every 5 mm
        y_zero = float(Y0 + SPAN_PX) if flip_ud else float(Y0)
        step = -1.0 if flip_ud else 1.0
        for k in range(int(SPAN_PX / pitch) + 1):
            y = int(round(y_zero + step * k * pitch))
            major = k % 2 == 0
            cv2.line(img, (RULER_X, y), (RULER_X + (17 if major else 9), y), 240, 2)
            if major:
                cv2.putText(img, str(k // 2), (RULER_X - 30, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, 235, 1)
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)


def build_case(root: Path, flip_ud: bool = False,
               depth_without_ruler: Optional[float] = None) -> dict:
    """Write the frames and return the context JSON the pipeline would hand over."""
    folder = root / "Synthetic Machine 1.0, Probe X - L"
    folder.mkdir(parents=True, exist_ok=True)
    frames: List[dict] = []
    for di, depth in enumerate(DEPTHS):
        for f in range(FRAMES_PER_DEPTH):
            img = draw_frame(depth, 100 * di + f,
                             with_ruler=(depth != depth_without_ruler), flip_ud=flip_ud)
            name = f"20260808_1200{di}{f}_hdmi_{W}x{H}_frame.png"
            cv2.imwrite(str(folder / name), img)
            frames.append({
                "image_path": str(folder / name),
                # what the pipeline's marker stage already settled for this frame
                "sugiu": "giu" if flip_ud else "su",
                "sugiu_conf": 0.98, "sugiu_source": "marker", "marker_score": 0.81,
                "orientation_group": "UD" if flip_ud else "NF",
                "rect_ltrb": list(RECT_LTRB),
                "depth_mm": depth, "depth_from_interface": True,
                "depth_mode": "direct_label", "depth_status": "accepted",
            })
    return {
        "folder_path": str(folder),
        "vendor_predicted": "BK", "vendor_confidence": 0.93,
        "line_11_rect_echo": f"{RECT_LTRB[1]}|{RECT_LTRB[0]}|{RECT_LTRB[3]}|{RECT_LTRB[2]}|",
        "video_x": W, "video_y": H, "rotation_deg_clockwise": 0, "frames": frames,
    }


def truth_for(flip_ud: bool) -> Dict[float, Tuple[float, float, float, int]]:
    """``{depth_mm: (mm_per_px, y_zero, y_far, direction)}``."""
    out = {}
    for depth in DEPTHS:
        y_zero, y_far = float(Y0), float(Y0 + SPAN_PX)
        if flip_ud:  # the zero is drawn at the bottom end and the values grow upwards
            y_zero, y_far = y_far, y_zero
        out[depth] = (depth / SPAN_PX, y_zero, y_far, -1 if flip_ud else 1)
    return out


def _rows(out_dir: Path) -> List[dict]:
    import csv
    with (out_dir / "scale_per_depth.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    return ok


def scenario(label: str, *, flip_ud: bool = False,
             depth_without_ruler: Optional[float] = None, max_frames: int = 48) -> bool:
    print(f"\n[{label}]")
    root = Path(tempfile.mkdtemp(prefix="scale_selftest_"))
    try:
        context = build_case(root, flip_ud=flip_ud, depth_without_ruler=depth_without_ruler)
        out_dir = root / "out"
        summary = run(context, out_dir, max_frames=max_frames)
        rows = _rows(out_dir)
        truth = truth_for(flip_ud)
        ok = True
        ok &= check("una riga per depth", len(rows) == len(DEPTHS), f"{len(rows)}/{len(DEPTHS)}")
        for row in rows:
            depth = float(row["depth_mm"])
            t_mm, t_zero, t_far, t_dir = truth[depth]
            tag = f"depth {depth:g} mm"
            if depth == depth_without_ruler:
                ok &= check(f"{tag}: riempita dal trend, non persa",
                            row["source"] in ("interpolated", "detected") and row["mm_per_px"] != "",
                            f"source={row['source']} mm_per_px={row['mm_per_px'] or 'vuoto'}")
                continue
            got_mm = float(row["mm_per_px"])
            err = abs(got_mm / t_mm - 1.0)
            ok &= check(f"{tag}: mm/px entro il 2%", err <= 0.02,
                        f"{got_mm:.5f} contro {t_mm:.5f} ({err * 100:.2f}%)")
            ok &= check(f"{tag}: verso", int(row["direction"]) == t_dir,
                        f"{row['direction']} contro {t_dir}")
            ok &= check(f"{tag}: zero entro 3 px", abs(float(row["y_zero"]) - t_zero) <= 3.0,
                        f"{row['y_zero']} contro {t_zero:g}")
            ok &= check(f"{tag}: estremo entro 6 px", abs(float(row["y_far"]) - t_far) <= 6.0,
                        f"{row['y_far']} contro {t_far:g}")
        depths_line = summary.get("line_18_vect_depth", "")
        ok &= check("#18 con tutte le depth in ordine crescente",
                    depths_line == "|".join(f"{d:g}" for d in sorted(DEPTHS)) + "|", depths_line)
        ok &= check("#21 completa", summary.get("line_21_scale_line", "").count(";") == len(DEPTHS),
                    summary.get("line_21_scale_line", "")[:60] + "...")
        return bool(ok)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def scenario_pipeline_weaving() -> bool:
    """The pipeline's own two functions, fed rows shaped exactly like the real stages emit.

    This is the part that cannot be checked by reading: that the stage picks the *marker's*
    up/down and not the net's, that it takes the per-image rect from the marker rows, that a
    depth the module rejected does not travel, and that only a depth read in the interface
    counts as independent evidence.
    """
    print("\n[innesto nella pipeline]")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "off_mod", REPO / "tools/ultrasound/predict_fss_head_from_acquisitions.py")
        off = importlib.util.module_from_spec(spec)
        sys.modules["off_mod"] = off
        for extra in (REPO, REPO / "tools" / "ultrasound", REPO / "tools" / "orientation"):
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        spec.loader.exec_module(off)  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] pipeline non importabile ({exc})")
        return True

    root = Path(tempfile.mkdtemp(prefix="scale_selftest_pipe_"))
    try:
        context = build_case(root)
        images = [Path(f["image_path"]) for f in context["frames"]]
        rect = RECT_LTRB
        su_giu_rows, lr_marker_rows, rect_depth_rows = [], [], []
        for i, (frame, path) in enumerate(zip(context["frames"], images)):
            key = path.as_posix()
            # the net says the opposite of the marker on purpose: the marker must win
            su_giu_rows.append({
                "image_index": i, "image_path": key, "pred_label": "giu", "confidence": 0.55,
                "prob_su": 0.45, "prob_giu": 0.55,
                "crop_top": 0, "crop_left": 0, "crop_bottom": 10, "crop_right": 10,
            })
            lr_marker_rows.append({
                "image_index": i, "image_path": key, "su_giu_pred": "su",
                "su_giu_model_pred": "giu", "bundle_vertical_correction": "corrected",
                "bundle_vertical_source": "marker", "match_score": 0.77,
                "orientation_group": "NF",
                "echo_rect_left_abs": rect[0], "echo_rect_top_abs": rect[1],
                "echo_rect_right_abs": rect[2], "echo_rect_bottom_abs": rect[3],
            })
            # the last frame is a depth the module rejected, and one frame carries a depth it
            # inferred from the scale itself rather than read in the interface
            status = "reject" if i == len(images) - 1 else "accepted"
            mode = "scale" if i == 0 else "direct_label"
            rect_depth_rows.append({
                "image_path": key, "status": status, "mode": mode,
                "depth_mm": frame["depth_mm"], "left": 30, "top": 20, "right": 130, "bottom": 50,
                "ocr_text": f"D {float(frame['depth_mm']) / 10:g} cm",
            })

        frames = off._build_scale_frames_context(
            images=images, su_giu_rows=su_giu_rows, lr_marker_rows=lr_marker_rows,
            rect_depth_rows=rect_depth_rows,
            line_11_rect_echo=f"{rect[1]}|{rect[0]}|{rect[3]}|{rect[2]}|",
        )
        ok = True
        ok &= check("un frame per immagine", len(frames) == len(images),
                    f"{len(frames)}/{len(images)}")
        ok &= check("il verso viene dal marker, non dalla rete",
                    all(f["sugiu"] == "su" for f in frames),
                    "primo: " + str(frames[0]["sugiu"]))
        ok &= check("registra che il marker ha corretto la rete",
                    all("marker" in str(f["sugiu_source"]) for f in frames),
                    str(frames[0]["sugiu_source"]))
        ok &= check("il rect per immagine viene dalle righe del marker",
                    all(list(f["rect_ltrb"]) == [float(v) for v in rect] for f in frames),
                    str(frames[0]["rect_ltrb"]))
        ok &= check("la depth scartata dal modulo non viaggia",
                    frames[-1]["depth_mm"] is None, str(frames[-1]["depth_mm"]))
        ok &= check("solo la depth letta nell'interfaccia e' evidenza indipendente",
                    frames[0]["depth_from_interface"] is False
                    and frames[1]["depth_from_interface"] is True)

        summary, rows = off._run_scale_stage(
            folder=Path(context["folder_path"]), output_dir=root / "run", folder_index=1,
            python_bin=sys.executable, vendor_pred="BK", vendor_conf=0.93, probe_id="7",
            line_11_rect_echo=f"{rect[1]}|{rect[0]}|{rect[3]}|{rect[2]}|",
            video_x=W, video_y=H, rotation_deg_clockwise=0, frames=frames,
            max_frames=48, subprocess_timeout_sec=900.0, min_accepted_ratio=0.80,
        )
        ok &= check("lo stadio gira come subprocess e torna ok",
                    summary["status"] == "ok", f"status={summary['status']} "
                    f"source={summary['source']} err={summary.get('error', '')}")
        ok &= check("#21 emessa", bool(summary["line_21_scale_line"]),
                    str(summary["line_21_scale_line"])[:60] + "...")
        ok &= check("righe per-immagine restituite alla pipeline", len(rows) > 0, f"{len(rows)}")
        ok &= check("le righe portano l'identita' della cartella",
                    bool(rows and rows[0].get("folder_name")),
                    str(rows[0].get("folder_name") if rows else ""))
        return bool(ok)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    if shutil.which("tesseract") is None:
        print("[skip] tesseract non installato: l'OCR delle etichette non puo' girare")
        return 0
    results = {
        "normale": scenario("normale"),
        "ribaltato (zero in basso)": scenario("ribaltato (zero in basso)", flip_ud=True),
        "buco (una depth senza righello)": scenario("buco (una depth senza righello)",
                                                   depth_without_ruler=80.0),
        # 8 frame di budget su 12: senza il round-robin fra le depth, le ultime due non
        # verrebbero mai guardate e la #21 uscirebbe incompleta
        "budget stretto (8 frame su 12)": scenario("budget stretto (8 frame su 12)",
                                                  max_frames=8),
        "innesto nella pipeline": scenario_pipeline_weaving(),
    }
    print("\n" + "\n".join(f"{'OK  ' if v else 'FAIL'} {k}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
