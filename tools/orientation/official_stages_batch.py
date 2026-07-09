#!/usr/bin/env python3
"""Resumable runner for the OFFICIAL pipeline stages needed by the marker review.

Re-uses the exact functions/models of `tools/ultrasound/predict_fss_head_from_acquisitions.py`
(imported as a module, so the logic is identical) but with image-level
checkpointing, so it can be stopped/resumed at any moment:

Per folder (same image set as the marker runner: dominant resolution only):
1. vendor: CNN mean-softmax on a sample (official `_predict_mean_probs`)
2. rect per-image: routing (vendor-specialized if conf >= threshold, e.g. BK)
   -> `_predict_rect_boxes_abs` (official net + coordinate post-process)
3. SU/GIU per-image on the per-image rect crops (official classifier)
4. line #11: component-wise median of per-image boxes (official `_median_box`)
   + red-rect override (`compute_rect_red_pipeline`), fallback median
5. line #13 RECT_NAME_ECHO: historical resolver (vendor + video size from
   filenames). NOTE: the per-vendor line13 nets / template post-process of the
   official pipeline are NOT run here (documented simplification).

Rotation normalization (OSD) is NOT applied (assumed 0), consistent with the
marker runner on the same volume.

Outputs in --output-dir (default <marker-run-dir>/official_stages):
- official_per_image.csv (rect box + su/giu per image; appended incrementally)
- official_folder.csv (vendor, line #11 + method, line #13 + source)
- state.json (vendor/routing cache)

Example:
  python3 tools/orientation/official_stages_batch.py \
    --marker-run-dir artifacts/43_orientation_envelopes_ssd_n3_trial/run2 \
    --dataset-root "/Volumes/SSD_esi1_n3/ACQUISITION ELABORATION" \
    --resume --time-budget 0
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPE_DIR = REPO_ROOT / "artifacts/10_active_pipeline/pipeline_fss_head"

IMAGE_FIELDS = [
    "folder", "image_id", "rect_box_abs", "sugiu_label", "sugiu_conf",
    "prob_su", "prob_giu", "crop_box_abs", "crop_source",
]
FOLDER_FIELDS = [
    "folder", "vendor_pred", "vendor_conf", "rect_source", "rect_checkpoint",
    "line11_median", "line11_final", "line11_method", "line13_text",
    "line13_source", "line13_support", "video_x", "video_y", "sugiu_majority",
]


def _import_official():
    sys.path.insert(0, str(REPO_ROOT / "tools/ultrasound"))
    spec = importlib.util.spec_from_file_location(
        "predict_fss_head_from_acquisitions",
        REPO_ROOT / "tools/ultrasound/predict_fss_head_from_acquisitions.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _append_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file() or not path.stat().st_size
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def _xyxy_to_tlbr_text(box: Tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    return f"{int(round(y1))}|{int(round(x1))}|{int(round(y2))}|{int(round(x2))}"


def _tlbr_text_to_xyxy(text: str) -> Optional[Tuple[float, float, float, float]]:
    parts = [float(p) for p in str(text).split("|") if p.strip()]
    if len(parts) < 4:
        return None
    top, left, bottom, right = parts[:4]
    return (left, top, right, bottom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marker-run-dir", type=Path, required=True,
                        help="Marker runner output dir: defines folders + image lists to process.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vendor-checkpoint", type=Path, default=PIPE_DIR / "models/vendor_training_no_negative_v2_power/best_model.pt")
    parser.add_argument("--rect-checkpoint", type=Path, default=PIPE_DIR / "models/rect_training_e40_run2/best_model.pt")
    parser.add_argument("--rect-vendor-map", type=Path, default=PIPE_DIR / "maps/vendor_rect_map_bk_only.json")
    parser.add_argument("--su-giu-checkpoint", type=Path,
                        default=REPO_ROOT / "artifacts/20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt")
    parser.add_argument("--reference-manifest", type=Path,
                        default=PIPE_DIR / "references/rect_dataset_no_negative_v2/manifest_rect_echo.csv")
    parser.add_argument("--vendor-sample", type=int, default=80)
    parser.add_argument("--rect-vendor-min-confidence", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rect-red-margin-pct", type=float, default=12.0)
    parser.add_argument("--rect-red-bright-thr", type=float, default=170.0)
    parser.add_argument("--rect-red-max-records", type=int, default=200,
                        help="Uniform sample cap for the red-rect stage (it re-reads every image; "
                             "the official pipeline uses all records — documented simplification).")
    parser.add_argument("--max-folders", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget", type=float, default=0.0)
    args = parser.parse_args()

    out = args.output_dir or (args.marker_run_dir / "official_stages")
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state: Dict[str, Dict[str, object]] = json.loads(state_path.read_text()) if state_path.is_file() else {}

    print("[load] importing official pipeline module + models (cpu)...", flush=True)
    off = _import_official()
    import torch

    from rect_red_pipeline import compute_rect_red_pipeline  # tools/ultrasound on sys.path

    device = torch.device("cpu")
    vendor_ckpt = torch.load(args.vendor_checkpoint, map_location="cpu", weights_only=False)
    vendor_classes = list(vendor_ckpt.get("class_names") or [])
    vendor_image_size = int(vendor_ckpt.get("args", {}).get("image_size", 320))
    vendor_model = off.VendorClassifier(num_classes=len(vendor_classes), pretrained=False).to(device)
    vendor_model.load_state_dict(vendor_ckpt["model_state_dict"])
    vendor_model.eval()

    rect_ckpt = torch.load(args.rect_checkpoint, map_location="cpu", weights_only=False)
    rect_image_size = int(rect_ckpt.get("args", {}).get("image_size", 320))
    rect_model_global = off.RectRegressor(pretrained=False).to(device)
    rect_model_global.load_state_dict(rect_ckpt["model_state_dict"])
    rect_model_global.eval()
    vendor_rect_map: Dict[str, str] = {}
    if args.rect_vendor_map and args.rect_vendor_map.is_file():
        vendor_rect_map = {str(k): str(v) for k, v in json.loads(args.rect_vendor_map.read_text()).items()}
    rect_vendor_cache: Dict[str, Tuple[object, int]] = {}

    su_ckpt = torch.load(args.su_giu_checkpoint, map_location="cpu", weights_only=False)
    su_classes = [str(x).strip().lower() for x in (su_ckpt.get("class_names") or ["su", "giu"])]
    su_image_size = int(su_ckpt.get("args", {}).get("image_size", 256))
    su_model = off.SuGiuRectClassifier().to(device)
    su_model.load_state_dict(su_ckpt["model_state_dict"])
    su_model.eval()

    print("[load] models ready", flush=True)

    resolver_holder: List[object] = []

    def _get_resolver():
        if not resolver_holder:
            t0 = time.time()
            resolver_holder.append(off.RectNameEchoResolver(args.reference_manifest))
            print(f"[load] line13 resolver ready ({time.time() - t0:.1f}s)", flush=True)
        return resolver_holder[0]

    marker_rows = _load_csv(args.marker_run_dir / "per_image_predictions.csv")
    images_by_folder: Dict[str, List[str]] = {}
    for row in marker_rows:
        images_by_folder.setdefault(str(row["folder"]), []).append(str(row["image_id"]))

    done_pairs = {(r["folder"], r["image_id"]) for r in _load_csv(out / "official_per_image.csv")} if args.resume else set()
    completed = {r["folder"] for r in _load_csv(out / "official_folder.csv")} if args.resume else set()

    folders = [f for f in sorted(images_by_folder) if f not in completed]
    if args.max_folders > 0:
        folders = folders[: args.max_folders]
    print(f"[start] {len(completed)} folders done, {len(folders)} to process", flush=True)

    run_started = time.time()
    out_of_time = False
    for index, folder in enumerate(folders, start=1):
        if out_of_time:
            break
        folder_path = args.dataset_root / folder
        image_ids = images_by_folder[folder]
        fstate = state.setdefault(folder, {})

        # 1) vendor (once per folder)
        if "vendor_pred" not in fstate:
            stride = max(1, len(image_ids) // max(1, args.vendor_sample))
            sample_paths = [folder_path / i for i in image_ids[::stride][: args.vendor_sample]]
            probs = off._predict_mean_probs(vendor_model, sample_paths, vendor_image_size, args.batch_size, device)
            if probs.size == 0:
                _append_csv(out / "official_folder.csv",
                            [{"folder": folder, "vendor_pred": "", "line11_method": "error:no_vendor_probs"}], FOLDER_FIELDS)
                completed.add(folder)
                continue
            best = int(probs.argmax())
            fstate["vendor_pred"] = vendor_classes[best]
            fstate["vendor_conf"] = float(probs[best])
            state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False))
        vendor_pred = str(fstate["vendor_pred"])
        vendor_conf = float(fstate["vendor_conf"])

        # 2) rect model routing (official rule)
        rect_model = rect_model_global
        rect_img_size = rect_image_size
        rect_source = "global"
        rect_checkpoint_used = str(args.rect_checkpoint)
        if vendor_pred in vendor_rect_map:
            if vendor_conf >= args.rect_vendor_min_confidence:
                if vendor_pred not in rect_vendor_cache:
                    ckpt_path = Path(vendor_rect_map[vendor_pred])
                    if not ckpt_path.is_absolute():
                        ckpt_path = REPO_ROOT / ckpt_path
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    model = off.RectRegressor(pretrained=False).to(device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()
                    rect_vendor_cache[vendor_pred] = (model, int(ckpt.get("args", {}).get("image_size", rect_image_size)))
                rect_model, rect_img_size = rect_vendor_cache[vendor_pred]
                rect_source = "vendor_specialized"
                rect_checkpoint_used = vendor_rect_map[vendor_pred]
            else:
                rect_source = "global_low_vendor_conf"

        # 3) per-image rect + su/giu, chunked with image-level checkpoint
        pending = [i for i in image_ids if (folder, i) not in done_pairs]
        processed_now = 0
        for start in range(0, len(pending), args.batch_size):
            if args.time_budget > 0 and time.time() - run_started > args.time_budget:
                out_of_time = True
                break
            chunk = pending[start : start + args.batch_size]
            paths = [folder_path / i for i in chunk]
            boxes_abs, _sizes = off._predict_rect_boxes_abs(rect_model, paths, rect_img_size, len(paths), device)
            if len(boxes_abs) != len(paths):
                # rare read failures: fall back to per-image calls to keep alignment
                boxes_abs = []
                for p in paths:
                    single, _s = off._predict_rect_boxes_abs(rect_model, [p], rect_img_size, 1, device)
                    boxes_abs.append(single[0] if single else (0.0, 0.0, 1.0, 1.0))
            su_rows = off._predict_su_giu_on_rect_crops(
                su_model, su_classes, paths, boxes_abs, None, su_image_size, len(paths), device
            )
            su_by_idx = {int(r["image_index"]): r for r in su_rows}
            out_rows = []
            for j, image_id in enumerate(chunk):
                su = su_by_idx.get(j, {})
                crop_txt = ""
                if su:
                    crop_txt = f"{su.get('crop_top', '')}|{su.get('crop_left', '')}|{su.get('crop_bottom', '')}|{su.get('crop_right', '')}"
                out_rows.append(
                    {
                        "folder": folder,
                        "image_id": image_id,
                        "rect_box_abs": _xyxy_to_tlbr_text(boxes_abs[j]),
                        "sugiu_label": str(su.get("pred_label", "")),
                        "sugiu_conf": f"{float(su.get('confidence', 0.0) or 0.0):.4f}",
                        "prob_su": f"{float(su.get('prob_su', 0.0) or 0.0):.4f}",
                        "prob_giu": f"{float(su.get('prob_giu', 0.0) or 0.0):.4f}",
                        "crop_box_abs": crop_txt,
                        "crop_source": str(su.get("crop_source", "")),
                    }
                )
                done_pairs.add((folder, image_id))
                processed_now += 1
            _append_csv(out / "official_per_image.csv", out_rows, IMAGE_FIELDS)

        remaining = len(pending) - processed_now
        if remaining > 0:
            print(f"[{index}/{len(folders)}] {folder} PARTIAL {processed_now} img, {remaining} left", flush=True)
            break

        # 4) folder aggregation: median + red-rect override (official)
        folder_rows = [r for r in _load_csv(out / "official_per_image.csv") if r["folder"] == folder]
        boxes_xyxy = [b for b in (_tlbr_text_to_xyxy(r["rect_box_abs"]) for r in folder_rows) if b]
        median = off._median_box(boxes_xyxy)
        line11_median = _xyxy_to_tlbr_text(median) if median else ""

        dominant = ""
        try:
            marker_state = json.loads((args.marker_run_dir / "state.json").read_text())
            dominant = str(marker_state.get(folder, {}).get("dominant_resolution", ""))
        except Exception:
            pass
        try:
            dom_w, dom_h = (int(v) for v in dominant.split("x"))
        except ValueError:
            dom_w = dom_h = 0

        records = []
        for r in folder_rows:
            box = _tlbr_text_to_xyxy(r["rect_box_abs"])
            if not box or dom_w <= 1 or dom_h <= 1:
                continue
            x1, y1, x2, y2 = box
            records.append(
                {
                    "image_id": Path(r["image_id"]).name,
                    "image_path": (folder_path / r["image_id"]).as_posix(),
                    "image_width": dom_w,
                    "image_height": dom_h,
                    "orientation_label": str(r.get("sugiu_label", "") or "unknown"),
                    "pred_rect_norm": {
                        "x": x1 / dom_w, "y": y1 / dom_h,
                        "w": max(1.0, x2 - x1) / dom_w, "h": max(1.0, y2 - y1) / dom_h,
                    },
                }
            )
        if args.rect_red_max_records > 0 and len(records) > args.rect_red_max_records:
            stride = len(records) / args.rect_red_max_records
            records = [records[int(i * stride)] for i in range(args.rect_red_max_records)]
        line11_final, line11_method = line11_median, "median_rect_fallback"
        if records:
            try:
                payload = compute_rect_red_pipeline(
                    records=records, output_width=dom_w, output_height=dom_h,
                    margin_pct=args.rect_red_margin_pct, bright_thr=args.rect_red_bright_thr,
                    detect_segments=True,
                )
                if bool(payload.get("available", False)):
                    red = payload.get("line11_red", {}) or {}
                    tlbr = red.get("rect_tlbr", {}) or {}
                    if red.get("text") and tlbr:
                        line11_final = f"{int(tlbr['top'])}|{int(tlbr['left'])}|{int(tlbr['bottom'])}|{int(tlbr['right'])}"
                        line11_method = "segment_top_red_rect"
            except Exception as exc:  # noqa: BLE001
                line11_method = f"median_rect_fallback(red_error:{type(exc).__name__})"

        # 5) line #13 via historical resolver (vendor + video size from filenames)
        meta_counter: Counter = Counter()
        for image_id in image_ids[:200]:
            cap_input, cap_x, cap_y = off._parse_capture_metadata_from_filename(folder_path / image_id)
            if cap_x and cap_y:
                meta_counter[(cap_input or "", cap_x, cap_y)] += 1
        video_x = video_y = None
        if meta_counter:
            (_, video_x, video_y), _n = meta_counter.most_common(1)[0]
        elif dom_w and dom_h:
            video_x, video_y = dom_w, dom_h
        try:
            line13_text, line13_source, line13_support = _get_resolver().resolve(
                vendor=vendor_pred, video_x=video_x, video_y=video_y
            )
        except Exception as exc:  # noqa: BLE001
            line13_text, line13_source, line13_support = "", f"resolver_error:{type(exc).__name__}", 0.0

        su_majority = Counter(r["sugiu_label"] for r in folder_rows if r["sugiu_label"]).most_common(1)
        _append_csv(
            out / "official_folder.csv",
            [
                {
                    "folder": folder,
                    "vendor_pred": vendor_pred,
                    "vendor_conf": f"{vendor_conf:.4f}",
                    "rect_source": rect_source,
                    "rect_checkpoint": rect_checkpoint_used,
                    "line11_median": line11_median,
                    "line11_final": line11_final,
                    "line11_method": line11_method,
                    "line13_text": line13_text,
                    "line13_source": line13_source,
                    "line13_support": f"{float(line13_support):.4f}",
                    "video_x": video_x or "",
                    "video_y": video_y or "",
                    "sugiu_majority": su_majority[0][0] if su_majority else "",
                }
            ],
            FOLDER_FIELDS,
        )
        completed.add(folder)
        print(f"[{index}/{len(folders)}] {folder} DONE {len(folder_rows)} img | vendor={vendor_pred}({vendor_conf:.2f}) "
              f"rect={rect_source} line11={line11_method} line13={line13_source}", flush=True)

    print(f"[end] completed: {len(completed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
