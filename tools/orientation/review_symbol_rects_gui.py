#!/usr/bin/env python3
"""Interactive GUI to manually review/fix symbol rectangles.

The tool opens rows from `symbol_detection_manifest.csv`, lets you edit the
small symbol box, mark decision (`keep`, `corrected`, `reject`) and export:

- `<manifest_stem>_manual_review.csv`
- `<manifest_stem>_train_ready.csv`
- session JSON for resume
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageFile, ImageTk

ImageFile.LOAD_TRUNCATED_IMAGES = True

Rect = Tuple[int, int, int, int]  # top, left, bottom, right (inclusive)
ORIENTATION_BY_IDX = {0: "NF", 1: "LR", 2: "UD", 3: "LRUD"}
ORIENTATION_TO_IDX = {v: k for k, v in ORIENTATION_BY_IDX.items()}
RECT_LINE_RE = re.compile(r"^\s*(-?\d+)\|(-?\d+)\|(-?\d+)\|(-?\d+)\|\s*$")


@dataclass(frozen=True)
class SampleRow:
    idx: int
    key: str
    raw: Dict[str, str]
    image_path: Path
    sample_id: str
    split: str
    setup_id: str
    manufacturer: str
    model_name: str
    dataset_folder: str
    source_type: str
    orientation_name: str
    label_kind: str
    auto_accept: str
    match_score: str
    fss_path: Optional[Path]
    coarse_rect: Optional[Rect]
    symbol_rect: Optional[Rect]


def _parse_int(value: str) -> Optional[int]:
    cleaned = (value or "").strip()
    if cleaned == "":
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _rect_from_fields(row: Dict[str, str], prefix: str) -> Optional[Rect]:
    t = _parse_int(row.get(f"{prefix}_top", ""))
    l = _parse_int(row.get(f"{prefix}_left", ""))
    b = _parse_int(row.get(f"{prefix}_bottom", ""))
    r = _parse_int(row.get(f"{prefix}_right", ""))
    if None in (t, l, b, r):
        return None
    return int(t), int(l), int(b), int(r)


def _extract_rect_line_index(lines: Sequence[str]) -> int:
    for candidate in (11, 10):
        if 1 <= candidate <= len(lines):
            if RECT_LINE_RE.match(lines[candidate - 1].strip()):
                return candidate
    for idx in range(1, min(len(lines), 20) + 1):
        if RECT_LINE_RE.match(lines[idx - 1].strip()):
            return idx
    raise ValueError("RECT_ECHO line not found.")


def _parse_line16_rects(raw_line: str) -> Dict[int, Rect]:
    segments = [seg.strip().rstrip(",") for seg in raw_line.split(";") if seg.strip()]
    out: Dict[int, Rect] = {}
    for idx, segment in enumerate(segments):
        if idx >= 4:
            break
        parts = [part.strip() for part in segment.split("|")]
        if len(parts) < 4:
            continue
        t = _parse_int(parts[0])
        l = _parse_int(parts[1])
        b = _parse_int(parts[2])
        r = _parse_int(parts[3])
        if None in (t, l, b, r):
            continue
        out[idx] = (int(t), int(l), int(b), int(r))
    return out


def _clip_rect(rect: Rect, width: int, height: int, min_size: int = 3) -> Rect:
    t, l, b, r = rect
    t = max(0, min(t, height - 1))
    b = max(0, min(b, height - 1))
    l = max(0, min(l, width - 1))
    r = max(0, min(r, width - 1))
    if b < t:
        t, b = b, t
    if r < l:
        l, r = r, l
    if b - t + 1 < min_size:
        b = min(height - 1, t + min_size - 1)
        t = max(0, b - min_size + 1)
    if r - l + 1 < min_size:
        r = min(width - 1, l + min_size - 1)
        l = max(0, r - min_size + 1)
    return t, l, b, r


def _point_in_rect(x: int, y: int, rect: Rect) -> bool:
    t, l, b, r = rect
    return l <= x <= r and t <= y <= b


def _rect_translate(rect: Rect, dy: int, dx: int, width: int, height: int) -> Rect:
    t, l, b, r = rect
    h = b - t
    w = r - l
    new_t = t + dy
    new_l = l + dx
    new_b = b + dy
    new_r = r + dx
    if new_t < 0:
        shift = -new_t
        new_t += shift
        new_b += shift
    if new_b >= height:
        shift = new_b - (height - 1)
        new_t -= shift
        new_b -= shift
    if new_l < 0:
        shift = -new_l
        new_l += shift
        new_r += shift
    if new_r >= width:
        shift = new_r - (width - 1)
        new_l -= shift
        new_r -= shift
    new_t = max(0, min(new_t, height - 1))
    new_b = max(0, min(new_b, height - 1))
    new_l = max(0, min(new_l, width - 1))
    new_r = max(0, min(new_r, width - 1))
    if new_b - new_t != h:
        new_b = min(height - 1, new_t + h)
    if new_r - new_l != w:
        new_r = min(width - 1, new_l + w)
    return _clip_rect((new_t, new_l, new_b, new_r), width=width, height=height)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _rect_to_normalized(rect: Rect, width: int, height: int) -> Tuple[float, float, float, float]:
    t, l, b, r = rect
    sx = max(1, width - 1)
    sy = max(1, height - 1)
    return t / sy, l / sx, b / sy, r / sx


def _rect_from_normalized(norm_rect: Tuple[float, float, float, float], width: int, height: int) -> Rect:
    nt, nl, nb, nr = norm_rect
    sx = max(1, width - 1)
    sy = max(1, height - 1)
    rect = (
        int(round(nt * sy)),
        int(round(nl * sx)),
        int(round(nb * sy)),
        int(round(nr * sx)),
    )
    return _clip_rect(rect, width=width, height=height)


def _read_manifest(path: Path, only_review_needed: bool, max_rows: int) -> Tuple[List[SampleRow], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows: List[SampleRow] = []
        for idx, raw in enumerate(reader, start=1):
            if only_review_needed and (raw.get("auto_accept", "") or "").strip() == "1":
                continue
            image_path = Path(raw.get("image_path", "")).expanduser().resolve()
            if not image_path.exists():
                continue
            fss_path_value = (raw.get("fss_path", "") or "").strip()
            fss_path: Optional[Path] = None
            if fss_path_value:
                candidate = Path(fss_path_value).expanduser().resolve()
                if candidate.exists():
                    fss_path = candidate
            sample_id = (raw.get("sample_id", "") or f"sample_{idx:08d}").strip()
            key = sample_id if sample_id else f"{image_path.as_posix()}::{idx}"
            rows.append(
                SampleRow(
                    idx=idx,
                    key=key,
                    raw=raw,
                    image_path=image_path,
                    sample_id=sample_id,
                    split=(raw.get("split", "") or ""),
                    setup_id=(raw.get("setup_id", "") or ""),
                    manufacturer=(raw.get("manufacturer", "") or ""),
                    model_name=(raw.get("model_name", "") or ""),
                    dataset_folder=(raw.get("dataset_folder", "") or ""),
                    source_type=(raw.get("source_type", "") or ""),
                    orientation_name=(raw.get("orientation_name", "") or ""),
                    label_kind=(raw.get("label_kind", "") or ""),
                    auto_accept=(raw.get("auto_accept", "") or ""),
                    match_score=(raw.get("match_score", "") or ""),
                    fss_path=fss_path,
                    coarse_rect=_rect_from_fields(raw, "coarse_rect"),
                    symbol_rect=_rect_from_fields(raw, "symbol_rect"),
                )
            )
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows, fieldnames


class RectReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        samples: Sequence[SampleRow],
        fieldnames: Sequence[str],
        manifest_path: Path,
        output_dir: Path,
        session_path: Path,
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.fieldnames = list(fieldnames)
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.session_path = session_path

        self.index = 0
        self.image_cache: Dict[str, Image.Image] = {}
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.current_scale = 1.0
        self.current_offx = 0
        self.current_offy = 0
        self.current_disp_w = 1
        self.current_disp_h = 1
        self.drag_mode: Optional[str] = None
        self.drag_start_img: Optional[Tuple[int, int]] = None
        self.drag_start_rect: Optional[Rect] = None
        self.drag_corner: Optional[str] = None
        self.temp_draw_rect: Optional[Rect] = None
        self.jump_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Ready.")
        self.batch_folder_var = tk.StringVar(value="")
        self.batch_same_orientation_var = tk.BooleanVar(value=True)
        self.batch_pending_only_var = tk.BooleanVar(value=False)
        self.batch_overwrite_rect_var = tk.BooleanVar(value=False)
        self.batch_include_subfolders_var = tk.BooleanVar(value=False)
        self.edit_mode_var = tk.StringVar(value="symbol")
        self.fss_rect_cache: Dict[str, Optional[Tuple[Dict[int, Rect], int, int]]] = {}

        self.state: Dict[str, Dict[str, object]] = self._load_state()

        self._build_ui()
        self._bind_keys()
        self._refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._autosave_tick()

    def _build_ui(self) -> None:
        self.root.title("Symbol Rect Review")
        self.root.geometry("1560x980")

        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        self.info_var = tk.StringVar()
        self.info_lbl = ttk.Label(top, textvariable=self.info_var, font=("TkDefaultFont", 11, "bold"))
        self.info_lbl.pack(side=tk.LEFT, padx=(0, 16), fill=tk.X, expand=True)

        self.progress_var = tk.StringVar()
        self.progress_lbl = ttk.Label(top, textvariable=self.progress_var)
        self.progress_lbl.pack(side=tk.LEFT)
        self.progress_ratio_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            top,
            mode="determinate",
            variable=self.progress_ratio_var,
            maximum=100.0,
            length=180,
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(8, 0))

        jump_fr = ttk.Frame(top)
        jump_fr.pack(side=tk.RIGHT)
        ttk.Label(jump_fr, text="Vai #").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(jump_fr, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=(4, 4))
        self.jump_entry.bind("<Return>", lambda _e: self.jump_to_index())
        ttk.Button(jump_fr, text="Go [G]", command=self.jump_to_index).pack(side=tk.LEFT)

        nav_controls = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        nav_controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(nav_controls, text="Prev [P]", command=self.prev_row).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav_controls, text="Next [N]", command=self.next_row).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav_controls, text="Next Pending [Space]", command=self.next_pending).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav_controls, text="Keep [K]", command=lambda: self.set_decision("keep", advance=True)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            nav_controls,
            text="Corrected [C]",
            command=lambda: self.set_decision("corrected", advance=True),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(nav_controls, text="Reject [X]", command=lambda: self.set_decision("reject", advance=True)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(nav_controls, text="Unset [U]", command=self.unset_decision).pack(side=tk.LEFT, padx=3)

        edit_controls = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        edit_controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(edit_controls, text="Clear Rect [Del]", command=self.clear_corrected_rect).pack(side=tk.LEFT, padx=3)
        ttk.Button(edit_controls, text="Copy Original Rect [O]", command=self.copy_original_rect).pack(side=tk.LEFT, padx=3)
        ttk.Button(edit_controls, text="Rect da FSS [F]", command=self.apply_fss_rect_current_orientation).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(edit_controls, text="ORI NF [1]", command=lambda: self.set_orientation_override(0)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(edit_controls, text="ORI LR [2]", command=lambda: self.set_orientation_override(1)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(edit_controls, text="ORI UD [3]", command=lambda: self.set_orientation_override(2)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(edit_controls, text="ORI LRUD [4]", command=lambda: self.set_orientation_override(3)).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(edit_controls, text="ORI Clear [0]", command=self.clear_orientation_override).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Label(edit_controls, text="Annota:").pack(side=tk.LEFT, padx=(14, 4))
        ttk.Radiobutton(
            edit_controls,
            text="Simbolo (rosso)",
            variable=self.edit_mode_var,
            value="symbol",
            command=self._refresh,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(
            edit_controls,
            text="Area ricerca orientamento",
            variable=self.edit_mode_var,
            value="envelope",
            command=self._refresh,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(edit_controls, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=3)
        ttk.Button(edit_controls, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=3)

        note_fr = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        note_fr.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(note_fr, text="Note:").pack(side=tk.LEFT)
        self.note_var = tk.StringVar()
        self.note_entry = ttk.Entry(note_fr, textvariable=self.note_var, width=160)
        self.note_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.note_entry.bind("<FocusOut>", lambda _e: self._store_current_note())

        batch_fr = ttk.LabelFrame(self.root, text="Batch su sottocartella", padding=(8, 6, 8, 8))
        batch_fr.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        folder_line = ttk.Frame(batch_fr)
        folder_line.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(folder_line, text="Cartella target:").pack(side=tk.LEFT)
        self.batch_folder_entry = ttk.Entry(folder_line, textvariable=self.batch_folder_var)
        self.batch_folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(folder_line, text="Usa cartella corrente", command=self.use_current_folder_as_scope).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(folder_line, text="Applica rect [A]", command=self.apply_rect_to_scope).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            folder_line,
            text="Applica orientamento [B]",
            command=self.apply_orientation_to_scope,
        ).pack(side=tk.LEFT, padx=3)

        opt_line = ttk.Frame(batch_fr)
        opt_line.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(
            opt_line,
            text="Solo stesso orientamento",
            variable=self.batch_same_orientation_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            opt_line,
            text="Solo pending",
            variable=self.batch_pending_only_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            opt_line,
            text="Sovrascrivi rect gia corretti",
            variable=self.batch_overwrite_rect_var,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(
            opt_line,
            text="Includi sottocartelle",
            variable=self.batch_include_subfolders_var,
        ).pack(side=tk.LEFT, padx=(0, 8))

        legend = ttk.Label(
            self.root,
            text=(
                "Mouse: drag outside box = draw new rect | drag inside box = move | drag near corner = resize | "
                "Middle click = clear rect. Red tratteggiato=simbolo originale, rosso pieno=simbolo annotato, "
                "azzurro=area ricerca orientamento. Keys: 1/2/3/4=ORI, 0=clear ORI, F=rect da FSS, G=go index, A=batch rect, B=batch orient."
            ),
            padding=(8, 0, 8, 6),
        )
        legend.pack(side=tk.TOP, fill=tk.X)

        canvas_fr = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        canvas_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_fr, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-2>", lambda _e: self.clear_corrected_rect())
        self.canvas.bind("<Button-3>", lambda _e: self.clear_corrected_rect())

        status_fr = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        status_fr.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_lbl = ttk.Label(status_fr, textvariable=self.status_var, anchor=tk.W)
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.use_current_folder_as_scope()

    def _focus_is_text_input(self) -> bool:
        focused = self.root.focus_get()
        if focused is None:
            return False
        cls = focused.winfo_class().lower()
        return cls in {"entry", "tentry", "text", "spinbox"}

    def _bind_shortcut(self, sequence: str, handler, allow_in_text: bool = False) -> None:
        def _wrapped(event: tk.Event):
            if not allow_in_text and self._focus_is_text_input():
                return None
            handler(event)
            return "break"

        self.root.bind(sequence, _wrapped)
        if len(sequence) == 1 and sequence.isalpha():
            self.root.bind(sequence.upper(), _wrapped)

    def _bind_keys(self) -> None:
        self._bind_shortcut("n", lambda _e: self.next_row())
        self._bind_shortcut("p", lambda _e: self.prev_row())
        self._bind_shortcut("k", lambda _e: self.set_decision("keep", advance=True))
        self._bind_shortcut("c", lambda _e: self.set_decision("corrected", advance=True))
        self._bind_shortcut("x", lambda _e: self.set_decision("reject", advance=True))
        self._bind_shortcut("u", lambda _e: self.unset_decision())
        self._bind_shortcut("o", lambda _e: self.copy_original_rect())
        self._bind_shortcut("f", lambda _e: self.apply_fss_rect_current_orientation())
        self._bind_shortcut("1", lambda _e: self.set_orientation_override(0))
        self._bind_shortcut("2", lambda _e: self.set_orientation_override(1))
        self._bind_shortcut("3", lambda _e: self.set_orientation_override(2))
        self._bind_shortcut("4", lambda _e: self.set_orientation_override(3))
        self._bind_shortcut("0", lambda _e: self.clear_orientation_override())
        self._bind_shortcut("g", lambda _e: self.jump_to_index())
        self._bind_shortcut("a", lambda _e: self.apply_rect_to_scope())
        self._bind_shortcut("b", lambda _e: self.apply_orientation_to_scope())
        self._bind_shortcut("<space>", lambda _e: self.next_pending())
        self._bind_shortcut("<Delete>", lambda _e: self.clear_corrected_rect())
        self._bind_shortcut("<Left>", self.on_arrow_key)
        self._bind_shortcut("<Right>", self.on_arrow_key)
        self._bind_shortcut("<Up>", self.on_arrow_key)
        self._bind_shortcut("<Down>", self.on_arrow_key)

    def _load_state(self) -> Dict[str, Dict[str, object]]:
        if not self.session_path.exists():
            return {}
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def save_session(self, autosave: bool = False) -> None:
        self._store_current_note()
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        if autosave:
            self._set_status("Autosave completato.")
            return
        self.root.title(f"Symbol Rect Review - saved {self.session_path.name}")
        self._set_status(f"Sessione salvata: {self.session_path.name}")

    def _autosave_tick(self) -> None:
        try:
            self.save_session(autosave=True)
        except Exception:
            pass
        self.root.after(25000, self._autosave_tick)

    def _on_close(self) -> None:
        self.save_session()
        self.root.destroy()

    def _current_sample(self) -> SampleRow:
        return self.samples[self.index]

    def _get_state(self, sample: SampleRow) -> Dict[str, object]:
        st = self.state.setdefault(sample.key, {})
        return st

    def _get_decision(self, sample: SampleRow) -> str:
        st = self._get_state(sample)
        return str(st.get("manual_decision", "") or "")

    def _get_note(self, sample: SampleRow) -> str:
        st = self._get_state(sample)
        return str(st.get("manual_note", "") or "")

    def _get_manual_orientation_idx(self, sample: SampleRow) -> Optional[int]:
        st = self._get_state(sample)
        value = st.get("manual_orientation_idx")
        try:
            if value is None or str(value) == "":
                return None
            idx = int(value)
        except Exception:
            return None
        if idx not in ORIENTATION_BY_IDX:
            return None
        return idx

    def _set_manual_orientation_idx(self, sample: SampleRow, idx: Optional[int]) -> None:
        st = self._get_state(sample)
        if idx is None:
            st.pop("manual_orientation_idx", None)
            st.pop("manual_orientation_name", None)
            return
        st["manual_orientation_idx"] = int(idx)
        st["manual_orientation_name"] = ORIENTATION_BY_IDX.get(int(idx), "")

    def _current_final_orientation(self, sample: SampleRow) -> Tuple[Optional[int], str]:
        manual_idx = self._get_manual_orientation_idx(sample)
        if manual_idx is not None:
            return manual_idx, ORIENTATION_BY_IDX.get(manual_idx, "")
        base = sample.orientation_name.strip().upper()
        idx = ORIENTATION_TO_IDX.get(base)
        return idx, base

    def _get_symbol_rect(self, sample: SampleRow) -> Optional[Rect]:
        st = self._get_state(sample)
        val = st.get("manual_symbol_rect")
        if not isinstance(val, list) or len(val) != 4:
            # Legacy key for backward compatibility with older session files.
            val = st.get("corrected_symbol_rect")
        if not isinstance(val, list) or len(val) != 4:
            return None
        try:
            return int(val[0]), int(val[1]), int(val[2]), int(val[3])
        except Exception:
            return None

    def _set_symbol_rect(self, sample: SampleRow, rect: Optional[Rect]) -> None:
        st = self._get_state(sample)
        if rect is None:
            st.pop("manual_symbol_rect", None)
            st.pop("corrected_symbol_rect", None)
        else:
            st["manual_symbol_rect"] = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]
            # Keep legacy field for compatibility with previous exports/tools.
            st["corrected_symbol_rect"] = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]
            if st.get("manual_decision", "") == "":
                st["manual_decision"] = "corrected"

    def _get_corrected_rect(self, sample: SampleRow) -> Optional[Rect]:
        return self._get_symbol_rect(sample)

    def _set_corrected_rect(self, sample: SampleRow, rect: Optional[Rect]) -> None:
        self._set_symbol_rect(sample, rect)

    def _set_decision(self, sample: SampleRow, decision: str) -> None:
        st = self._get_state(sample)
        if decision:
            st["manual_decision"] = decision
        else:
            st.pop("manual_decision", None)

    def _store_current_note(self) -> None:
        if not self.samples:
            return
        sample = self._current_sample()
        st = self._get_state(sample)
        note = self.note_var.get().strip()
        if note:
            st["manual_note"] = note
        else:
            st.pop("manual_note", None)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _display_path(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(Path.cwd())
            return rel.as_posix()
        except Exception:
            return path.resolve().as_posix()

    def _group_key(self, sample: SampleRow) -> str:
        if sample.setup_id.strip():
            return sample.setup_id.strip()
        if sample.fss_path is not None:
            return sample.fss_path.as_posix()
        return sample.dataset_folder or sample.image_path.parent.as_posix()

    def _group_annotations(self) -> Dict[str, Dict[str, object]]:
        group_root = self.state.setdefault("__group_annotations__", {})
        if not isinstance(group_root, dict):
            self.state["__group_annotations__"] = {}
            group_root = self.state["__group_annotations__"]
        return group_root  # type: ignore[return-value]

    def _get_group_orientation_rect(self, sample: SampleRow, orientation_idx: int) -> Optional[Rect]:
        if orientation_idx not in ORIENTATION_BY_IDX:
            return None
        group_state = self._group_annotations().get(self._group_key(sample), {})
        if not isinstance(group_state, dict):
            return None
        by_orientation = group_state.get("orientation_rects", {})
        if not isinstance(by_orientation, dict):
            return None
        raw = by_orientation.get(str(int(orientation_idx)))
        if not isinstance(raw, list) or len(raw) != 4:
            return None
        try:
            return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
        except Exception:
            return None

    def _set_group_orientation_rect(self, sample: SampleRow, orientation_idx: int, rect: Optional[Rect]) -> None:
        if orientation_idx not in ORIENTATION_BY_IDX:
            return
        group_root = self._group_annotations()
        group_key = self._group_key(sample)
        group_state = group_root.setdefault(group_key, {})
        if not isinstance(group_state, dict):
            group_state = {}
            group_root[group_key] = group_state
        by_orientation = group_state.setdefault("orientation_rects", {})
        if not isinstance(by_orientation, dict):
            by_orientation = {}
            group_state["orientation_rects"] = by_orientation
        if rect is None:
            by_orientation.pop(str(int(orientation_idx)), None)
            return
        by_orientation[str(int(orientation_idx))] = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]

    def _current_orientation_idx(self, sample: SampleRow) -> Optional[int]:
        idx, _ = self._current_final_orientation(sample)
        return idx

    def _current_parent_folder(self) -> Path:
        return self._current_sample().image_path.parent.resolve()

    def _resolve_scope_folder(self) -> Path:
        raw = (self.batch_folder_var.get() or "").strip()
        if raw == "":
            return self._current_parent_folder()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        return path

    def _sample_in_scope(self, sample: SampleRow, scope_folder: Path, include_subfolders: bool) -> bool:
        folder = sample.image_path.parent.resolve()
        if include_subfolders:
            return _path_is_within(folder, scope_folder)
        return folder == scope_folder

    def use_current_folder_as_scope(self) -> None:
        folder = self._current_parent_folder()
        self.batch_folder_var.set(self._display_path(folder))
        self._set_status(f"Scope cartella impostato: {self._display_path(folder)}")

    def jump_to_index(self) -> None:
        raw = (self.jump_var.get() or "").strip()
        if raw == "":
            self._set_status("Inserisci un indice valido.")
            return
        try:
            wanted = int(raw)
        except ValueError:
            self._set_status(f"Indice non valido: {raw}")
            return
        if wanted < 1 or wanted > len(self.samples):
            self._set_status(f"Indice fuori range: {wanted} (1..{len(self.samples)})")
            return
        self._store_current_note()
        self.index = wanted - 1
        self.temp_draw_rect = None
        self._refresh()
        self._set_status(f"Vai a sample #{wanted}.")

    def _load_image(self, sample: SampleRow) -> Image.Image:
        key = sample.image_path.as_posix()
        img = self.image_cache.get(key)
        if img is None:
            with Image.open(sample.image_path) as src:
                img = src.convert("RGB")
            self.image_cache[key] = img
        return img

    def _load_fss_rects(self, fss_path: Path) -> Optional[Tuple[Dict[int, Rect], int, int]]:
        cache_key = fss_path.as_posix()
        cached = self.fss_rect_cache.get(cache_key, None)
        if cache_key in self.fss_rect_cache:
            return cached

        try:
            lines = fss_path.read_text(encoding="utf-8", errors="replace").splitlines()
            rect_line = _extract_rect_line_index(lines)
            offset = rect_line - 11
            x_line = 9 + offset
            y_line = 10 + offset
            video_x = _parse_int(lines[x_line - 1] if 1 <= x_line <= len(lines) else "") or 0
            video_y = _parse_int(lines[y_line - 1] if 1 <= y_line <= len(lines) else "") or 0

            line16_idx = rect_line + 5
            if line16_idx > len(lines):
                self.fss_rect_cache[cache_key] = None
                return None
            line16_raw = lines[line16_idx - 1].strip()
            rects = _parse_line16_rects(line16_raw)
            if not rects:
                self.fss_rect_cache[cache_key] = None
                return None
            self.fss_rect_cache[cache_key] = (rects, int(video_x), int(video_y))
            return self.fss_rect_cache[cache_key]
        except Exception:
            self.fss_rect_cache[cache_key] = None
            return None

    def _rect_from_fss_for_orientation(self, sample: SampleRow, orientation_idx: int) -> Optional[Rect]:
        if orientation_idx not in ORIENTATION_BY_IDX:
            return None
        if sample.fss_path is None:
            return None
        payload = self._load_fss_rects(sample.fss_path)
        if payload is None:
            return None
        rect_map, video_x, video_y = payload
        rect = rect_map.get(orientation_idx)
        if rect is None:
            return None
        img = self._load_image(sample)
        src_w = max(1, int(video_x)) if video_x > 0 else img.width
        src_h = max(1, int(video_y)) if video_y > 0 else img.height
        src_rect = _clip_rect(rect, width=src_w, height=src_h)
        if src_w == img.width and src_h == img.height:
            return src_rect
        normalized = _rect_to_normalized(src_rect, width=src_w, height=src_h)
        return _rect_from_normalized(normalized, width=img.width, height=img.height)

    def _apply_fss_rect_for_orientation(
        self,
        sample: SampleRow,
        orientation_idx: int,
        overwrite_existing: bool = True,
    ) -> bool:
        if not overwrite_existing and self._get_group_orientation_rect(sample, orientation_idx) is not None:
            return False
        rect = self._rect_from_fss_for_orientation(sample, orientation_idx)
        if rect is None:
            return False
        self._set_group_orientation_rect(sample, orientation_idx, rect)
        self._set_decision(sample, "corrected")
        st = self._get_state(sample)
        st["orientation_search_rect_source"] = "fss_line16"
        st["orientation_search_rect_source_orientation_idx"] = int(orientation_idx)
        st["orientation_search_rect_source_orientation_name"] = ORIENTATION_BY_IDX.get(int(orientation_idx), "")
        return True

    def _active_rect(self, sample: SampleRow) -> Optional[Rect]:
        mode = (self.edit_mode_var.get() or "symbol").strip().lower()
        if mode == "envelope":
            orientation_idx = self._current_orientation_idx(sample)
            if orientation_idx is None:
                return None
            manual_envelope = self._get_group_orientation_rect(sample, orientation_idx)
            if manual_envelope is not None:
                return manual_envelope
            return self._rect_from_fss_for_orientation(sample, orientation_idx)

        corr = self._get_symbol_rect(sample)
        if corr is not None:
            return corr
        return sample.symbol_rect

    def _set_active_rect(self, sample: SampleRow, rect: Optional[Rect]) -> None:
        mode = (self.edit_mode_var.get() or "symbol").strip().lower()
        if mode == "envelope":
            orientation_idx = self._current_orientation_idx(sample)
            if orientation_idx is None:
                return
            self._set_group_orientation_rect(sample, orientation_idx, rect)
            if rect is not None:
                self._set_decision(sample, "corrected")
            return
        self._set_symbol_rect(sample, rect)

    def _refresh(self) -> None:
        sample = self._current_sample()
        self.note_var.set(self._get_note(sample))
        self.jump_var.set(str(self.index + 1))
        if (self.batch_folder_var.get() or "").strip() == "":
            self.batch_folder_var.set(self._display_path(self._current_parent_folder()))
        self._redraw()
        self._refresh_header()

    def _refresh_header(self) -> None:
        sample = self._current_sample()
        decision = self._get_decision(sample) or "-"
        score = sample.match_score or "-"
        _, final_name = self._current_final_orientation(sample)
        ori_txt = sample.orientation_name
        if final_name and final_name != sample.orientation_name:
            ori_txt = f"{sample.orientation_name}->{final_name}"
        mode_txt = "simbolo" if (self.edit_mode_var.get() or "symbol") == "symbol" else "area_ricerca"
        folder_txt = self._display_path(sample.image_path.parent)
        self.info_var.set(
            f"{self.index + 1}/{len(self.samples)} | {sample.sample_id} | {sample.manufacturer} | "
            f"ori={ori_txt} | mode={mode_txt} | score={score} | decision={decision} | folder={folder_txt}"
        )
        counts = {"keep": 0, "corrected": 0, "reject": 0}
        for s in self.samples:
            d = self._get_decision(s)
            if d in counts:
                counts[d] += 1
        pending = len(self.samples) - sum(counts.values())
        done = len(self.samples) - pending
        ratio = 100.0 * float(done) / float(max(1, len(self.samples)))
        self.progress_ratio_var.set(ratio)
        self.progress_var.set(
            f"keep={counts['keep']} corrected={counts['corrected']} reject={counts['reject']} pending={pending} ({ratio:.1f}%)"
        )

    def _image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return self.current_offx + x * self.current_scale, self.current_offy + y * self.current_scale

    def _canvas_to_image(self, cx: float, cy: float) -> Tuple[int, int]:
        sample = self._current_sample()
        img = self._load_image(sample)
        x = int(round((cx - self.current_offx) / max(1e-8, self.current_scale)))
        y = int(round((cy - self.current_offy) / max(1e-8, self.current_scale)))
        x = max(0, min(img.width - 1, x))
        y = max(0, min(img.height - 1, y))
        return x, y

    def _draw_rect(self, rect: Rect, color: str, width: int, dash: Optional[Tuple[int, int]] = None) -> None:
        t, l, b, r = rect
        x0, y0 = self._image_to_canvas(l, t)
        x1, y1 = self._image_to_canvas(r, b)
        kwargs = {"outline": color, "width": width}
        if dash is not None:
            kwargs["dash"] = dash
        self.canvas.create_rectangle(x0, y0, x1, y1, **kwargs)

    def _redraw(self) -> None:
        if not self.samples:
            return
        sample = self._current_sample()
        img = self._load_image(sample)
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())
        self.current_scale = min(cw / img.width, ch / img.height)
        self.current_disp_w = max(1, int(round(img.width * self.current_scale)))
        self.current_disp_h = max(1, int(round(img.height * self.current_scale)))
        self.current_offx = int((cw - self.current_disp_w) // 2)
        self.current_offy = int((ch - self.current_disp_h) // 2)

        resized = img.resize((self.current_disp_w, self.current_disp_h), resample=Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(self.current_offx, self.current_offy, anchor=tk.NW, image=self.tk_image)

        if sample.coarse_rect is not None:
            self._draw_rect(sample.coarse_rect, color="#ffd24a", width=2)
        if sample.symbol_rect is not None:
            self._draw_rect(sample.symbol_rect, color="#ff4d4d", width=2, dash=(6, 4))

        corr = self._get_symbol_rect(sample)
        if corr is not None:
            self._draw_rect(corr, color="#ff0000", width=3)
        envelope_idx = self._current_orientation_idx(sample)
        if envelope_idx is not None:
            envelope = self._get_group_orientation_rect(sample, envelope_idx)
            if envelope is not None:
                self._draw_rect(envelope, color="#33d1ff", width=2)
            else:
                envelope_fss = self._rect_from_fss_for_orientation(sample, envelope_idx)
                if envelope_fss is not None:
                    self._draw_rect(envelope_fss, color="#33d1ff", width=2, dash=(4, 4))
        if self.temp_draw_rect is not None:
            self._draw_rect(self.temp_draw_rect, color="#66c2ff", width=2)

    def prev_row(self) -> None:
        self._store_current_note()
        if self.index > 0:
            self.index -= 1
            self.temp_draw_rect = None
            self._refresh()

    def next_row(self) -> None:
        self._store_current_note()
        if self.index + 1 < len(self.samples):
            self.index += 1
            self.temp_draw_rect = None
            self._refresh()

    def next_pending(self) -> None:
        self._store_current_note()
        start = self.index + 1
        for i in range(start, len(self.samples)):
            if self._get_decision(self.samples[i]) == "":
                self.index = i
                self._refresh()
                return
        for i in range(0, start):
            if self._get_decision(self.samples[i]) == "":
                self.index = i
                self._refresh()
                return
        messagebox.showinfo("Review", "Nessun sample pending.")

    def set_decision(self, decision: str, advance: bool = False) -> None:
        sample = self._current_sample()
        self._set_decision(sample, decision)
        self._store_current_note()
        self._refresh_header()
        if advance:
            self.next_row()

    def unset_decision(self) -> None:
        sample = self._current_sample()
        self._set_decision(sample, "")
        self._refresh_header()

    def clear_corrected_rect(self) -> None:
        sample = self._current_sample()
        self._set_active_rect(sample, None)
        self.temp_draw_rect = None
        self._redraw()
        self._refresh_header()

    def copy_original_rect(self) -> None:
        sample = self._current_sample()
        if sample.symbol_rect is None:
            return
        img = self._load_image(sample)
        rect = _clip_rect(sample.symbol_rect, width=img.width, height=img.height)
        self._set_corrected_rect(sample, rect)
        self._set_decision(sample, "corrected")
        self._redraw()
        self._refresh_header()

    def apply_fss_rect_current_orientation(self) -> None:
        sample = self._current_sample()
        orientation_idx, orientation_name = self._current_final_orientation(sample)
        if orientation_idx is None:
            messagebox.showwarning("Area da FSS", "Orientamento corrente non definito.")
            self._set_status("Area da FSS non applicata: orientamento non definito.")
            return
        applied = self._apply_fss_rect_for_orientation(sample, orientation_idx, overwrite_existing=True)
        if not applied:
            messagebox.showwarning(
                "Area da FSS",
                "Area di ricerca da .fss non disponibile per questo sample/orientamento.",
            )
            self._set_status("Area da FSS non disponibile per il sample corrente.")
            return
        self._redraw()
        self._refresh_header()
        self._set_status(f"Area ricerca da FSS applicata per orientamento {orientation_name}.")

    def set_orientation_override(self, orientation_idx: int) -> None:
        sample = self._current_sample()
        if orientation_idx not in ORIENTATION_BY_IDX:
            return
        self._set_manual_orientation_idx(sample, orientation_idx)
        applied = self._apply_fss_rect_for_orientation(sample, orientation_idx, overwrite_existing=True)
        if applied:
            self._set_status(f"Orientamento {ORIENTATION_BY_IDX[orientation_idx]}: proposta area ricerca da FSS.")
        else:
            self._set_status(
                f"Orientamento {ORIENTATION_BY_IDX[orientation_idx]} impostato (area ricerca FSS non trovata)."
            )
        self._redraw()
        self._refresh_header()

    def clear_orientation_override(self) -> None:
        sample = self._current_sample()
        self._set_manual_orientation_idx(sample, None)
        self._refresh_header()

    def _collect_scope_samples(self, scope_folder: Path, include_subfolders: bool) -> List[SampleRow]:
        selected: List[SampleRow] = []
        for sample in self.samples:
            if self._sample_in_scope(sample, scope_folder=scope_folder, include_subfolders=include_subfolders):
                selected.append(sample)
        return selected

    def apply_rect_to_scope(self) -> None:
        source = self._current_sample()
        source_rect = self._get_symbol_rect(source)
        if source_rect is None:
            source_rect = source.symbol_rect
        if source_rect is None:
            messagebox.showwarning("Batch rect", "Il sample corrente non ha rect disponibile.")
            self._set_status("Batch rect non eseguito: rect sorgente assente.")
            return

        source_img = self._load_image(source)
        source_norm_rect = _rect_to_normalized(source_rect, width=source_img.width, height=source_img.height)
        scope_folder = self._resolve_scope_folder()
        include_subfolders = bool(self.batch_include_subfolders_var.get())
        only_pending = bool(self.batch_pending_only_var.get())
        same_orientation = bool(self.batch_same_orientation_var.get())
        overwrite_rect = bool(self.batch_overwrite_rect_var.get())

        source_ori_idx, source_ori_name = self._current_final_orientation(source)
        if same_orientation and source_ori_idx is None:
            messagebox.showwarning(
                "Batch rect",
                "Orientamento corrente non definito: imposta 1/2/3/4 oppure disattiva 'Solo stesso orientamento'.",
            )
            self._set_status("Batch rect non eseguito: orientamento sorgente non definito.")
            return

        scoped_samples = self._collect_scope_samples(scope_folder=scope_folder, include_subfolders=include_subfolders)
        target_samples: List[SampleRow] = []
        for sample in scoped_samples:
            if only_pending and self._get_decision(sample) != "":
                continue
            if same_orientation:
                target_ori_idx, _ = self._current_final_orientation(sample)
                if target_ori_idx != source_ori_idx:
                    continue
            if not overwrite_rect and self._get_corrected_rect(sample) is not None:
                continue
            target_samples.append(sample)

        if not target_samples:
            messagebox.showinfo("Batch rect", "Nessun sample compatibile con i filtri correnti.")
            self._set_status("Batch rect: 0 sample aggiornati.")
            return

        scope_txt = self._display_path(scope_folder)
        msg = (
            f"Applico il rect corrente a {len(target_samples)} sample?\n\n"
            f"Scope: {scope_txt}\n"
            f"Include sottocartelle: {'si' if include_subfolders else 'no'}\n"
            f"Solo pending: {'si' if only_pending else 'no'}\n"
            f"Solo stesso orientamento: {'si' if same_orientation else 'no'}"
        )
        if same_orientation and source_ori_name:
            msg += f" ({source_ori_name})"
        msg += f"\nSovrascrivi rect esistenti: {'si' if overwrite_rect else 'no'}"
        if not messagebox.askyesno("Conferma batch rect", msg):
            self._set_status("Batch rect annullato.")
            return

        changed = 0
        for sample in target_samples:
            target_img = self._load_image(sample)
            target_rect = _rect_from_normalized(source_norm_rect, width=target_img.width, height=target_img.height)
            self._set_corrected_rect(sample, target_rect)
            self._set_decision(sample, "corrected")
            changed += 1

        self._refresh()
        self._set_status(f"Batch rect completato: {changed} sample aggiornati.")

    def apply_orientation_to_scope(self) -> None:
        source = self._current_sample()
        source_ori_idx, source_ori_name = self._current_final_orientation(source)
        if source_ori_idx is None:
            messagebox.showwarning("Batch orientamento", "Orientamento corrente non definito.")
            self._set_status("Batch orientamento non eseguito: orientamento sorgente assente.")
            return

        scope_folder = self._resolve_scope_folder()
        include_subfolders = bool(self.batch_include_subfolders_var.get())
        only_pending = bool(self.batch_pending_only_var.get())
        scoped_samples = self._collect_scope_samples(scope_folder=scope_folder, include_subfolders=include_subfolders)

        target_samples: List[SampleRow] = []
        for sample in scoped_samples:
            if only_pending and self._get_decision(sample) != "":
                continue
            target_samples.append(sample)

        if not target_samples:
            messagebox.showinfo("Batch orientamento", "Nessun sample compatibile con i filtri correnti.")
            self._set_status("Batch orientamento: 0 sample aggiornati.")
            return

        msg = (
            f"Imposto orientamento {source_ori_name} su {len(target_samples)} sample?\n\n"
            f"Scope: {self._display_path(scope_folder)}\n"
            f"Include sottocartelle: {'si' if include_subfolders else 'no'}\n"
            f"Solo pending: {'si' if only_pending else 'no'}"
        )
        if not messagebox.askyesno("Conferma batch orientamento", msg):
            self._set_status("Batch orientamento annullato.")
            return

        changed = 0
        rect_applied = 0
        rect_missing = 0
        for sample in target_samples:
            self._set_manual_orientation_idx(sample, source_ori_idx)
            if self._apply_fss_rect_for_orientation(sample, source_ori_idx, overwrite_existing=True):
                rect_applied += 1
            else:
                rect_missing += 1
            changed += 1

        self._redraw()
        self._refresh_header()
        self._set_status(
            f"Batch orientamento completato: {changed} sample -> {source_ori_name}. Rect FSS applicato={rect_applied}, mancanti={rect_missing}."
        )

    def _corner_hit(self, rect: Rect, x: float, y: float, tol: float = 10.0) -> Optional[str]:
        t, l, b, r = rect
        corners = {
            "tl": self._image_to_canvas(l, t),
            "tr": self._image_to_canvas(r, t),
            "bl": self._image_to_canvas(l, b),
            "br": self._image_to_canvas(r, b),
        }
        for name, (cx, cy) in corners.items():
            if abs(cx - x) <= tol and abs(cy - y) <= tol:
                return name
        return None

    def on_left_press(self, event: tk.Event) -> None:
        sample = self._current_sample()
        img = self._load_image(sample)
        ix, iy = self._canvas_to_image(event.x, event.y)
        self.drag_start_img = (ix, iy)
        self.drag_start_rect = self._active_rect(sample)
        self.temp_draw_rect = None
        self.drag_corner = None

        active = self._active_rect(sample)
        if active is not None:
            corner = self._corner_hit(active, event.x, event.y)
            if corner is not None:
                self.drag_mode = "resize"
                self.drag_corner = corner
                return
            if _point_in_rect(ix, iy, active):
                self.drag_mode = "move"
                return
        self.drag_mode = "draw"

    def on_left_drag(self, event: tk.Event) -> None:
        sample = self._current_sample()
        img = self._load_image(sample)
        if self.drag_mode is None or self.drag_start_img is None:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        sx, sy = self.drag_start_img

        if self.drag_mode == "draw":
            t = min(sy, iy)
            b = max(sy, iy)
            l = min(sx, ix)
            r = max(sx, ix)
            self.temp_draw_rect = _clip_rect((t, l, b, r), width=img.width, height=img.height)
            self._redraw()
            return

        base = self.drag_start_rect
        if base is None:
            base = self._active_rect(sample)
        if base is None and (self.edit_mode_var.get() or "").strip().lower() == "symbol":
            base = sample.symbol_rect
        if base is None:
            return

        if self.drag_mode == "move":
            dy = iy - sy
            dx = ix - sx
            moved = _rect_translate(base, dy=dy, dx=dx, width=img.width, height=img.height)
            self._set_active_rect(sample, moved)
            self._set_decision(sample, "corrected")
            self._redraw()
            self._refresh_header()
            return

        if self.drag_mode == "resize":
            t, l, b, r = base
            corner = self.drag_corner or "br"
            if corner == "tl":
                t, l = iy, ix
            elif corner == "tr":
                t, r = iy, ix
            elif corner == "bl":
                b, l = iy, ix
            elif corner == "br":
                b, r = iy, ix
            resized = _clip_rect((t, l, b, r), width=img.width, height=img.height)
            self._set_active_rect(sample, resized)
            self._set_decision(sample, "corrected")
            self._redraw()
            self._refresh_header()
            return

    def on_left_release(self, _event: tk.Event) -> None:
        if self.drag_mode == "draw" and self.temp_draw_rect is not None:
            sample = self._current_sample()
            self._set_active_rect(sample, self.temp_draw_rect)
            self._set_decision(sample, "corrected")
            self.temp_draw_rect = None
            self._redraw()
            self._refresh_header()
        self.drag_mode = None
        self.drag_corner = None
        self.drag_start_img = None
        self.drag_start_rect = None

    def on_arrow_key(self, event: tk.Event) -> None:
        sample = self._current_sample()
        img = self._load_image(sample)
        rect = self._active_rect(sample)
        if rect is None:
            mode = (self.edit_mode_var.get() or "symbol").strip().lower()
            if mode != "symbol" or sample.symbol_rect is None:
                return
            rect = sample.symbol_rect

        state_mask = int(event.state)
        step = 10 if (state_mask & 0x0001) else 1  # Shift
        ctrl = bool(state_mask & 0x0004)  # Control

        if ctrl:
            t, l, b, r = rect
            if event.keysym == "Left":
                l -= step
            elif event.keysym == "Right":
                r += step
            elif event.keysym == "Up":
                t -= step
            elif event.keysym == "Down":
                b += step
            rect = _clip_rect((t, l, b, r), width=img.width, height=img.height)
        else:
            dy = 0
            dx = 0
            if event.keysym == "Left":
                dx = -step
            elif event.keysym == "Right":
                dx = step
            elif event.keysym == "Up":
                dy = -step
            elif event.keysym == "Down":
                dy = step
            rect = _rect_translate(rect, dy=dy, dx=dx, width=img.width, height=img.height)

        self._set_active_rect(sample, rect)
        self._set_decision(sample, "corrected")
        self._redraw()
        self._refresh_header()

    def _search_rect_for_orientation(self, sample: SampleRow, orientation_idx: int) -> Optional[Rect]:
        manual_rect = self._get_group_orientation_rect(sample, orientation_idx)
        if manual_rect is not None:
            return manual_rect
        return self._rect_from_fss_for_orientation(sample, orientation_idx)

    def _search_rects_map(self, sample: SampleRow) -> Dict[int, Rect]:
        out: Dict[int, Rect] = {}
        for orientation_idx in sorted(ORIENTATION_BY_IDX.keys()):
            rect = self._search_rect_for_orientation(sample, orientation_idx)
            if rect is not None:
                out[orientation_idx] = rect
        return out

    def export_csv(self) -> None:
        self._store_current_note()
        self.save_session()

        out_manual = self.output_dir / f"{self.manifest_path.stem}_manual_review.csv"
        out_train = self.output_dir / f"{self.manifest_path.stem}_train_ready.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        extra_fields = [
            "manual_decision",
            "manual_note",
            "manual_orientation_idx",
            "manual_orientation_name",
            "final_orientation_idx",
            "final_orientation_name",
            "manual_symbol_rect_top",
            "manual_symbol_rect_left",
            "manual_symbol_rect_bottom",
            "manual_symbol_rect_right",
            "corrected_symbol_rect_top",
            "corrected_symbol_rect_left",
            "corrected_symbol_rect_bottom",
            "corrected_symbol_rect_right",
            "search_rect_nf_top",
            "search_rect_nf_left",
            "search_rect_nf_bottom",
            "search_rect_nf_right",
            "search_rect_lr_top",
            "search_rect_lr_left",
            "search_rect_lr_bottom",
            "search_rect_lr_right",
            "search_rect_ud_top",
            "search_rect_ud_left",
            "search_rect_ud_bottom",
            "search_rect_ud_right",
            "search_rect_lrud_top",
            "search_rect_lrud_left",
            "search_rect_lrud_bottom",
            "search_rect_lrud_right",
            "final_search_rect_top",
            "final_search_rect_left",
            "final_search_rect_bottom",
            "final_search_rect_right",
            "final_symbol_rect_top",
            "final_symbol_rect_left",
            "final_symbol_rect_bottom",
            "final_symbol_rect_right",
            "train_use",
        ]

        manual_fields = list(self.fieldnames) + [f for f in extra_fields if f not in self.fieldnames]
        train_rows: List[Dict[str, str]] = []

        with out_manual.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=manual_fields)
            writer.writeheader()
            for sample in self.samples:
                row = dict(sample.raw)
                st = self._get_state(sample)
                decision = str(st.get("manual_decision", "") or "")
                note = str(st.get("manual_note", "") or "")
                manual_ori_idx = self._get_manual_orientation_idx(sample)
                manual_ori_name = "" if manual_ori_idx is None else ORIENTATION_BY_IDX.get(manual_ori_idx, "")
                final_ori_idx, final_ori_name = self._current_final_orientation(sample)
                corr = self._get_symbol_rect(sample)
                orig = sample.symbol_rect
                final = corr if corr is not None else orig
                search_rects = self._search_rects_map(sample)
                final_search = search_rects.get(final_ori_idx) if final_ori_idx is not None else None

                if decision == "reject":
                    final = None
                    train_use = "0"
                else:
                    train_use = "1" if final is not None else "0"

                row["manual_decision"] = decision
                row["manual_note"] = note
                row["manual_orientation_idx"] = "" if manual_ori_idx is None else str(manual_ori_idx)
                row["manual_orientation_name"] = manual_ori_name
                row["final_orientation_idx"] = "" if final_ori_idx is None else str(final_ori_idx)
                row["final_orientation_name"] = final_ori_name
                row["manual_symbol_rect_top"] = "" if corr is None else str(corr[0])
                row["manual_symbol_rect_left"] = "" if corr is None else str(corr[1])
                row["manual_symbol_rect_bottom"] = "" if corr is None else str(corr[2])
                row["manual_symbol_rect_right"] = "" if corr is None else str(corr[3])
                row["corrected_symbol_rect_top"] = "" if corr is None else str(corr[0])
                row["corrected_symbol_rect_left"] = "" if corr is None else str(corr[1])
                row["corrected_symbol_rect_bottom"] = "" if corr is None else str(corr[2])
                row["corrected_symbol_rect_right"] = "" if corr is None else str(corr[3])
                nf = search_rects.get(0)
                lr = search_rects.get(1)
                ud = search_rects.get(2)
                lrud = search_rects.get(3)
                row["search_rect_nf_top"] = "" if nf is None else str(nf[0])
                row["search_rect_nf_left"] = "" if nf is None else str(nf[1])
                row["search_rect_nf_bottom"] = "" if nf is None else str(nf[2])
                row["search_rect_nf_right"] = "" if nf is None else str(nf[3])
                row["search_rect_lr_top"] = "" if lr is None else str(lr[0])
                row["search_rect_lr_left"] = "" if lr is None else str(lr[1])
                row["search_rect_lr_bottom"] = "" if lr is None else str(lr[2])
                row["search_rect_lr_right"] = "" if lr is None else str(lr[3])
                row["search_rect_ud_top"] = "" if ud is None else str(ud[0])
                row["search_rect_ud_left"] = "" if ud is None else str(ud[1])
                row["search_rect_ud_bottom"] = "" if ud is None else str(ud[2])
                row["search_rect_ud_right"] = "" if ud is None else str(ud[3])
                row["search_rect_lrud_top"] = "" if lrud is None else str(lrud[0])
                row["search_rect_lrud_left"] = "" if lrud is None else str(lrud[1])
                row["search_rect_lrud_bottom"] = "" if lrud is None else str(lrud[2])
                row["search_rect_lrud_right"] = "" if lrud is None else str(lrud[3])
                row["final_search_rect_top"] = "" if final_search is None else str(final_search[0])
                row["final_search_rect_left"] = "" if final_search is None else str(final_search[1])
                row["final_search_rect_bottom"] = "" if final_search is None else str(final_search[2])
                row["final_search_rect_right"] = "" if final_search is None else str(final_search[3])
                row["final_symbol_rect_top"] = "" if final is None else str(final[0])
                row["final_symbol_rect_left"] = "" if final is None else str(final[1])
                row["final_symbol_rect_bottom"] = "" if final is None else str(final[2])
                row["final_symbol_rect_right"] = "" if final is None else str(final[3])
                row["train_use"] = train_use
                writer.writerow(row)

                if train_use == "1" and final is not None:
                    train_row = dict(row)
                    train_row["symbol_rect_top"] = str(final[0])
                    train_row["symbol_rect_left"] = str(final[1])
                    train_row["symbol_rect_bottom"] = str(final[2])
                    train_row["symbol_rect_right"] = str(final[3])
                    if final_ori_idx is not None:
                        train_row["orientation_idx"] = str(final_ori_idx)
                    if final_ori_name:
                        train_row["orientation_name"] = final_ori_name
                    train_rows.append(train_row)

        with out_train.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=manual_fields)
            writer.writeheader()
            writer.writerows(train_rows)

        messagebox.showinfo(
            "Export completed",
            f"Manual review CSV:\n{out_manual}\n\nTraining-ready CSV:\n{out_train}\n\nRows for training: {len(train_rows)}",
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive review/fix GUI for symbol rects.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbol_detection_dataset_v2/symbol_detection_manifest.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbol_detection_dataset_v2/manual_review"),
    )
    p.add_argument(
        "--session-path",
        type=Path,
        default=Path("artifacts/20_datasets/orientation_symbol_detection_dataset_v2/manual_review/review_state.json"),
    )
    p.add_argument("--only-review-needed", action="store_true", help="Start from rows with auto_accept != 1.")
    p.add_argument("--max-rows", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    session_path = args.session_path.expanduser().resolve()

    samples, fieldnames = _read_manifest(
        manifest_path,
        only_review_needed=bool(args.only_review_needed),
        max_rows=max(0, int(args.max_rows)),
    )
    if not samples:
        raise RuntimeError("No samples found. Check manifest path/filters.")

    root = tk.Tk()
    app = RectReviewApp(
        root=root,
        samples=samples,
        fieldnames=fieldnames,
        manifest_path=manifest_path,
        output_dir=output_dir,
        session_path=session_path,
    )
    # Keep reference to avoid linters complaining about unused variable.
    _ = app
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
