"""Model inference for the app: vendor classifier + echo rectangle, with vendor routing.

Uses the checkpoints of the active pipeline as they are, with the same preprocessing as
training (`tools/ultrasound/train_ultrasound_rect_net.py`, `..._vendor_classifier.py`):

* vendor: ResNet18 classifier, 384 px, mean of the per-image softmax over a sample of frames;
* rect: ResNet18 regressor, 320 px, sigmoid output normalised to the image, one box per frame,
  folder box = per-component median (same aggregation as the pipeline);
* routing: the BK-specialised checkpoint is used only for vendor BK with confidence >= 0.70,
  exactly as `--rect-vendor-min-confidence` does in the pipeline.

Not included here: the red-rect override (`rect_red_pipeline`) that the pipeline can apply on
top of the median box, and the OSD rotation. Both belong to later stages.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

ULTRASOUND_DIR = Path(__file__).resolve().parents[1] / "ultrasound"

VENDOR_IMAGE_SIZE = 384
RECT_IMAGE_SIZE = 320
PROBE_IMAGE_SIZE = 320
LT_IMAGE_SIZE = 320
SU_GIU_IMAGE_SIZE = 256
RECT_VENDOR_MIN_CONFIDENCE = 0.70
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_line13_map(path: Path) -> Dict[str, Path]:
    """Per-vendor networks for #13 RECT_NAME_ECHO, as the pipeline's --line13-vendor-map."""
    try:
        import json

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, Path] = {}
    for vendor, target in raw.items():
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = (Path(path).parent / candidate).resolve()
        if candidate.exists():
            out[str(vendor)] = candidate
    return out


@dataclass
class ModelPaths:
    vendor: Path
    rect_global: Path
    rect_by_vendor: Dict[str, Path] = field(default_factory=dict)
    probe: Optional[Path] = None
    lt: Optional[Path] = None
    su_giu: Optional[Path] = None
    needle: Optional[Path] = None
    line13_by_vendor: Dict[str, Path] = field(default_factory=dict)
    line14: Optional[Path] = None

    @classmethod
    def from_active_pipeline(cls, root: Path, models_30: Optional[Path] = None) -> "ModelPaths":
        models = Path(root) / "models"
        # The L/T checkpoint lives outside the active-pipeline folder: this is the one the
        # pipeline uses by default (predict_fss_head_from_acquisitions.py:6371).
        lt_root = Path(models_30) if models_30 else Path(root).parents[1] / "30_models"
        return cls(
            vendor=models / "vendor_training_no_negative_v2_power" / "best_model.pt",
            rect_global=models / "rect_training_e40_run2" / "best_model.pt",
            rect_by_vendor={"BK": models / "rect_training_vendor_bk" / "best_model.pt"},
            probe=models / "probe_training_no_negative_v1" / "best_model.pt",
            lt=lt_root / "lt_training_transrectal_rect_only_v2_trecall" / "best_model.pt",
            su_giu=(
                Path(root).parents[1]
                / "20_datasets/us_orientation_binary_20260320/rect_pipeline/model_su_giu_rect/best_model.pt"
            ),
            # Il classificatore del materiale di calibrazione aghi sta fuori dalla pipeline
            # attiva: e' un blocco a se', e se manca l'app funziona come prima.
            needle=(
                Path(root).parents[1]
                / "91_needle_models/resnet18_256_cpu/best_model.pt"
            ),
            line13_by_vendor=_load_line13_map(Path(root) / "maps" / "vendor_line13_template_map.json"),
            # #14 e' una rete sola per tutti i vendor: separare per marchio non
            # guadagna nulla (13 vittorie, 12 sconfitte, 23 pari su 48 cartelle).
            line14=models / "probe_template_line14" / "best_model.pt",
        )

    def missing(self) -> List[str]:
        """Only vendor and rect are required: probe and L/T degrade gracefully."""
        absent = [str(p) for p in [self.vendor, self.rect_global] if not p.exists()]
        absent += [str(p) for p in self.rect_by_vendor.values() if not p.exists()]
        return absent


class Engine:
    """Loads torch and the checkpoints on first use, then keeps them in memory."""

    def __init__(self, paths: ModelPaths, device: Optional[str] = None):
        self.paths = paths
        self._device_name = device
        self._torch = None
        self._device = None
        self._vendor = None
        self._vendor_classes: List[str] = []
        self._rect: Dict[str, object] = {}
        self._probe = None
        self._probe_classes: List[str] = []
        self._lt = None
        self._lt_classes: List[str] = []
        self._su_giu = None
        self._su_giu_classes: List[str] = []
        self._needle = None
        self._needle_size = 0
        self._line13: Dict[str, tuple] = {}
        self._line14 = None

    # -- setup -------------------------------------------------------------
    def _setup(self):
        if self._torch is not None:
            return self._torch
        import sys

        if str(ULTRASOUND_DIR) not in sys.path:
            sys.path.insert(0, str(ULTRASOUND_DIR))
        import torch

        self._torch = torch
        if self._device_name:
            self._device = torch.device(self._device_name)
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        return torch

    @property
    def device(self) -> str:
        self._setup()
        return str(self._device)

    def _load_vendor(self):
        if self._vendor is not None:
            return self._vendor
        torch = self._setup()
        from train_ultrasound_vendor_classifier import VendorClassifier

        checkpoint = torch.load(self.paths.vendor, map_location="cpu", weights_only=False)
        self._vendor_classes = list(checkpoint.get("class_names") or [])
        model = VendorClassifier(num_classes=len(self._vendor_classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._vendor = model
        return model

    def _load_rect(self, key: str, path: Path):
        if key in self._rect:
            return self._rect[key]
        torch = self._setup()
        from train_ultrasound_rect_net import RectRegressor

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = RectRegressor(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._rect[key] = model
        return model

    def _load_probe(self):
        if self._probe is not None:
            return self._probe
        if not self.paths.probe or not self.paths.probe.exists():
            return None
        torch = self._setup()
        from train_ultrasound_probe_classifier import ProbeClassifier

        checkpoint = torch.load(self.paths.probe, map_location="cpu", weights_only=False)
        self._probe_classes = list(checkpoint.get("class_names") or [])
        model = ProbeClassifier(num_classes=len(self._probe_classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._probe = model
        return model

    def _load_lt(self):
        if self._lt is not None:
            return self._lt
        if not self.paths.lt or not self.paths.lt.exists():
            return None
        torch = self._setup()
        from torch import nn
        from torchvision.models import resnet18

        class LTRectClassifier(nn.Module):
            """Same architecture the pipeline builds for this checkpoint."""

            def __init__(self) -> None:
                super().__init__()
                backbone = resnet18(weights=None)
                in_features = backbone.fc.in_features
                backbone.fc = nn.Identity()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.30),
                    nn.Linear(256, 2),
                )

            def forward(self, x):  # noqa: ANN001
                return self.head(self.backbone(x))

        checkpoint = torch.load(self.paths.lt, map_location="cpu", weights_only=False)
        self._lt_classes = [str(name).strip().upper() for name in (checkpoint.get("class_names") or ["L", "T"])]
        model = LTRectClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._lt = model
        return model

    def _load_su_giu(self):
        if self._su_giu is not None:
            return self._su_giu
        if not self.paths.su_giu or not self.paths.su_giu.exists():
            return None
        torch = self._setup()
        from torch import nn
        from torchvision.models import resnet18

        class SuGiuRectClassifier(nn.Module):
            """Same architecture the pipeline builds for this checkpoint."""

            def __init__(self) -> None:
                super().__init__()
                model = resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 2)
                self.model = model

            def forward(self, x):  # noqa: ANN001
                return self.model(x)

        checkpoint = torch.load(self.paths.su_giu, map_location="cpu", weights_only=False)
        self._su_giu_classes = [str(name) for name in (checkpoint.get("class_names") or ["su", "giu"])]
        model = SuGiuRectClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._su_giu = model
        return model

    # -- tensors -----------------------------------------------------------
    def _tensor(self, path: Path, size: int):
        torch = self._setup()
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            resized = TF.resize(
                image, size=[size, size], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
        tensor = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
        return tensor, width, height

    # -- vendor ------------------------------------------------------------
    def predict_vendor(self, paths: Sequence[Path], batch: int = 8) -> Dict:
        torch = self._setup()
        model = self._load_vendor()
        totals = None
        used = 0
        for start in range(0, len(paths), batch):
            tensors = []
            for path in paths[start : start + batch]:
                try:
                    tensor, _, _ = self._tensor(path, VENDOR_IMAGE_SIZE)
                except Exception:
                    continue
                tensors.append(tensor)
            if not tensors:
                continue
            with torch.no_grad():
                probs = torch.softmax(model(torch.stack(tensors).to(self._device)), dim=1)
            summed = probs.sum(dim=0).cpu()
            totals = summed if totals is None else totals + summed
            used += len(tensors)
        if totals is None or not used:
            return {"vendor": None, "confidence": None, "top": [], "images": 0}

        mean = (totals / used).tolist()
        ranked = sorted(zip(self._vendor_classes, mean), key=lambda item: -item[1])
        return {
            "vendor": ranked[0][0],
            "confidence": round(float(ranked[0][1]), 4),
            "margin": round(float(ranked[0][1] - ranked[1][1]), 4) if len(ranked) > 1 else None,
            "top": [{"vendor": name, "prob": round(float(prob), 4)} for name, prob in ranked[:3]],
            "images": used,
        }

    def _crop_tensor(self, path: Path, rect: Dict[str, int], size: int):
        """Rect crop, resized square and normalised: same convention as the pipeline."""
        torch = self._setup()
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            left = max(0, min(int(rect["left"]), width - 1))
            top = max(0, min(int(rect["top"]), height - 1))
            right = max(left + 1, min(int(rect["right"]), width))
            bottom = max(top + 1, min(int(rect["bottom"]), height))
            crop = image.crop((left, top, right, bottom))
            resized = TF.resize(
                crop, size=[size, size], interpolation=InterpolationMode.BILINEAR, antialias=True
            )
        return TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)

    # -- probe -------------------------------------------------------------
    def predict_probe(self, paths: Sequence[Path], batch: int = 8) -> Dict:
        """The classifier's classes are the probe IDs, so this lands straight on #03."""
        torch = self._setup()
        model = self._load_probe()
        if model is None:
            return {"probe_id": None, "reason": "checkpoint probe assente"}
        totals, used = None, 0
        for start in range(0, len(paths), batch):
            tensors = []
            for path in paths[start : start + batch]:
                try:
                    tensor, _, _ = self._tensor(path, PROBE_IMAGE_SIZE)
                except Exception:
                    continue
                tensors.append(tensor)
            if not tensors:
                continue
            with torch.no_grad():
                probs = torch.softmax(model(torch.stack(tensors).to(self._device)), dim=1)
            summed = probs.sum(dim=0).cpu()
            totals = summed if totals is None else totals + summed
            used += len(tensors)
        if totals is None or not used:
            return {"probe_id": None, "reason": "nessuna immagine analizzata"}
        mean = (totals / used).tolist()
        ranked = sorted(zip(self._probe_classes, mean), key=lambda item: -item[1])
        return {
            "probe_id": int(ranked[0][0]) if str(ranked[0][0]).lstrip("-").isdigit() else None,
            "confidence": round(float(ranked[0][1]), 4),
            "top": [
                {"probe_id": name, "prob": round(float(prob), 4)} for name, prob in ranked[:3]
            ],
            "images": used,
        }

    # -- piano L/T ---------------------------------------------------------
    def predict_lt(self, paths: Sequence[Path], rect: Dict[str, int], batch: int = 8) -> Dict:
        """L/T on the rect crops, as the pipeline does."""
        torch = self._setup()
        model = self._load_lt()
        if model is None:
            return {"plane": None, "reason": "checkpoint L/T assente"}
        counts = {"L": 0, "T": 0}
        confidences = []
        for start in range(0, len(paths), batch):
            tensors = []
            for path in paths[start : start + batch]:
                try:
                    tensors.append(self._crop_tensor(path, rect, LT_IMAGE_SIZE))
                except Exception:
                    continue
            if not tensors:
                continue
            with torch.no_grad():
                probs = torch.softmax(model(torch.stack(tensors).to(self._device)), dim=1).cpu()
            for row in probs:
                index = int(row.argmax())
                label = self._lt_classes[index] if index < len(self._lt_classes) else str(index)
                counts[label] = counts.get(label, 0) + 1
                confidences.append(float(row[index]))
        decided = counts.get("L", 0) + counts.get("T", 0)
        if not decided:
            return {"plane": None, "reason": "nessun crop analizzato"}
        plane = "L" if counts.get("L", 0) >= counts.get("T", 0) else "T"
        return {
            "plane": plane,
            "counts": counts,
            "share": round(max(counts.values()) / decided, 3),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "images": decided,
            "checkpoint": str(self.paths.lt),
        }

    def predict_lt_each(self, paths: Sequence[Path], rect: Dict[str, int],
                        batch: int = 8, progress=None) -> List[Dict]:  # noqa: ANN001
        """L/T **per immagine**, non il voto della cartella.

        `predict_lt` risponde alla domanda "di che piano e' questa acquisizione". Ma una
        cartella puo' contenerne due, ed e' proprio il caso da cui nascono due progetti: li
        serve sapere di che piano e' ogni singolo fotogramma.
        """
        torch = self._setup()
        model = self._load_lt()
        if model is None:
            return []
        righe: List[Dict] = []
        for start in range(0, len(paths), batch):
            fetta = list(paths[start : start + batch])
            tensori, validi = [], []
            for path in fetta:
                try:
                    tensori.append(self._crop_tensor(path, rect, LT_IMAGE_SIZE))
                    validi.append(path)
                except Exception:  # noqa: BLE001, PERF203
                    righe.append({"path": str(path), "plane": None, "confidence": None})
            if tensori:
                with torch.no_grad():
                    probs = torch.softmax(
                        model(torch.stack(tensori).to(self._device)), dim=1).cpu()
                for path, row in zip(validi, probs):
                    indice = int(row.argmax())
                    etichetta = (self._lt_classes[indice] if indice < len(self._lt_classes)
                                 else str(indice))
                    righe.append({"path": str(path), "plane": etichetta,
                                  "confidence": round(float(row[indice]), 4)})
            if progress is not None:
                progress(min(start + batch, len(paths)), len(paths))
        return righe

    # -- SU/GIU ------------------------------------------------------------
    def predict_su_giu(self, paths: Sequence[Path], rect: Dict[str, int], batch: int = 8) -> Dict:
        """Up/down per frame on the rect crops: one row per image, as the scale stage wants."""
        torch = self._setup()
        model = self._load_su_giu()
        if model is None:
            return {"rows": [], "reason": "checkpoint su/giu assente"}

        rows: List[Dict] = []
        for start in range(0, len(paths), batch):
            chunk = paths[start : start + batch]
            tensors, kept = [], []
            for path in chunk:
                try:
                    tensors.append(self._crop_tensor(path, rect, SU_GIU_IMAGE_SIZE))
                except Exception:
                    continue
                kept.append(path)
            if not tensors:
                continue
            with torch.no_grad():
                probs = torch.softmax(model(torch.stack(tensors).to(self._device)), dim=1).cpu()
            for index, path in enumerate(kept):
                row = probs[index]
                best = int(row.argmax())
                label = (
                    self._su_giu_classes[best]
                    if best < len(self._su_giu_classes)
                    else str(best)
                )
                rows.append(
                    {
                        "image_path": Path(path).as_posix(),
                        "pred_label": label,
                        "confidence": round(float(row[best]), 4),
                        "crop_left": int(rect["left"]),
                        "crop_top": int(rect["top"]),
                        "crop_right": int(rect["right"]),
                        "crop_bottom": int(rect["bottom"]),
                    }
                )

        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["pred_label"]] = counts.get(row["pred_label"], 0) + 1
        majority = max(counts, key=counts.get) if counts else None
        return {
            "rows": rows,
            "counts": counts,
            "majority": majority,
            "images": len(rows),
            "mean_confidence": (
                round(sum(r["confidence"] for r in rows) / len(rows), 4) if rows else None
            ),
            "checkpoint": str(self.paths.su_giu),
        }

    # -- #13 template ecografo ---------------------------------------------
    # -- aghi --------------------------------------------------------------
    def _load_needle(self):
        """Il classificatore del materiale per la sessione di calibrazione della guida aghi.

        Riconosce i fotogrammi che servono a `WdgPageCalibration` del vecchio ESIBuilder:
        l'ago ripreso in acqua, quello che l'operatore ricalca per ricavare angolo e
        distanza dal centro. Opzionale come sonda e L/T: senza checkpoint l'app non
        mostra la scheda e tutto il resto funziona uguale.
        """
        if self._needle is not None:
            return self._needle
        torch = self._setup()
        path = self.paths.needle
        if not path or not Path(path).is_file():
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self._needle_size = int(checkpoint.get("image_size") or 256)
        arch = str(checkpoint.get("arch") or "resnet18")

        from torchvision import models as tv

        if arch == "resnet18":
            model = tv.resnet18(weights=None)
            model.fc = torch.nn.Sequential(torch.nn.Dropout(0.0),
                                           torch.nn.Linear(model.fc.in_features, 1))
        elif arch == "resnet50":
            model = tv.resnet50(weights=None)
            model.fc = torch.nn.Sequential(torch.nn.Dropout(0.0),
                                           torch.nn.Linear(model.fc.in_features, 1))
        elif arch == "efficientnet_b0":
            model = tv.efficientnet_b0(weights=None)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.0), torch.nn.Linear(model.classifier[1].in_features, 1))
        elif arch == "convnext_tiny":
            model = tv.convnext_tiny(weights=None)
            model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 1)
        else:
            return None
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._needle = model
        return model

    def needle_available(self) -> bool:
        return bool(self.paths.needle and Path(self.paths.needle).is_file())

    def _needle_tensor(self, path: Path, rect: Dict[str, int], size: int, margin_pct: float):
        """Ritaglio del rettangolo, lato lungo a `size`, riempito di nero fino al quadrato.

        Non usa `_crop_tensor`: quello schiaccia il ritaglio in un quadrato, e schiacciare
        cambia l'inclinazione degli aghi, che e' esattamente il segno da riconoscere. Il
        modello e' stato addestrato con questo riempimento, e la preparazione a inferenza
        deve essere la stessa.
        """
        self._setup()
        from PIL import Image
        from torchvision.transforms import functional as TF

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            left = max(0, min(int(rect["left"]), width - 1))
            top = max(0, min(int(rect["top"]), height - 1))
            right = max(left + 1, min(int(rect["right"]), width))
            bottom = max(top + 1, min(int(rect["bottom"]), height))
            mx = (right - left) * margin_pct / 100.0
            my = (bottom - top) * margin_pct / 100.0
            crop = image.crop((
                int(max(0, round(left - mx))), int(max(0, round(top - my))),
                int(min(width, round(right + mx))), int(min(height, round(bottom + my))),
            ))
            crop.thumbnail((size, size), Image.LANCZOS)
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
        return TF.normalize(TF.to_tensor(canvas), IMAGENET_MEAN, IMAGENET_STD)

    def predict_needle(
        self,
        paths: Sequence[Path],
        rect: Dict[str, int],
        batch: int = 8,
        margin_pct: float = 2.0,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        """Punteggio per fotogramma: e' materiale per la calibrazione della guida aghi?"""
        torch = self._setup()
        model = self._load_needle()
        if model is None:
            return {"available": False, "scores": {}}

        size = self._needle_size or 256
        scores: Dict[str, float] = {}
        done = 0
        for start in range(0, len(paths), batch):
            chunk = list(paths[start : start + batch])
            tensors, names = [], []
            for path in chunk:
                try:
                    tensors.append(self._needle_tensor(path, rect, size, margin_pct))
                except Exception:
                    continue
                names.append(path)
            if tensors:
                with torch.no_grad():
                    logits = model(torch.stack(tensors).to(self._device)).squeeze(1).float()
                    probabilities = torch.sigmoid(logits).cpu().tolist()
                for path, score in zip(names, probabilities):
                    scores[str(path)] = float(score)
            done += len(chunk)
            if progress:
                progress(done, len(paths))
        return {"available": True, "scores": scores, "model": str(self.paths.needle)}

    def _load_line13(self, vendor: str):
        if vendor in self._line13:
            return self._line13[vendor]
        path = self.paths.line13_by_vendor.get(vendor)
        if path is None:
            return None
        torch = self._setup()
        from train_ultrasound_rect_net import RectRegressor

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        size = int((checkpoint.get("args") or {}).get("image_size") or 384)
        model = RectRegressor(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self._device)
        self._line13[vendor] = (model, size, path)
        return self._line13[vendor]

    def predict_line13(
        self,
        paths: Sequence[Path],
        vendor: Optional[str],
        batch: int = 8,
        consensus_min_iou: float = 0.35,
        consensus_min_keep: int = 3,
    ) -> Dict:
        """#13 with the vendor-specific network, then the IoU consensus of the pipeline.

        Not replicated here: the NCC template gate and the dark-border trim that the pipeline
        can apply after the consensus.
        """
        torch = self._setup()
        if not vendor:
            return {"box": None, "reason": "vendor non riconosciuto"}
        loaded = self._load_line13(vendor)
        if loaded is None:
            return {"box": None, "reason": f"nessun modello #13 per il vendor {vendor}"}
        model, size, path = loaded

        boxes: List[Dict] = []
        for start in range(0, len(paths), batch):
            chunk = paths[start : start + batch]
            tensors, sizes = [], []
            for image_path in chunk:
                try:
                    tensor, width, height = self._tensor(image_path, size)
                except Exception:
                    continue
                tensors.append(tensor)
                sizes.append((width, height))
            if not tensors:
                continue
            with torch.no_grad():
                predicted = model(torch.stack(tensors).to(self._device)).cpu()
            for index, (width, height) in enumerate(sizes):
                x1, y1, x2, y2 = [float(v) for v in predicted[index].tolist()]
                left, right = sorted((x1 * width, x2 * width))
                top, bottom = sorted((y1 * height, y2 * height))
                boxes.append({"top": int(round(top)), "left": int(round(left)),
                              "bottom": int(round(bottom)), "right": int(round(right))})
        if not boxes:
            return {"box": None, "reason": "nessuna immagine analizzata"}

        median = {
            side: int(round(statistics.median(box[side] for box in boxes)))
            for side in ("top", "left", "bottom", "right")
        }
        # IoU consensus: drop the frames that disagree with the folder box, then median again.
        kept = [box for box in boxes if _iou(box, median) >= consensus_min_iou]
        if len(kept) >= consensus_min_keep:
            median = {
                side: int(round(statistics.median(box[side] for box in kept)))
                for side in ("top", "left", "bottom", "right")
            }
        else:
            kept = boxes

        return {
            "box": median,
            "vendor": vendor,
            "model": str(path),
            "images": len(boxes),
            "images_kept": len(kept),
            "agreement_iou": _median_iou(kept, median),
            "source": "vendor_line13_model",
        }

    # -- #14 template sonda -------------------------------------------------
    def _load_line14(self):
        if self._line14 is not None:
            return self._line14
        path = self.paths.line14
        if path is None or not Path(path).is_file():
            return None
        torch = self._setup()
        from train_probe_template_net import ProbeTemplateNet

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ProbeTemplateNet(pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval().to(self._device)
        self._line14 = (model, int(checkpoint.get("image_size") or 512), Path(path))
        return self._line14

    def _tensor_letterbox(self, path: Path, size: int):
        """Come il trainer: si rimpicciolisce mantenendo le proporzioni e si appoggia
        in alto a sinistra su una tela quadrata nera. Il resize quadrato di `_tensor`
        deformerebbe i fotogrammi verticali (1024x1280) e sposterebbe il box."""
        self._setup()
        from PIL import Image
        from torchvision.transforms import functional as TF

        with Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            scale = size / max(width, height)
            resized = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR
            )
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            canvas.paste(resized, (0, 0))
        tensor = TF.normalize(TF.to_tensor(canvas), IMAGENET_MEAN, IMAGENET_STD)
        return tensor, width, height, scale

    def predict_line14(
        self,
        paths: Sequence[Path],
        batch: int = 8,
        min_score: float = 0.20,
        consensus_min_iou: float = 0.35,
        consensus_min_keep: int = 3,
    ) -> Dict:
        """#14 con la rete unica: picco della heatmap per fotogramma, poi mediana e
        consenso IoU come per la #13.

        `min_score` scarta i fotogrammi in cui la rete non ha trovato la scritta (le
        schermate senza nome sonda, e le UI mai viste in training). Misurato su 61
        cartelle di test mai viste: sopra 0.20 il box e' corretto nel 91% dei casi,
        sotto quasi sempre sbagliato. E' la soglia da tarare sull'uso reale.
        """
        torch = self._setup()
        loaded = self._load_line14()
        if loaded is None:
            return {"box": None, "reason": "modello #14 non installato"}
        model, size, path = loaded
        from train_probe_template_net import decode_batch

        boxes: List[Dict] = []
        scores: List[float] = []
        scanned = 0
        for start in range(0, len(paths), batch):
            tensors, geometry = [], []
            for image_path in paths[start : start + batch]:
                try:
                    tensor, width, height, scale = self._tensor_letterbox(image_path, size)
                except Exception:
                    continue
                tensors.append(tensor)
                geometry.append((width, height, scale))
            if not tensors:
                continue
            with torch.no_grad():
                heat, offset, extent = model(torch.stack(tensors).to(self._device))
            frame_scores, frame_boxes = decode_batch(heat, offset, extent)
            scanned += len(geometry)
            for index, (width, height, scale) in enumerate(geometry):
                score = float(frame_scores[index])
                if score < min_score:
                    continue
                x1, y1, x2, y2 = [float(v) / scale for v in frame_boxes[index]]
                left, right = sorted((x1, x2))
                top, bottom = sorted((y1, y2))
                boxes.append({
                    "top": int(round(max(0.0, top))),
                    "left": int(round(max(0.0, left))),
                    "bottom": int(round(min(float(height), bottom))),
                    "right": int(round(min(float(width), right))),
                })
                scores.append(score)

        if not boxes:
            return {
                "box": None,
                "images": scanned,
                "images_kept": 0,
                "source": "line14_model",
                "model": str(path),
                "reason": "la rete non ha trovato la scritta della sonda in nessun fotogramma",
            }

        median = {
            side: int(round(statistics.median(box[side] for box in boxes)))
            for side in ("top", "left", "bottom", "right")
        }
        kept = [box for box in boxes if _iou(box, median) >= consensus_min_iou]
        if len(kept) >= consensus_min_keep:
            median = {
                side: int(round(statistics.median(box[side] for box in kept)))
                for side in ("top", "left", "bottom", "right")
            }
        else:
            kept = boxes

        return {
            "box": median,
            "images": scanned,
            "images_kept": len(kept),
            "score": round(statistics.median(scores), 4),
            "agreement_iou": _median_iou(kept, median),
            "model": str(path),
            "source": "line14_model",
            "reason": "",
        }

    # -- rect --------------------------------------------------------------
    def predict_rect(
        self,
        paths: Sequence[Path],
        vendor: Optional[str] = None,
        vendor_confidence: Optional[float] = None,
        batch: int = 8,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        torch = self._setup()

        key, model_path, source = "global", self.paths.rect_global, "global"
        if vendor and vendor in self.paths.rect_by_vendor:
            if (vendor_confidence or 0.0) >= RECT_VENDOR_MIN_CONFIDENCE:
                key, model_path, source = vendor, self.paths.rect_by_vendor[vendor], "vendor_specialized"
            else:
                source = "global_low_vendor_conf"
        model = self._load_rect(key, model_path)

        boxes: List[Dict] = []
        for start in range(0, len(paths), batch):
            chunk = paths[start : start + batch]
            tensors, sizes, names = [], [], []
            for path in chunk:
                try:
                    tensor, width, height = self._tensor(path, RECT_IMAGE_SIZE)
                except Exception:
                    continue
                tensors.append(tensor)
                sizes.append((width, height))
                names.append(path)
            if not tensors:
                continue
            with torch.no_grad():
                predicted = model(torch.stack(tensors).to(self._device)).cpu()
            for index, (width, height) in enumerate(sizes):
                x1, y1, x2, y2 = [float(v) for v in predicted[index].tolist()]
                left, right = sorted((x1 * width, x2 * width))
                top, bottom = sorted((y1 * height, y2 * height))
                boxes.append(
                    {
                        "name": names[index].name,
                        "path": str(names[index]),
                        "top": int(round(top)),
                        "left": int(round(left)),
                        "bottom": int(round(bottom)),
                        "right": int(round(right)),
                        "size": [width, height],
                    }
                )
            if progress:
                progress(min(start + batch, len(paths)), len(paths))

        if not boxes:
            return {"rect_echo": None, "source": source, "images": 0, "boxes": []}

        median = {
            side: int(round(statistics.median(box[side] for box in boxes)))
            for side in ("top", "left", "bottom", "right")
        }
        agreement = _median_iou(boxes, median)
        return {
            "rect_echo": median,
            "source": source,
            "model": str(model_path),
            "images": len(boxes),
            "agreement_iou": agreement,
            "spread_px": {
                side: int(
                    round(
                        max(box[side] for box in boxes) - min(box[side] for box in boxes)
                    )
                )
                for side in ("top", "left", "bottom", "right")
            },
            "boxes": boxes[:40],
        }


def _iou(box: Dict, reference: Dict) -> float:
    inter_w = min(box["right"], reference["right"]) - max(box["left"], reference["left"])
    inter_h = min(box["bottom"], reference["bottom"]) - max(box["top"], reference["top"])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = (box["right"] - box["left"]) * (box["bottom"] - box["top"])
    area_b = (reference["right"] - reference["left"]) * (reference["bottom"] - reference["top"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _median_iou(boxes: List[Dict], reference: Dict) -> float:
    """How much the per-image boxes agree with the folder box: a confidence proxy."""
    values = []
    for box in boxes:
        inter_w = min(box["right"], reference["right"]) - max(box["left"], reference["left"])
        inter_h = min(box["bottom"], reference["bottom"]) - max(box["top"], reference["top"])
        if inter_w <= 0 or inter_h <= 0:
            values.append(0.0)
            continue
        inter = inter_w * inter_h
        area_a = (box["right"] - box["left"]) * (box["bottom"] - box["top"])
        area_b = (reference["right"] - reference["left"]) * (reference["bottom"] - reference["top"])
        union = area_a + area_b - inter
        values.append(inter / union if union > 0 else 0.0)
    return round(statistics.median(values), 4) if values else 0.0


def sample_paths(paths: Sequence[Path], limit: int) -> List[Path]:
    """Uniform sample: the pipeline runs the rect on every frame, the app trades a bit of
    accuracy for an interactive wait."""
    paths = list(paths)
    if limit <= 0 or len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(index * step)] for index in range(limit)]
