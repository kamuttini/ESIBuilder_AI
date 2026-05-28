#!/usr/bin/env python3
"""Local web UI (drag&drop) for single-image rotation check."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from flask import Flask, jsonify, render_template_string, request


HTML_PAGE = """<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rotation Check - Single Image</title>
  <style>
    :root {
      --bg: #f2f4f8;
      --card: #ffffff;
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --primary: #0f766e;
      --primary-2: #115e59;
      --good: #166534;
      --warn: #92400e;
      --bad: #991b1b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 10% 0%, #dbeafe 0, #f2f4f8 45%);
      min-height: 100vh;
      padding: 20px;
    }
    .container { max-width: 980px; margin: 0 auto; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 14px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    h1 { margin: 0 0 8px; font-size: 24px; }
    .muted { color: var(--muted); font-size: 14px; }
    .dropzone {
      border: 2px dashed #9ca3af;
      border-radius: 12px;
      padding: 22px;
      text-align: center;
      background: #f9fafb;
      transition: 0.15s ease;
      cursor: pointer;
    }
    .dropzone.dragover {
      border-color: var(--primary);
      background: #ecfeff;
      transform: translateY(-1px);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    @media (max-width: 900px) {
      .row { grid-template-columns: 1fr; }
    }
    label { font-size: 13px; color: var(--muted); display: block; margin-bottom: 4px; }
    input[type="number"], select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
      background: white;
    }
    .actions { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    button {
      border: 1px solid var(--primary);
      background: var(--primary);
      color: white;
      border-radius: 8px;
      padding: 9px 13px;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      color: var(--primary-2);
      background: #e6fffb;
      border-color: #99f6e4;
    }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .status { font-size: 14px; margin-top: 8px; }
    .status.good { color: var(--good); }
    .status.warn { color: var(--warn); }
    .status.bad { color: var(--bad); }
    .result-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    @media (max-width: 900px) {
      .result-grid { grid-template-columns: 1fr; }
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fff;
      font-size: 14px;
    }
    .pill b { font-size: 16px; display: block; margin-top: 2px; }
    .preview-wrap { margin-top: 12px; }
    .preview-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    @media (max-width: 900px) {
      .preview-grid { grid-template-columns: 1fr; }
    }
    .preview-col-title {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .preview {
      max-width: 100%;
      max-height: 360px;
      border: 1px solid var(--line);
      border-radius: 10px;
      display: none;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0b1220;
      color: #d1e7ff;
      padding: 12px;
      border-radius: 10px;
      font-size: 12px;
      max-height: 300px;
      overflow: auto;
    }
    .file-name { margin-top: 8px; font-size: 13px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>Controllo Rotazione Immagine</h1>
      <div class="muted">Trascina un file immagine o clicca nell'area sotto.</div>
    </div>

    <div class="card">
      <div id="dropzone" class="dropzone">
        <div><b>Drag & Drop immagine qui</b></div>
        <div class="muted">oppure clicca per selezionare il file</div>
        <input id="fileInput" type="file" accept="image/*" style="display:none;">
      </div>
      <div id="fileName" class="file-name">Nessun file selezionato.</div>

      <div class="row">
        <div>
          <label for="minOsdConf">Soglia min OSD confidence</label>
          <input id="minOsdConf" type="number" min="0" step="0.1" value="0.0">
        </div>
        <div>
          <label for="ocrPsm">OCR PSM</label>
          <select id="ocrPsm">
            <option value="6" selected>6 (blocco testo)</option>
            <option value="11">11 (sparso)</option>
            <option value="3">3 (auto)</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label for="ocrMinDelta">OCR min score delta per override</label>
          <input id="ocrMinDelta" type="number" min="0" step="1" value="12">
        </div>
        <div>
          <label for="disableOcr">Validazione OCR</label>
          <select id="disableOcr">
            <option value="0" selected>Attiva</option>
            <option value="1">Disattiva</option>
          </select>
        </div>
      </div>

      <div class="actions">
        <button id="checkBtn" type="button" disabled>Analizza immagine</button>
        <button id="applySuggestedBtn" class="secondary" type="button" disabled>Applica rotazione consigliata</button>
        <button id="resetPreviewBtn" class="secondary" type="button" disabled>Ripristina preview</button>
        <button id="clearBtn" class="secondary" type="button">Pulisci</button>
      </div>
      <div id="status" class="status"></div>
      <div class="preview-wrap">
        <div class="preview-grid">
          <div>
            <div class="preview-col-title">Originale</div>
            <img id="preview" class="preview" alt="preview originale">
          </div>
          <div>
            <div class="preview-col-title">Risultato ruotato</div>
            <img id="previewRotated" class="preview" alt="preview ruotata">
            <div id="rotatedHint" class="muted">Nessuna rotazione applicata.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="result-grid">
        <div class="pill">Rotazione finale<b id="finalRotate">-</b></div>
        <div class="pill">OSD (deg/conf)<b id="osdInfo">-</b></div>
        <div class="pill">OCR best (deg/score)<b id="ocrBest">-</b></div>
      </div>
      <div class="status" id="resultMsg"></div>
    </div>

    <div class="card">
      <div class="muted" style="margin-bottom:8px;">Dettagli JSON</div>
      <pre id="jsonOut">{}</pre>
    </div>
  </div>

  <script>
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const checkBtn = document.getElementById("checkBtn");
    const applySuggestedBtn = document.getElementById("applySuggestedBtn");
    const resetPreviewBtn = document.getElementById("resetPreviewBtn");
    const clearBtn = document.getElementById("clearBtn");
    const statusEl = document.getElementById("status");
    const resultMsg = document.getElementById("resultMsg");
    const finalRotate = document.getElementById("finalRotate");
    const osdInfo = document.getElementById("osdInfo");
    const ocrBest = document.getElementById("ocrBest");
    const jsonOut = document.getElementById("jsonOut");
    const preview = document.getElementById("preview");
    const previewRotated = document.getElementById("previewRotated");
    const rotatedHint = document.getElementById("rotatedHint");

    let selectedFile = null;
    let selectedFileDataUrl = "";
    let lastResult = null;

    function normalizeDeg(value) {
      const n = Number(value || 0);
      return ((Math.round(n) % 360) + 360) % 360;
    }

    function rotateDataUrl(sourceDataUrl, rotateDeg) {
      return new Promise((resolve, reject) => {
        const deg = normalizeDeg(rotateDeg);
        if (!sourceDataUrl) {
          reject(new Error("immagine non disponibile"));
          return;
        }
        if (deg === 0) {
          resolve(sourceDataUrl);
          return;
        }
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement("canvas");
          if (deg === 90 || deg === 270) {
            canvas.width = img.height;
            canvas.height = img.width;
          } else {
            canvas.width = img.width;
            canvas.height = img.height;
          }
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            reject(new Error("canvas context non disponibile"));
            return;
          }
          ctx.translate(canvas.width / 2, canvas.height / 2);
          ctx.rotate((deg * Math.PI) / 180.0);
          ctx.drawImage(img, -img.width / 2, -img.height / 2);
          resolve(canvas.toDataURL("image/png"));
        };
        img.onerror = () => reject(new Error("errore lettura immagine"));
        img.src = sourceDataUrl;
      });
    }

    async function applySuggestedRotationPreview() {
      if (!selectedFileDataUrl || !lastResult) return;
      const deg = normalizeDeg(lastResult.final_rotate_deg);
      const dataUrl = await rotateDataUrl(selectedFileDataUrl, deg);
      previewRotated.src = dataUrl;
      previewRotated.style.display = "block";
      rotatedHint.textContent = "Rotazione applicata: " + String(deg) + "° CW";
      resetPreviewBtn.disabled = false;
    }

    function setStatus(msg, cls) {
      statusEl.textContent = msg || "";
      statusEl.className = "status " + (cls || "");
    }

    function setFile(file) {
      selectedFile = file || null;
      if (selectedFile) {
        fileName.textContent = "File: " + selectedFile.name;
        checkBtn.disabled = false;
        applySuggestedBtn.disabled = true;
        resetPreviewBtn.disabled = true;
        lastResult = null;
        const reader = new FileReader();
        reader.onload = () => {
          selectedFileDataUrl = String(reader.result || "");
          preview.src = selectedFileDataUrl;
          preview.style.display = "block";
          previewRotated.removeAttribute("src");
          previewRotated.style.display = "none";
          rotatedHint.textContent = "Nessuna rotazione applicata.";
        };
        reader.readAsDataURL(selectedFile);
      } else {
        fileName.textContent = "Nessun file selezionato.";
        checkBtn.disabled = true;
        applySuggestedBtn.disabled = true;
        resetPreviewBtn.disabled = true;
        lastResult = null;
        selectedFileDataUrl = "";
        preview.removeAttribute("src");
        preview.style.display = "none";
        previewRotated.removeAttribute("src");
        previewRotated.style.display = "none";
        rotatedHint.textContent = "Nessuna rotazione applicata.";
      }
    }

    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      const f = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
      setFile(f);
    });

    ["dragenter", "dragover"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]
        ? e.dataTransfer.files[0]
        : null;
      setFile(f);
    });

    clearBtn.addEventListener("click", () => {
      setFile(null);
      fileInput.value = "";
      setStatus("", "");
      resultMsg.textContent = "";
      resultMsg.className = "status";
      finalRotate.textContent = "-";
      osdInfo.textContent = "-";
      ocrBest.textContent = "-";
      jsonOut.textContent = "{}";
      applySuggestedBtn.disabled = true;
      resetPreviewBtn.disabled = true;
    });

    applySuggestedBtn.addEventListener("click", async () => {
      if (!selectedFile || !lastResult) return;
      applySuggestedBtn.disabled = true;
      setStatus("Applico rotazione consigliata alla preview...", "warn");
      try {
        await applySuggestedRotationPreview();
        setStatus("Preview ruotata aggiornata.", "good");
      } catch (err) {
        const msg = err && err.message ? err.message : String(err);
        setStatus("Errore preview ruotata: " + msg, "bad");
      } finally {
        applySuggestedBtn.disabled = false;
      }
    });

    resetPreviewBtn.addEventListener("click", () => {
      previewRotated.removeAttribute("src");
      previewRotated.style.display = "none";
      rotatedHint.textContent = "Nessuna rotazione applicata.";
      setStatus("Preview ruotata ripristinata.", "good");
      resetPreviewBtn.disabled = true;
    });

    checkBtn.addEventListener("click", async () => {
      if (!selectedFile) return;
      checkBtn.disabled = true;
      setStatus("Analisi in corso...", "warn");
      resultMsg.textContent = "";
      resultMsg.className = "status";
      try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("min_orientation_confidence", document.getElementById("minOsdConf").value);
        formData.append("ocr_psm", document.getElementById("ocrPsm").value);
        formData.append("ocr_validation_min_score_delta", document.getElementById("ocrMinDelta").value);
        formData.append("disable_ocr_validation", document.getElementById("disableOcr").value);

        const resp = await fetch("/api/check", { method: "POST", body: formData });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data && data.error ? data.error : "Errore API");
        }

        finalRotate.textContent = String(data.final_rotate_deg) + "° CW";
        osdInfo.textContent = String(data.osd_rotate_deg) + "° / " + Number(data.osd_confidence || 0).toFixed(3);
        ocrBest.textContent = String(data.ocr_best_rotate_deg) + "° / " + Number(data.ocr_best_score || 0).toFixed(1);
        resultMsg.textContent = data.message || "";
        resultMsg.className = "status " + (data.final_rotate_deg === 0 ? "good" : "warn");
        jsonOut.textContent = JSON.stringify(data, null, 2);
        lastResult = data;
        applySuggestedBtn.disabled = false;
        resetPreviewBtn.disabled = true;
        setStatus("Analisi completata.", "good");
      } catch (err) {
        const msg = err && err.message ? err.message : String(err);
        setStatus("Errore: " + msg, "bad");
      } finally {
        checkBtn.disabled = !selectedFile;
      }
    });
  </script>
</body>
</html>
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web UI for single image rotation checker.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checker-script",
        type=Path,
        default=Path(__file__).resolve().parent / "check_single_image_rotation.py",
    )
    parser.add_argument(
        "--checker-python",
        type=str,
        default=sys.executable,
        help="Python executable used to run checker script.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def _build_checker_cmd(
    python_exec: str,
    checker_script: Path,
    image_path: Path,
    min_orientation_confidence: float,
    ocr_psm: int,
    ocr_validation_min_score_delta: float,
    disable_ocr_validation: bool,
) -> List[str]:
    cmd: List[str] = [
        python_exec,
        str(checker_script),
        "--image",
        str(image_path),
        "--json-only",
        "--min-orientation-confidence",
        str(float(min_orientation_confidence)),
        "--ocr-psm",
        str(int(ocr_psm)),
        "--ocr-validation-min-score-delta",
        str(float(ocr_validation_min_score_delta)),
    ]
    if disable_ocr_validation:
        cmd.append("--disable-ocr-validation")
    return cmd


def create_app(checker_script: Path, checker_python: str) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_PAGE)

    @app.post("/api/check")
    def api_check():
        if "file" not in request.files:
            return jsonify({"error": "file mancante"}), 400
        up = request.files["file"]
        if up is None or up.filename is None or up.filename == "":
            return jsonify({"error": "nessun file selezionato"}), 400

        suffix = Path(up.filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(prefix="rotation_check_", suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            up.save(tmp_path)

        try:
            min_conf = float(request.form.get("min_orientation_confidence", "0.0"))
            ocr_psm = int(request.form.get("ocr_psm", "6"))
            min_delta = float(request.form.get("ocr_validation_min_score_delta", "12"))
            disable_ocr = str(request.form.get("disable_ocr_validation", "0")).strip() in {"1", "true", "True"}
        except Exception:
            tmp_path.unlink(missing_ok=True)
            return jsonify({"error": "parametri non validi"}), 400

        cmd = _build_checker_cmd(
            python_exec=checker_python,
            checker_script=checker_script,
            image_path=tmp_path,
            min_orientation_confidence=min_conf,
            ocr_psm=ocr_psm,
            ocr_validation_min_score_delta=min_delta,
            disable_ocr_validation=disable_ocr,
        )

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60.0,
                check=False,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "checker failed").strip()
                return jsonify({"error": err, "returncode": proc.returncode}), 500

            payload = (proc.stdout or "").strip()
            if not payload:
                return jsonify({"error": "checker output vuoto"}), 500
            data = json.loads(payload)
            return jsonify(data), 200
        except json.JSONDecodeError:
            return jsonify({"error": "checker output non JSON"}), 500
        except subprocess.TimeoutExpired:
            return jsonify({"error": "timeout checker"}), 504
        finally:
            tmp_path.unlink(missing_ok=True)

    return app


def main() -> int:
    args = _build_parser().parse_args()
    checker_script = args.checker_script.expanduser().resolve()
    if not checker_script.is_file():
        raise RuntimeError(f"Checker script not found: {checker_script}")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be in 1..65535.")

    app = create_app(checker_script=checker_script, checker_python=args.checker_python)
    print(
        f"Rotation web UI running on http://{args.host}:{args.port} | "
        f"checker={checker_script} | python={args.checker_python}",
        flush=True,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
