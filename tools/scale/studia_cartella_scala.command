#!/bin/zsh
# Doppio clic: apre nel browser lo studio della scala, dove scegli la cartella.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/OldSoftwareEsiBuilder/.venv-mps/bin/python"
[ -x "$PY" ] || PY="python3"
cd "$REPO" || exit 1
echo "Studio della scala: si apre nel browser, la cartella la scegli da la'."
echo "Lascia questa finestra aperta mentre lavori. Ctrl+C per fermare."
echo
exec "$PY" tools/scale/scale_study_web.py --open "$@"
