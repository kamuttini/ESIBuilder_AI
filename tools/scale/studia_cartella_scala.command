#!/bin/zsh
# Doppio clic: chiede la cartella con una finestra, studia la scala, apre il risultato.
# Nessun percorso da scrivere. Serve solo che il volume con le immagini sia montato.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/OldSoftwareEsiBuilder/.venv-mps/bin/python"
[ -x "$PY" ] || PY="python3"

cd "$REPO" || exit 1
echo "Studio della scala — scegli la cartella nella finestra che si apre."
echo

# --pick apre la finestra di selezione, --open apre la pagina a fine lavoro.
# Lo stadio depth costa qualche minuto: se lo vuoi saltare, aggiungi --no-depth qui sotto.
"$PY" tools/scale/study_scale_folder.py --pick --open --python-bin "$PY" "$@"
STATUS=$?

echo
if [ $STATUS -eq 0 ]; then
  echo "Fatto. La pagina si e' aperta nel browser."
else
  echo "Interrotto (codice $STATUS)."
fi
echo "Premi INVIO per chiudere questa finestra."
read -r _
