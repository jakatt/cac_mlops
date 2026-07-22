#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# ds_session_start.sh — Routine de début de session DS (branche DS)
#
# Automatise :
#   1. Sync branche DS avec origin/main (git fetch + reset --hard)
#   2. dvc pull (données à jour)
#   3. Preprocessing cumulatif sur toutes les années disponibles (si absent en local)
#
# Usage : ./scripts/ds_session_start.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[DS]${NC} $*"; }
warn()  { echo -e "${YELLOW}[DS]${NC} $*"; }
error() { echo -e "${RED}[DS]${NC} $*" >&2; }
step()  { echo -e "${CYAN}[DS]${NC} $*"; }

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# ── venv ──────────────────────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "my_env/bin/activate" ]]; then
    warn "venv non activé — activation automatique de my_env"
    # shellcheck disable=SC1091
    source my_env/bin/activate
  else
    error "venv 'my_env' introuvable et aucun venv actif — active-le manuellement puis relance."
    exit 1
  fi
fi

# ── 1. Sync branche DS avec origin/main ───────────────────────────────────────
echo ""
step "════════════════════════════════════════════════════════════════"
step "  1/3 — Sync branche DS avec origin/main"
step "════════════════════════════════════════════════════════════════"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  error "Modifications non commitées sur des fichiers suivis — commit ou stash avant de continuer :"
  git status --short --untracked-files=no
  exit 1
fi

git fetch origin
git checkout DS
git reset --hard origin/main
info "DS synced sur $(git rev-parse --short HEAD) (origin/main)"

# ── 2. dvc pull ────────────────────────────────────────────────────────────────
echo ""
step "════════════════════════════════════════════════════════════════"
step "  2/3 — dvc pull (données à jour)"
step "════════════════════════════════════════════════════════════════"

if [[ -f ".env" ]]; then
  # Extraction ciblée (pas de `source .env` : certaines valeurs, ex. mot de
  # passe SMTP Gmail, contiennent des espaces non compatibles avec le shell)
  SCW_ACCESS_KEY_ID=$(grep -E '^SCW_ACCESS_KEY_ID=' .env | cut -d= -f2-)
  SCW_SECRET_ACCESS_KEY=$(grep -E '^SCW_SECRET_ACCESS_KEY=' .env | cut -d= -f2-)
  export SCW_ACCESS_KEY_ID SCW_SECRET_ACCESS_KEY
fi

dvc pull
info "Données DVC à jour"

# Faute de frappe ONISR connue sur 2021/2022 (carcteristiques → caracteristiques)
# — corrigée manuellement sur le VPS mais toujours présente dans les fichiers
# tels que versionnés dans DVC/S3, donc reproduite à chaque dvc pull sur une
# nouvelle machine.
shopt -s nullglob
for f in data/raw/*/carcteristiques-*.csv; do
  fixed="${f/carcteristiques/caracteristiques}"
  warn "Correction faute de frappe ONISR : $(basename "$f") → $(basename "$fixed")"
  mv "$f" "$fixed"
done
shopt -u nullglob

# ── 3. Preprocessing cumulatif (si absent en local) ───────────────────────────
echo ""
step "════════════════════════════════════════════════════════════════"
step "  3/3 — Preprocessing cumulatif (toutes les années disponibles)"
step "════════════════════════════════════════════════════════════════"

read -r MAX_YEAR PREPROCESSED_DIR <<PYEOF
$(python3 - <<'PYINNER'
from src.data.import_raw_data import discover_available_years
from src.models.train_model import _preprocessed_dir

years = discover_available_years()
if not years:
    raise SystemExit("Aucune année disponible dans data/raw/ — dvc pull a-t-il fonctionné ?")
print(years[-1], _preprocessed_dir(years))
PYINNER
)
PYEOF

if [[ -f "$PREPROCESSED_DIR/X_train.csv" ]]; then
  info "Preprocessing déjà présent : $PREPROCESSED_DIR — skip"
else
  info "Génération du preprocessing cumulatif (jusqu'à $MAX_YEAR)..."
  python -m src.data.make_dataset --year "$MAX_YEAR" --cumul
fi

echo ""
info "════════════════════════════════════════════════════════════════"
info "  Session DS prête ✓"
info "  Dataset cumulatif : $PREPROCESSED_DIR"
info "  Prochaine étape : python -m src.models.train_model --year $MAX_YEAR --cumul --algorithm lgbm ..."
info "════════════════════════════════════════════════════════════════"
