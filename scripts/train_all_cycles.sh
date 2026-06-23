#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# train_all_cycles.sh — Lance tous les cycles de training en séquence
#
# Détecte automatiquement les années disponibles depuis les tags DVC :
#   data-v1 → 2021 seul      (cycle 1 : cumul=false)
#   data-v2 → +2022           (cycle 2 : cumul=true)
#   data-v3 → +2023           (cycle 3 : cumul=true)
#   data-v4 → +2024           (cycle 4 : cumul=true)  ← automatique si tag présent
#   ...
#
# Usage :
#   ./scripts/train_all_cycles.sh                    # lgbm, tous les cycles
#   ./scripts/train_all_cycles.sh --algorithm rf     # random forest
#   ./scripts/train_all_cycles.sh --dry-run          # affiche sans déclencher
#   ./scripts/train_all_cycles.sh --from-cycle 2     # reprend à partir du cycle 2
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
ALGORITHM="lgbm"
DRY_RUN=false
FROM_CYCLE=1

for arg in "$@"; do
  case "$arg" in
    --algorithm) shift; ALGORITHM="$1" ;;
    --dry-run)   DRY_RUN=true ;;
    --from-cycle) shift; FROM_CYCLE="$1" ;;
  esac
done
# Parsing avec valeurs suivantes
while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithm)  ALGORITHM="$2";  shift 2 ;;
    --dry-run)    DRY_RUN=true;    shift ;;
    --from-cycle) FROM_CYCLE="$2"; shift 2 ;;
    *)            shift ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[TRAIN]${NC} $*"; }
warn()  { echo -e "${YELLOW}[TRAIN]${NC} $*"; }
error() { echo -e "${RED}[TRAIN]${NC} $*" >&2; }
step()  { echo -e "${CYAN}[TRAIN]${NC} $*"; }

# ── Détecter les années disponibles depuis les tags DVC ───────────────────────
BASE_YEAR=2021

TAGS=$(git tag -l "data-v*" | sort -V)
N_VERSIONS=$(echo "$TAGS" | grep -c "data-v" || echo 0)

if [ "$N_VERSIONS" -eq 0 ]; then
  error "Aucun tag DVC 'data-v*' trouvé dans git."
  error "Vérifiez : git tag -l 'data-v*'"
  exit 1
fi

# Construire la liste des cycles : (version, année, cumul)
CYCLES=()
for i in $(seq 1 "$N_VERSIONS"); do
  YEAR=$(( BASE_YEAR + i - 1 ))
  if [ "$i" -eq 1 ]; then
    CUMUL="false"
  else
    CUMUL="true"
  fi
  CYCLES+=("$i:$YEAR:$CUMUL")
done

# ── Résumé ────────────────────────────────────────────────────────────────────
echo ""
info "════════════════════════════════════════════════════════════════"
info "  TRAINING ALL CYCLES — $(date '+%Y-%m-%d %H:%M:%S')"
info "  Tags DVC détectés : $N_VERSIONS ($TAGS)"
info "  Algorithme        : $ALGORITHM"
$DRY_RUN && warn "  MODE DRY-RUN — aucun workflow ne sera déclenché"
[ "$FROM_CYCLE" -gt 1 ] && warn "  Reprise à partir du cycle $FROM_CYCLE"
info "════════════════════════════════════════════════════════════════"
echo ""

for ENTRY in "${CYCLES[@]}"; do
  CYCLE_NUM=$(echo "$ENTRY" | cut -d: -f1)
  YEAR=$(echo "$ENTRY"      | cut -d: -f2)
  CUMUL=$(echo "$ENTRY"     | cut -d: -f3)

  if [ "$CYCLE_NUM" -lt "$FROM_CYCLE" ]; then
    warn "  Cycle $CYCLE_NUM (année $YEAR) — sauté (--from-cycle $FROM_CYCLE)"
    continue
  fi

  DATA_TAG="data-v${CYCLE_NUM}"
  echo ""
  step "──────────────────────────────────────────────────────────────"
  step "  Cycle $CYCLE_NUM / $N_VERSIONS"
  step "  Année   : $YEAR"
  step "  Cumul   : $CUMUL  (données depuis 2021 jusqu'à $YEAR)"
  step "  DVC tag : $DATA_TAG"
  step "  Algo    : $ALGORITHM"
  step "──────────────────────────────────────────────────────────────"

  if $DRY_RUN; then
    warn "  [DRY-RUN] gh workflow run train.yml -f year=$YEAR -f cumul=$CUMUL -f algorithm=$ALGORITHM -f promote=true"
    continue
  fi

  # ── Déclencher le workflow ─────────────────────────────────────────────────
  info "  Déclenchement du workflow..."
  gh workflow run train.yml \
    -f year="$YEAR" \
    -f cumul="$CUMUL" \
    -f algorithm="$ALGORITHM" \
    -f promote=true

  # ── Récupérer le run ID (le workflow met quelques secondes à apparaître) ───
  info "  Attente démarrage du run (15s)..."
  sleep 15

  RUN_ID=""
  for attempt in 1 2 3 4 5; do
    RUN_ID=$(gh run list --workflow=train.yml --limit 1 --json databaseId,status \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['databaseId']) if d else print('')" 2>/dev/null || echo "")
    if [ -n "$RUN_ID" ]; then
      break
    fi
    warn "  Attente run ID (tentative $attempt/5)..."
    sleep 10
  done

  if [ -z "$RUN_ID" ]; then
    error "  Impossible de récupérer le run ID — vérifiez GitHub Actions manuellement."
    error "  URL : https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/actions"
    exit 1
  fi

  info "  Run ID : $RUN_ID — attente de la fin (max 45 min)..."
  info "  Suivre : gh run watch $RUN_ID"

  # ── Attendre la fin ────────────────────────────────────────────────────────
  if gh run watch "$RUN_ID" --exit-status; then
    info "  ✓ Cycle $CYCLE_NUM terminé avec succès (année $YEAR, $DATA_TAG)"
  else
    EXIT_CODE=$?
    error "  ✗ Cycle $CYCLE_NUM ÉCHOUÉ (année $YEAR)"
    error "  Logs : gh run view $RUN_ID --log-failed"
    echo ""
    error "  Cycles suivants non lancés. Pour reprendre :"
    NEXT_CYCLE=$(( CYCLE_NUM + 1 ))
    error "    ./scripts/train_all_cycles.sh --from-cycle $NEXT_CYCLE --algorithm $ALGORITHM"
    exit $EXIT_CODE
  fi
done

# ── Rapport final ─────────────────────────────────────────────────────────────
echo ""
info "════════════════════════════════════════════════════════════════"
info "  TOUS LES CYCLES TERMINÉS ✓ — $(date '+%Y-%m-%d %H:%M:%S')"
info ""
info "  $N_VERSIONS cycles exécutés :"
for ENTRY in "${CYCLES[@]}"; do
  CYCLE_NUM=$(echo "$ENTRY" | cut -d: -f1)
  YEAR=$(echo "$ENTRY"      | cut -d: -f2)
  info "    Cycle $CYCLE_NUM → année $YEAR (data-v${CYCLE_NUM}, algo=$ALGORITHM, @Production)"
done
info ""
info "  Vérifications :"
info "    MLflow    → http://51.159.187.132:5001  (3 versions, @Production = v$N_VERSIONS)"
info "    Gradio    → http://51.159.187.132:7860  (onglet Drift : $((N_VERSIONS - 1)) rapport(s))"
info "    Prefect   → http://51.159.187.132:4200  ($N_VERSIONS runs enregistrés)"
info "════════════════════════════════════════════════════════════════"
