#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# raz_mlops.sh — Remise à zéro complète de la stack MLOps (VPS Scaleway)
#
# Usage:
#   ./scripts/raz_mlops.sh              # Exécute avec confirmation interactive
#   ./scripts/raz_mlops.sh --dry-run    # Affiche les actions sans les exécuter
#   ./scripts/raz_mlops.sh --yes        # Exécute sans confirmation
#
# Ce qui est réinitialisé :
#   - MLflow (runs, modèles, artifacts) → PostgreSQL + MinIO vidés
#   - Prefect (flows, deployments, runs) → volume prefect_data vidé
#   - Prometheus (métriques)             → volume prometheus_data vidé
#   - Grafana (user settings)            → volume grafana_data vidé
#   - data/raw + data/preprocessed       → supprimés (re-pull DVC au training)
#   - src/models/*.joblib                → supprimé (régénéré au training)
#   - reports/drift/                     → supprimé (régénéré au training)
#   - S3 Scaleway k8s-model/             → supprimé (repopulé après 3ème cycle)
#   - S3 Scaleway k8s-gradio-data/       → supprimé (repopulé après 3ème cycle)
#   - S3 Scaleway mlflow-k8s/            → supprimé (artefacts K8s obsolètes)
#
# Ce qui est CONSERVÉ :
#   - S3 DVC remote (s3://cac-mlops-data/dvc/) → données brutes ONISR intactes
#   - DVC cache local (/data/dvc_cache/)        → accélère les futurs dvc pull
#   - Code source + secrets GitHub              → inchangés
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
DEPLOY_DIR="${DEPLOY_DIR:-/data/cac_mlops}"
VOLUMES_PATH="${DOCKER_VOLUMES_PATH:-/data}"
SCW_BUCKET="cac-mlops-data"
SCW_ENDPOINT="https://s3.fr-par.scw.cloud"
SCW_REGION="fr-par"

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
AUTO_YES=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes)     AUTO_YES=true ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[RAZ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[RAZ]${NC} $*"; }
error() { echo -e "${RED}[RAZ]${NC} $*" >&2; }

run() {
  if $DRY_RUN; then
    echo -e "${YELLOW}[DRY-RUN]${NC} $*"
  else
    eval "$@"
  fi
}

# ── Confirmation interactive ──────────────────────────────────────────────────
if ! $DRY_RUN && ! $AUTO_YES; then
  echo ""
  warn "════════════════════════════════════════════════════════════════"
  warn "  RAZ COMPLÈTE — Tout l'historique MLflow, Prefect,"
  warn "  Prometheus, Grafana et les données locales seront effacés."
  warn "  Les données brutes ONISR sur S3/DVC sont CONSERVÉES."
  warn "════════════════════════════════════════════════════════════════"
  echo ""
  read -r -p "Confirmer la remise à zéro ? Tapez 'yes' pour continuer : " CONFIRM
  if [[ "$CONFIRM" != "yes" ]]; then
    echo "Annulé."
    exit 0
  fi
fi

cd "$DEPLOY_DIR"

# Charger .env pour les credentials
if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo ""
info "════════════════════════════════════════════════════════════════"
info "  DÉMARRAGE RAZ — $(date '+%Y-%m-%d %H:%M:%S')"
info "════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# Phase A — Arrêt de la stack + suppression volumes Docker nommés
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase A — Arrêt de la stack Docker (+ suppression volumes nommés)..."
run "docker compose down -v --remove-orphans"
# -v supprime les named volumes : prometheus_data, grafana_data, prefect_data

# ─────────────────────────────────────────────────────────────────────────────
# Phase B — Suppression bind mounts PostgreSQL + MinIO
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase B — Suppression bind mounts PostgreSQL + MinIO..."
run "docker run --rm -v ${VOLUMES_PATH}:/mnt alpine rm -rf /mnt/postgres_data /mnt/minio_data"
info "  ✓ postgres_data + minio_data supprimés (via container — contourne les droits root)"

# ─────────────────────────────────────────────────────────────────────────────
# Phase C — Nettoyage données locales
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase C — Nettoyage données locales..."
# data/raw/ conservé localement (DVC remote S3 intact, dvc pull rapide en démo)
run "docker run --rm -v ${DEPLOY_DIR}/data:/mnt alpine rm -rf /mnt/preprocessed"
info "  ✓ data/preprocessed/ supprimé (via container)"
run "docker run --rm -v ${DEPLOY_DIR}/src/models:/mnt alpine rm -f /mnt/trained_model.joblib"
info "  ✓ src/models/trained_model.joblib supprimé"
run "docker run --rm -v ${DEPLOY_DIR}/reports:/mnt alpine sh -c 'rm -rf /mnt/drift && mkdir -p /mnt/drift'"
info "  ✓ reports/drift/ vidé"

# ─────────────────────────────────────────────────────────────────────────────
# Phase D — Nettoyage S3 Scaleway (préfixes K8s uniquement — DVC conservé)
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase D — Nettoyage S3 Scaleway (préfixes k8s)..."

SCW_AK="${SCW_ACCESS_KEY:-}"
SCW_SK="${SCW_SECRET_KEY:-}"

# Fallback : lire depuis .dvc/config.local (déjà présent sur le VPS, gitignored)
if [ -z "$SCW_AK" ] && [ -f "${DEPLOY_DIR}/.dvc/config.local" ]; then
  SCW_AK=$(grep 'access_key_id'     "${DEPLOY_DIR}/.dvc/config.local" | awk '{print $3}')
  SCW_SK=$(grep 'secret_access_key' "${DEPLOY_DIR}/.dvc/config.local" | awk '{print $3}')
  [ -n "$SCW_AK" ] && info "  Credentials SCW lus depuis .dvc/config.local"
fi

if [ -n "$SCW_AK" ] && [ -n "$SCW_SK" ]; then
  for PREFIX in k8s-model k8s-gradio-data mlflow-k8s; do
    info "  Suppression s3://${SCW_BUCKET}/${PREFIX}/..."
    run "docker run --rm \
      -e AWS_ACCESS_KEY_ID=${SCW_AK} \
      -e AWS_SECRET_ACCESS_KEY=${SCW_SK} \
      -e AWS_DEFAULT_REGION=${SCW_REGION} \
      amazon/aws-cli s3 rm s3://${SCW_BUCKET}/${PREFIX}/ \
        --recursive --endpoint-url ${SCW_ENDPOINT} 2>/dev/null || true"
  done
  info "  ✓ Préfixes K8s nettoyés (DVC s3://${SCW_BUCKET}/dvc/ conservé)"
else
  warn "  SCW_ACCESS_KEY absent dans .env — nettoyage S3 ignoré"
  warn "  Relancer avec : SCW_ACCESS_KEY=... SCW_SECRET_KEY=... ./scripts/raz_mlops.sh --yes"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Dry-run : sortie ici
# ─────────────────────────────────────────────────────────────────────────────
if $DRY_RUN; then
  echo ""
  info "Dry-run terminé — aucune modification effectuée."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase E — Redémarrage de la stack
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase E — Redémarrage de la stack..."
docker compose up -d
info "  ✓ Stack démarrée"

# ─────────────────────────────────────────────────────────────────────────────
# Phase F — Healthchecks (max 5 min par service)
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase F — Healthchecks..."

wait_url() {
  local NAME="$1"; local URL="$2"; local MAX="${3:-30}"
  for i in $(seq 1 "$MAX"); do
    if curl -sf "$URL" > /dev/null 2>&1; then
      info "  ✓ ${NAME} ready (${i}×10s)"
      return 0
    fi
    echo "    [${i}×10s] ${NAME} en attente..."
    sleep 10
  done
  error "  ✗ ${NAME} timeout — vérifier les logs : docker compose logs ${NAME}"
  return 1
}

wait_url "MLflow"  "http://localhost:5001/health"
wait_url "API"     "http://localhost:8080/health"
wait_url "Prefect" "http://localhost:4200/api/health"

# ─────────────────────────────────────────────────────────────────────────────
# Phase G — Re-enregistrement deployments Prefect
# ─────────────────────────────────────────────────────────────────────────────
info ""
info "Phase G — Re-enregistrement deployments Prefect..."
# Attendre que le worker ait créé le work-pool (fait au démarrage du container)
info "  Attente worker Prefect (45s)..."
sleep 45

PREFECT_OK=false
for attempt in 1 2 3; do
  info "  Tentative ${attempt}/3 — prefect deploy --all..."
  if printf 'n\nn\nn\nn\n' | docker compose exec -T prefect-worker \
      sh -c "cd /app && prefect deploy --all" 2>&1; then
    info "  ✓ Deployments Prefect enregistrés (etl, train, retrain-annual, drift-check)"
    PREFECT_OK=true
    break
  fi
  if [ "$attempt" -lt 3 ]; then
    warn "  ⚠ Échec tentative ${attempt} — nouvel essai dans 20s..."
    sleep 20
  fi
done

if ! $PREFECT_OK; then
  warn "  ⚠ prefect deploy --all a échoué après 3 tentatives"
  warn "  Relancer manuellement :"
  warn "    docker compose exec prefect-worker sh -c 'cd /app && prefect deploy --all'"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Fin
# ─────────────────────────────────────────────────────────────────────────────
echo ""
info "════════════════════════════════════════════════════════════════"
info "  RAZ TERMINÉE ✓  — $(date '+%Y-%m-%d %H:%M:%S')"
info ""
info "  Vérifications manuelles recommandées :"
info "    MLflow  → http://$(hostname -I | awk '{print $1}'):5001"
info "    Prefect → http://$(hostname -I | awk '{print $1}'):4200"
info "    Grafana → http://$(hostname -I | awk '{print $1}'):3000"
info ""
info "  Prochaines étapes — GitHub Actions → Train :"
info "    1. year=2021  cumul=false  algorithm=lgbm  promote=true"
info "    2. year=2022  cumul=true   algorithm=lgbm  promote=true"
info "    3. year=2023  cumul=true   algorithm=lgbm  promote=true"
info "════════════════════════════════════════════════════════════════"
