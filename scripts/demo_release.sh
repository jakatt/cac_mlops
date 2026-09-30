#!/usr/bin/env bash
# Démo live Trigger 2 — ouvre une PR d'une ligne, la plus rapide possible à déployer.
#
# Change uniquement docs/release.json (date et heure du lancement) sur une
# branche neuve créée depuis main, directement via l'API GitHub : rien n'est
# touché dans le dépôt local ni sur la branche mlops.
#
# Pourquoi ce fichier : un fichier de docs/ ne reconstruit aucune image et ne
# redémarre aucun service (CD ~1 min 30 jusqu'à la gate), et il n'est mis en
# ligne qu'au GO. La nouvelle date apparaît alors en bas de la page d'accueil
# des 3 Cockpits (services/gradio/_release.py).
#
# Déroulé : ce script → CI (~1 min 15) → merge sur GitHub → CD → gate dans le
# Cockpit → GO → date visible.
#
# Usage : scripts/demo_release.sh        (prérequis : gh authentifié)
set -euo pipefail

REPO="jakatt/cac_mlops"
FILE="docs/release.json"
NOW_ISO=$(TZ=Europe/Paris date +%Y-%m-%dT%H:%M:%S%z)
NOW_TXT=$(TZ=Europe/Paris date '+%d/%m/%Y à %H:%M')
BRANCH="demo/release-$(TZ=Europe/Paris date +%Y%m%d-%H%M%S)"

echo "==> Branche $BRANCH depuis main"
MAIN_SHA=$(gh api "repos/$REPO/git/ref/heads/main" -q .object.sha)
gh api -X POST "repos/$REPO/git/refs" -f ref="refs/heads/$BRANCH" -f sha="$MAIN_SHA" >/dev/null

echo "==> $FILE : mise à jour du $NOW_TXT"
FILE_SHA=$(gh api "repos/$REPO/contents/$FILE?ref=main" -q .sha)
CONTENT=$(printf '{"published_at": "%s", "display": "%s"}\n' "$NOW_ISO" "$NOW_TXT" | base64 | tr -d '\n')
gh api -X PUT "repos/$REPO/contents/$FILE" \
  -f message="demo: mise à jour du $NOW_TXT" \
  -f content="$CONTENT" -f branch="$BRANCH" -f sha="$FILE_SHA" >/dev/null

echo "==> Ouverture de la PR"
PR_URL=$(gh pr create --repo "$REPO" --base main --head "$BRANCH" \
  --title "Démo — mise à jour du $NOW_TXT" \
  --body "Démo live Trigger 2 : une seule ligne changée (\`$FILE\`).

Aucune image reconstruite, aucun service redémarré. La nouvelle date n'apparaît en bas de la page d'accueil des Cockpits qu'après le **GO** dans le Cockpit.")
echo "$PR_URL"
open "$PR_URL" 2>/dev/null || true
