#!/usr/bin/env bash
# demo_t2.sh — Déclenche un pipeline T2 complet (nouveau code) pour démo E2E.
# Usage : ./demo_t2.sh
# Ce script :
#   1. Vérifie qu'aucune autre PR n'est ouverte
#   2. Resynce mlops sur main
#   3. Met à jour docs/demo_t2_log.txt (date/heure courante)
#   4. Commite, pousse et crée la PR GitHub
#
# Étapes restantes (manuelles) :
#   - Attendre CI vert (~2 min) puis merger la PR
#   - Cliquer GO dans le Cockpit Admin quand la gate apparaît (~1 min après merge)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_FILE="docs/demo_t2_log.txt"
TIMESTAMP=$(date "+%Y-%m-%d - %H:%M")

cd "$REPO_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        DEMO T2 — Pipeline E2E            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Vérifier PRs ouvertes ──────────────────────────────────────────────────
echo "▶ Vérification PRs ouvertes..."
OPEN_PRS=$(gh pr list --state open --json number,title --jq '.[] | "  PR #\(.number) — \(.title)"' 2>/dev/null || true)
if [ -n "$OPEN_PRS" ]; then
    echo "⚠️  PRs ouvertes détectées :"
    echo "$OPEN_PRS"
    echo ""
    read -rp "  Continuer quand même ? (y/N) : " confirm
    if [[ "${confirm,,}" != "y" ]]; then
        echo "Abandon. Merger les PRs ouvertes d'abord."
        exit 1
    fi
else
    echo "  ✓ Aucune PR ouverte"
fi

# ── 2. Resync mlops → main ────────────────────────────────────────────────────
echo ""
echo "▶ Resync mlops → main..."
git fetch origin
git reset --hard origin/main
git push origin mlops --force-with-lease
echo "  ✓ mlops à jour sur main"

# ── 3. Mise à jour fichier log demo ──────────────────────────────────────────
echo ""
echo "▶ Mise à jour ${DEMO_FILE}..."
printf 'demo T2 — %s\n' "${TIMESTAMP}" > "${DEMO_FILE}"
echo "  ✓ ${DEMO_FILE} : demo T2 — ${TIMESTAMP}"

# ── 4. Commit + push ──────────────────────────────────────────────────────────
echo ""
echo "▶ Commit et push..."
git add "$DEMO_FILE"
git commit -m "chore: demo T2 — ${TIMESTAMP}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin mlops
echo "  ✓ Poussé sur mlops"

# ── 5. Création PR ────────────────────────────────────────────────────────────
echo ""
echo "▶ Création PR GitHub..."
PR_BODY=$(cat <<EOF
## Démo E2E — Trigger 2 (nouveau code)

Pipeline T2 complet déclenché automatiquement par \`demo_t2.sh\`.

**Étapes suivantes :**
1. ✅ CI en cours (~2 min) — attendre le check vert
2. Merger cette PR → déclenche \`deploy.yml\` (CD) — build skippé, ~1 min
3. Cockpit Admin → onglet **Validation** → gate apparaît
4. Cliquer **GO** → nginx restart + test-api + Kapsule

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)

PR_URL=$(gh pr create \
    --title "chore: demo T2 — ${TIMESTAMP}" \
    --body "$PR_BODY")

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅  PR créée avec succès                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  PR     : ${PR_URL}"
echo ""
echo "  Prochaines étapes :"
echo "  1. Attendre CI vert (~2 min) → ${PR_URL}"
echo "  2. Merger la PR — deploy lance sans rebuild (~1 min)"
echo "  3. Cockpit Admin : http://100.117.99.62:7860 → GO"
echo ""
