#!/bin/bash
# Protection de branche main — à exécuter UNE SEULE FOIS par le mainteneur du repo.
# Prérequis : gh CLI installé et authentifié (gh auth login)
#
# Ce script configure les règles suivantes sur la branche main :
#   - Le workflow CI doit passer avant tout merge
#   - Force push interdit
#   - Suppression de la branche interdite
#   - Au moins 1 review requise sur les PR

set -e

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "==> Configuration branch protection sur $REPO/main"

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/$REPO/branches/main/protection" \
  --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "==> Branch protection activée sur main ✓"
echo ""
echo "Règles appliquées :"
echo "  - CI (job 'test') obligatoire avant merge"
echo "  - 1 review requise sur les PR"
echo "  - Force push interdit"
echo "  - Suppression de branche interdite"
