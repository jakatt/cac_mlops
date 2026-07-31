---
name: feedback-new-machine-startup
description: "Procédure à suivre quand une session démarre sur une machine où il n'y a aucun historique de conversation (nouveau Mac, nouvelle install Claude Code)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Le repo `cac_mlops` vit dans `~/Documents/IA & Python/cac_mlops/cac_mlops`,
synchronisé iCloud (cf. [[feedback_icloud_sync]]) — présent sur toutes les
machines du user sans clone git manuel. La mémoire globale Claude
(`~/.claude/projects/.../memory/`) est en revanche propre à chaque machine
et absente au premier lancement sur une nouvelle install.

**Séquence à suivre, dans l'ordre :**

1. `cd` dans le repo, puis `git status`, `git branch --show-current`,
   `git log -5 --oneline`. Si `git status` se comporte bizarrement ou est
   lent, penser aux fichiers "dataless" iCloud (placeholders non
   téléchargés) — cf. [[feedback_icloud_sync]].
2. Si des changements semblent manquants ou la branche `DS` désynchronisée :
   proposer `./scripts/ds_session_start.sh` (sync DS sur `origin/main` +
   `dvc pull`) plutôt que des commandes git manuelles — c'est la routine
   déjà établie pour ça. Le script refuse de tourner s'il y a des
   modifications locales non commitées (vérifier avant de le lancer).
3. Lire `.claude/memory/MEMORY.md` (repo-local) en premier — c'est la
   source de vérité à jour du projet, à privilégier sur toute mémoire
   globale potentiellement absente ou périmée sur cette machine.
4. Ne jamais supposer que l'accès SSH VPS fonctionne sans le vérifier :
   `ssh deploy@51.159.187.132 "echo ok"`. Un échec signifie très
   probablement que les clés SSH ne sont pas encore configurées sur cette
   machine — action manuelle du user, hors de portée de Claude.
5. Si `~/.claude/projects/.../memory/` est absent ou clairsemé sur cette
   machine : normal sur une nouvelle install, ne pas s'en inquiéter — tout
   ce qui compte pour le projet est déjà dans `.claude/memory/` du repo.

**How to apply** : si le user colle un message du type "on reprend, voici
la procédure de démarrage" en tout début de conversation sans autre
contexte, dérouler cette séquence avant de répondre à quoi que ce soit
d'autre.
