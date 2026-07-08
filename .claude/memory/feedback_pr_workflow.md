---
name: feedback-pr-workflow
description: "Règle PR — une PR à la fois, jamais ajouter de commits sur une PR ouverte, attendre le merge avant la suivante"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

Une PR = un changement. Ne JAMAIS ajouter des commits supplémentaires sur une PR déjà ouverte et en cours de déploiement.

**Why:** Corrigé explicitement par le user le 2026-07-03 : "tu te mélanges systématiquement sur les PRs. Tu en crées une, je la lance, elle se déploie et tu plugs le changement suivant sur la même PR." — le user lance le deploy CI/CD dès la PR créée, ajouter des commits dessus après coup perturbe le deploy en cours.

**How to apply:**
1. AVANT de commiter/pusher : `git fetch origin && git rebase origin/main` pour éviter les conflits.
2. Créer la PR, la pousser → STOP. Attendre que le user dise "la PR X est passée" (= mergée et déployée).
3. Après merge : `git fetch origin && git reset --hard origin/main && git push origin mlops --force-with-lease` AVANT tout nouveau travail.
4. Seulement ensuite : commencer la PR suivante sur `mlops` propre.
5. Si plusieurs changements sont prévus → séquencer : PR1 → merge → resync → PR2 → merge → resync → ...
6. Jamais `git push` sur une branche dont la PR est déjà ouverte sauf correction urgente demandée explicitement par le user.

**Pattern fiabilité anti-conflit :** Toujours `git fetch origin && git rebase origin/main` immédiatement avant le premier `git add` d'une nouvelle PR.

Voir aussi [[feedback-branching]] pour les règles de resync après squash merge.
