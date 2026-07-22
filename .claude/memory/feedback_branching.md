---
name: feedback-branching
description: "Règle de branching cac_mlops — jamais commiter sur main directement, toujours sur mlops (ex-jacques) ou DS (ex-noel)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8980ac59-2dd0-47a0-ab4c-af544d359117
---

Tout changement de code — un seul caractère — doit passer par `mlops` ou `DS` → PR → `main`. Jamais de raccourci.

Branches : `mlops` (Jacques) et `DS` (Noël). Renommées le 2026-06-30 depuis `jacques`/`noel`.

**Why:** `main` est en branch protection — PR obligatoire + CI requise. Un commit direct ou un deploy SCP bypass la CI, la review, et l'audit trail. Règle rappelée explicitement le 2026-07-03 après un deploy SCP direct de `app.py` (session Orchestration/disclaimer) qui a contourné la chaîne CI/CD qu'on venait de mettre en place.

**How to apply:**
1. **En début de toute session** : `git fetch origin && git reset --hard origin/main` sur `mlops` — TOUJOURS, sans exception.
2. Avant tout `git commit`, vérifier avec `git branch` qu'on est sur `mlops` ou `DS`.
3. Quand le travail est prêt : PR vers `main`. Ne jamais `git push origin main` ni `scp ... && docker restart`.
4. Après chaque squash & merge d'une PR, refaire le step 1 immédiatement avant d'entamer le travail suivant.
5. **SCP direct = interdit**, même pour un "petit fix UI". Le fast lane CI/CD (restart gradio en ~15s) remplace ce besoin.

**Pourquoi récurrent :** Les squash-merges créent une divergence d'historique entre `mlops` et `main`. Sans reset, les vieux commits réapparaissent comme conflits à la PR suivante (problème sur PR #64, #65, #69 — 2026-07-01/03). Le reset est la seule protection fiable.

**Piège découvert le 2026-07-22 : `origin/mlops`/`origin/DS` peut contenir des commits jamais mergés dans `main`** (ex. un commit mémoire fait directement sur la branche sans passer par PR, une session précédente qui n'a pas fini son travail). Après `git reset --hard origin/main` + tentative de `git push origin mlops`, ça se traduit par un rejet "non-fast-forward" — **ne pas paniquer, ne pas merger l'ancien contenu** : vérifier d'abord ce que contient `origin/mlops` (`git log --oneline mlops..origin/mlops`), et si son contenu est confirmé obsolète/déjà traité ailleurs, `git push origin mlops --force-with-lease` (jamais `--force` sans lease) écrase proprement — c'est exactement le cas que `--force-with-lease` est censé couvrir, pas une opération exceptionnelle à éviter.
