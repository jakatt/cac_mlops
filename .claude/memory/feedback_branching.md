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


**Piège découvert le 2026-08-25 : ne JAMAIS `git reset --hard origin/main` sans vérifier `git status` d'abord.** Après le merge de la PR #257, resync lancé directement sans checker s'il y avait du travail en cours — 3 fichiers avaient des modifications non commitées (implémentation de G, faite pendant l'attente du merge précédent) et ont été silencieusement effacés par le reset. Récupérable seulement parce que le contenu perdu avait été écrit quelques minutes plus tôt et pouvait être reconstitué de mémoire — dans le cas général, un reset --hard sur du travail non commité est une perte de données irréversible. Le fichier NON suivi par git (`model_diff.py`, jamais ajouté) a survécu sans problème — seuls les fichiers déjà trackés mais modifiés sont à risque.

**How to apply (ajout) :** avant tout `git reset --hard`, TOUJOURS `git status --short` en premier. Si des modifications non commitées existent et représentent du travail à garder, soit les committer d'abord (si sur la bonne branche), soit les stash (`git stash`) avant le reset, jamais reset à l'aveugle même dans la routine "resync après merge" déjà establie ci-dessus — cette routine suppose implicitement un état propre, ce qui n'est pas garanti si du travail a été fait en parallèle d'un merge en attente.

**Piège n°2 découvert le 2026-08-25, quelques minutes après le premier : committer sur `mlops` sans d'abord resynchroniser sur `origin/main` après un squash-merge récent recrée la même divergence que ce fichier documente déjà plus haut.** Après avoir corrigé le piège n°1 (stash avant reset), un commit mémoire a été fait AVANT le reset --hard plutôt qu'après — ce commit s'est donc retrouvé sur l'ancien historique pré-squash-merge (PR #258), diverge de `origin/main`. Corrigé en re-stashant tout travail en cours, reset --hard proprement, puis stash pop — le commit mémoire orphelin a simplement été recréé après coup (contenu trivial à reconstituer). **How to apply :** l'ORDRE correct après un merge est TOUJOURS (1) vérifier `git status`, (2) stash si du travail non commité existe, (3) `git reset --hard origin/main`, (4) stash pop, (5) SEULEMENT ENSUITE commiter quoi que ce soit (code ou mémoire) — ne jamais commiter entre un merge et le resync, même pour un commit mémoire "anodin".