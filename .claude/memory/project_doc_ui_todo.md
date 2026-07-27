---
name: project-doc-ui-todo
description: "RÉSOLU (PR225) — compteur tests + accordéons fermés par défaut, docs + cockpit"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

TODOs identifiés le 2026-07-22, tous résolus le 2026-07-26 (PR225 + PR220).

**Why:** Découverts pendant la session de renforcement ETL (PR #178-182), hors scope de ces PRs à l'époque, reportés ici pour ne pas les perdre.

## Résolu

- **Tuile "Catalogue des tests" (Cockpit Docs)** : label hardcodé "36 tests" corrigé → "54 tests unitaires CI (11 API + 43 ETL/Data)" (PR220).
- **Accordéons fermés par défaut — docs HTML** : 45 `<details open>` répartis sur 10 fichiers (`guide_administrateur.html`, `etl_catalogue.html`, `architecture.html`, `ds_guide.html`, `execsum.html`, `mlops_lead_guide.html`, `tests_catalogue.html`, `data_dictionary.html`, `mlops_eng_guide.html`, `readme.html`) — attribut `open` retiré partout (PR225).
- **Accordéons fermés par défaut — Cockpit Gradio** : `gr.Accordion("⏸ Validation des déploiements en attente", open=True)` → `open=False` (PR225). `app_public.py` n'a aucun accordéon, rien à faire.

**How to apply si un nouveau doc HTML ou onglet Cockpit est ajouté :** vérifier qu'aucun `<details>`/`gr.Accordion` n'est ouvert par défaut — règle constatée 2 fois de suite (2026-07-22 et 2026-07-26, où de nouveaux docs ajoutés en session avaient réintroduit le problème par erreur). [[feedback_verify_before_asserting]]
