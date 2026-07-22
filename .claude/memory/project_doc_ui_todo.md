---
name: project-doc-ui-todo
description: "TODOs UI docs/cockpit identifiés le 2026-07-22 — compteur tests + accordéons fermés par défaut"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

TODOs identifiés en session, à traiter dans une prochaine PR docs/UI.

**Why:** Découverts pendant la session de renforcement ETL (PR #178-182) — pas dans le scope de ces PRs, reportés ici pour ne pas les perdre.

**How to apply:** Une seule PR pour les 3 points (docs + cockpit, pas de changement de logique métier).

## TODOs identifiés (2026-07-22)

- [ ] **Tuile "Catalogue des tests" (cockpit Docs)** : `services/gradio/app.py` ligne ~1659, description de la tuile hardcodée à "36 tests unitaires CI · pipeline CD · 6 tests Prefect post-deploy" → mettre à jour vers le compte réel (54 au 2026-07-22, cf. `docs/tests_catalogue.html`). Attention : ce compte va continuer à évoluer, envisager d'automatiser sa génération plutôt que de le garder en dur.
- [ ] **Accordéons fermés par défaut — docs HTML** : tous les fichiers `docs/*.html` doivent s'ouvrir avec **tous les `<details>` fermés** par défaut (retirer l'attribut `open` partout). État constaté 2026-07-22 (nombre de `<details ... open>` par fichier) : `guide_administrateur.html` (11) · `architecture.html` (7) · `ds_guide.html` (6) · `execsum.html` (6) · `mlops_lead_guide.html` (6) · `tests_catalogue.html` (6) · `data_dictionary.html` (4) · `mlops_eng_guide.html` (4) · `readme.html` (3) · `ci_cd_pipeline_runbook.html` / `hyperparams_guide.html` / `resilience_mechanisms.html` (déjà 0, rien à faire).
- [ ] **Accordéons fermés par défaut — cockpit Gradio (tous onglets)** : même règle pour `services/gradio/app.py` — actuellement un seul `gr.Accordion(..., open=True)` détecté : "⏸ Validation des déploiements en attente" (ligne ~2105). Le passer à `open=False` pour cohérence avec le reste des onglets (déjà `open=False` ailleurs). Vérifier aussi `app_public.py` au cas où.
