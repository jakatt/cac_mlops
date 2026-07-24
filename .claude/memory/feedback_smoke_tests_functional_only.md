---
name: feedback-smoke-tests-functional-only
description: "les tests post-déploiement (test_api_flow) doivent vérifier que ça répond, jamais juger le sens/la qualité des prédictions du modèle"
metadata:
  node_type: memory
  type: feedback
---

Un smoke test automatique post-déploiement (ex. `test_whatif_speed` dans `src/flows/test_api_flow.py`) doit uniquement s'assurer que la fonctionnalité répond correctement — comme si un utilisateur l'utilisait (HTTP 200, JSON valide, endpoint accessible) — jamais juger si le résultat métier/statistique est "cohérent" ou "correct".

**Why:** Incident réel (2026-07-23, PR #202) : un blueprint `rf` statistiquement meilleur que `@Production` (f1 +0.0198, recall +0.12, toutes les métriques KPI passées) a été rollback automatiquement parce que `test_whatif_speed` assertait `proba(vma=90) > proba(vma=50)` sur UN scénario synthétique fixe — un modèle différent peut légitimement donner un résultat différent sur un cas particulier sans être un mauvais modèle. Juger la "cohérence métier" d'une prédiction individuelle n'a pas sa place dans un test de smoke fonctionnel — c'est un problème de qualité de modèle (à évaluer via les métriques agrégées sur le vrai jeu de test), pas de disponibilité de service.

**How to apply:** Quand on écrit ou révise un test dans `test_api_flow.py` (ou tout futur smoke test post-déploiement), se limiter à : le endpoint répond-il ? la réponse est-elle bien formée ? Ne jamais ajouter d'assertion sur la valeur/le sens d'une prédiction individuelle — si un doute existe sur la qualité du modèle, il doit se traiter en amont (gate KPI sur métriques agrégées, comparaison à `@Production`), pas via un test métier ad-hoc après coup qui peut bloquer à tort un modèle par ailleurs meilleur.
