---
name: project-test-layers-model
description: "Modèle à 3 couches de validation post-déploiement (technique/fonctionnel interne/fonctionnel externe) — structurant pour toute discussion sur les tests, rollback ou Healthcheck"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Établi le 2026-07-29 en rationalisant les mécanismes de validation post-déploiement
(VPS + K8s). Sert de cadre de référence pour toute future discussion sur les
tests, le rollback, ou le Healthcheck — évite de refaire cette analyse depuis zéro.

## Les 3 couches, du plus superficiel au plus complet

1. **Smoke test technique** (`smoke_test_task`, `src/flows/deploy_vps_flow.py:214`)
   — ping `/health` interne (Docker), retry patient 90s (tolère un cold-boot
   juste après `compose up`). Répond à : *"le conteneur tourne-t-il ?"*
   Aucune logique métier testée. VPS uniquement (pré-gate + post-compose).

2. **Fonctionnel interne** (`test-api`/`test-gradio-public`, état actuel avant
   TODO ci-dessous) — JWT, `/predict` réel, what-if (vma=90 vs 50), rate-limit
   429. Répond à : *"la logique métier fonctionne-t-elle ?"* Mais exécuté via
   `NGINX_URL`/`K8S_NGINX_URL` (adresses internes Docker/ClusterIP) — **ne
   prouve pas que l'utilisateur réel peut y accéder** (contourne Caddy/DNS/HTTPS
   public).

3. **Fonctionnel externe** (à ajouter — TODO priorité 1, cf. `.claude/memory/`
   session courante) — mêmes flows que la couche 2, mais pointés vers
   `https://mlops.jakat-inc.fr` / `https://kapsule.jakat-inc.fr`. **C'est la
   seule couche qui garantit réellement "l'utilisateur peut s'en servir"**
   (logique métier + vrai chemin réunis).

**Le Healthcheck du Cockpit** (`_check_url`, `services/gradio/app.py:1769`) est
un GET superficiel comme la couche 1, mais ses lignes "Accès" (Gradio Public
VPS/K8s, FastAPI VPS/K8s) testent déjà le **chemin public** (contrairement à
`smoke_test_task` qui reste 100% interne) — donc pas un doublon strict, plutôt
une variante "couche 1 sur chemin externe".

## Décisions prises en découlant

- **TODO #1 (à faire)** : ajouter la couche 3 (externe) à `test-api`/
  `test-gradio-public`, en gardant la couche 2 (interne) — pas de remplacement,
  les deux tournent séquentiellement dans le même `try/except` partagé
  (échec interne → externe jamais lancé, rollback immédiat ; échec externe →
  même rollback). Permet de distinguer "bug applicatif" (interne KO) de
  "souci d'infra/accès" (externe KO seul).
- **Ne pas renforcer `smoke_test_task`** avec des checks externes : ça
  dupliquerait ce que la couche 3 fera de toute façon quelques secondes plus
  tard dans le même run, en brouillant le diagnostic rapide/étroit que
  `smoke_test_task` est censé fournir (conteneur cassé vs chemin externe cassé).
- **Ne pas fusionner Healthcheck et test-api/test-gradio-public** : coûts/effets
  de bord réels (vraie prédiction, spam rate-limit, latence via Prefect) vs
  bénéfice — le vrai manque est la découvrabilité (le bouton "Tester l'API"
  existe déjà dans Orchestration), pas l'absence de la fonctionnalité. Piste
  envisagée : un lien depuis les lignes "Accès" du Healthcheck vers ce bouton,
  plutôt qu'une fusion des mécanismes.

**Why** : demandé explicitement par l'utilisateur ("Bien garder cette logique
en mémoire car structurante") après une session de rationalisation complète
des mécanismes de test post-déploiement.
