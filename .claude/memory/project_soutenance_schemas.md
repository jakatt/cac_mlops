---
name: project-soutenance-schemas
description: "Décisions de schémas soutenance — triggers 1/2/3, CI/CD, sécurité, slide v2"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

Schémas PowerPoint développés pour la soutenance MLOps. Fichiers dans le répertoire racine du projet.

## Schéma de principe (3 triggers)

**Trigger 1 — Nouvelles données** (`**` Pas de CI/CD)
- Source : données sécurité routière gouv.fr, MAJ annuelle
- Workflow : 100% Prefect automatique — full-retrain-flow → gate manuelle → promote → Kapsule
- Pas de CI/CD, pas de push de code

**Trigger 2 — Nouveau code** (`*` Pas de réentraînement) — Elon (MLOps eng), branche `mlops`
- Maintenance de la chaîne MLOps (infra, monitoring…)
- Workflow : CI → Léon merge → CD (build Docker + Trivy + deploy + smoke) → deploy-vps-flow → gate → promote → Kapsule

**Trigger 3 — Trigger dégradation** (Lucie DS, branche `DS`)
- Réoptimisation hyperparamètres quand métriques post-retrain annuel insuffisantes
- Cas réel : hyperparamètres 2021 potentiellement sous-optimaux avec 3x plus de données en 2024
- Workflow : CI → Léon merge → CD (pas de build Docker) → update-model-flow → train_flow(year=2023, cumul=True) → gate → promote → Kapsule
- STOP si aucun modèle ne bat @Production → blueprint restauré + mail DS

**Why:** Trigger 3 n'est PAS une activité régulière planifiée. C'est réactif : déclenché si les métriques dégradent après trigger 1.

## Schéma CI/CD — Trigger 2 (Flux de travail collaboratif)
- CI : tests auto (pip-audit, flake8, pytest) + STOP
- CD : build 3 images + Trivy STOP, deploy VPS, smoke test rollback
- Mise en Prod : "Flow Prefect déclenché automatiquement après le déploiement. Orchestre l'activation du modèle en production." → Gate manuelle (step 1, en rouge), promote, 5 tests rollback, Kapsule rollback

## Schéma CI/CD — Trigger 3 (Flux de travail collaboratif)
- CI : identique trigger 2
- CD : sans build Docker (juste deploy VPS + smoke test)
- Mise en Prod : "Flow Prefect déclenché automatiquement après le déploiement. Orchestre le réentraînement et l'activation du modèle en production." → Step 1 : réentraînement 3 modèles (train-flow) + STOP si pas d'amélioration, Step 2 : Gate manuelle, Step 3 : promote, Step 4 : 5 tests rollback, Step 5 : Kapsule rollback

## Mécanisme blueprint (implémenté, rien à modifier)
- `config/model_params.yml` (YAML) = blueprint hyperparamètres DS
- `src/scripts/extract_blueprint.py` : lit run MLflow tagué `export_to_prod=true` dans `accidents_severity_explore` → met à jour le YAML
- `src/flows/update_model_flow.py` : chaîne complète trigger 3 (backup → extract → train_flow → deploy_vps_flow si champion)
- `deploy.yml` : détecte changement `config/model_params.yml` → déclenche `update-model-flow` au lieu de `deploy-vps-flow`

## Slide sécurité v2
- Fichier `security_slide_v2.pptx` supprimé lors du ménage workspace (2026-07-02)
- 4 colonnes : PRÉVENTION (pip-audit, Trivy, Secrets) → ISOLATION (UFW, Tailscale, Docker network) → PROTECTION (Nginx, JWT, fail2ban) → DÉTECTION (Brute force, DDoS, Loki)
- **Mises à jour post-session 2026-07-03 :**
  - ISOLATION : UFW note → port 8090 maintenant `127.0.0.1` binding (Docker bypass UFW via iptables)
  - PROTECTION : Nginx → row port `8090` → `127.0.0.1:8090`
  - Ajouter **Caddy** comme couche TLS : terminaison HTTPS · Let's Encrypt · HTTP→HTTPS redirect · proxy `mlops.jakat-inc.fr → nginx:8090`
  - Accès public : `http://51.159.187.132:8090` → `https://mlops.jakat-inc.fr`

## Slide architecture globale — état final session 2026-07-04

Fichier : `architecture_slide.pptx` (racine du projet). Script : `/scratchpad/make_archi_slide.py`.

**Structure validée :**
- **DEV LOCAL** (zone gauche) : Code & Qualité, Expérimentation ML, Admin à distance
- **GitHub CI/CD** (boîte en haut, entre DEV et VPS) — cloud Microsoft, indépendant
  - Flèche DEV → GitHub : **"git push · PR"** (déclenche ci.yml)
  - Flèche GitHub → VPS : **"merge main → déploiement"** (déclenche deploy.yml auto)
- **VPS SCALEWAY** (zone centrale) : Entrée du trafic · Monitoring · Interface · Orchestration · Plateforme ML
- **CLOUD SCALEWAY** (boîte pointillée englobante) : VPS + S3 + Kapsule
- **Service Scaleway S3** (bande inférieure, DANS Cloud Scaleway mais HORS VPS)
  - DVC push : Orchestration VPS → S3
  - DVC pull : S3 → DEV LOCAL
  - Promotion modèle : Plateforme ML VPS → S3 → Kapsule
- **CLOUD SERVICE KUBERNETES / Kapsule** (zone droite) : API publique · Orchestration · Monitoring · Auto-scaling
- **VPN Tailscale** (icône entre DEV LOCAL et VPS) — tunnel bidirectionnel DEV ↔ VPS spécifiquement

**Distinctions importantes à retenir pour l'oral :**
- **MinIO** tourne sur le VPS (Docker container) = stockage local artefacts MLflow
- **Scaleway Object Storage (S3)** = service cloud Scaleway externe = remote DVC + modèles pour Kapsule
- Tailscale connecte DEV ↔ VPS uniquement (pas S3 ni Kapsule)
