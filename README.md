# CAC MLOps — Gravité des accidents de la route

Système MLOps de bout en bout qui prédit si un accident de la route impliquera une victime grave (hospitalisée ou tuée), à partir des données officielles ONISR publiées sur data.gouv.fr (2021 → 2024).

- **Cockpit public** : [mlops.jakat-inc.fr](https://mlops.jakat-inc.fr) (Predict · What-if · Points noirs)
- **Documentation complète** (15 documents, à jour) : [mlops.jakat-inc.fr/ci-docs](https://mlops.jakat-inc.fr/ci-docs/presentation.html)

## En bref

| Élément | Détail |
| --- | --- |
| Modèle en production | `lgbm_accidents` v4 @Production — entraîné sur 2021-2023, testé sur 2024 · F1 0.689 · AUC 0.837 · Recall 0.767 · Accuracy 0.764 |
| Données | ONISR, 4 fichiers CSV par an — validation Pandera 3 niveaux + auto-correction, versioning DVC (Scaleway S3) |
| Entraînement | Benchmark RF / XGBoost / LightGBM à chaque cycle, suivi MLflow (runs + Model Registry) |
| Mise en production | 3 déclencheurs : nouvelles données (T1), nouveau code (T2), nouveau blueprint d'hyperparamètres (T3) — toujours une **validation humaine GO/STOP** avant toute interruption, rollback automatique si un test échoue |
| Orchestration | Prefect (18 deployments) · CI/CD GitHub Actions (3 workflows) |
| Service | FastAPI + JWT derrière nginx (rate-limit à 2 niveaux) et Caddy (TLS) |
| Infrastructure | VPS Scaleway (Docker Compose, 16 conteneurs) + cluster Kubernetes Kapsule à la demande (zéro interruption) |
| Observabilité | Prometheus · Loki · Grafana (7 dashboards, 12 alertes email) · drift Evidently |
| Sécurité | Outils d'administration accessibles uniquement via Tailscale · scan Trivy des images · pip-audit |

## Documentation

| Document | Pour qui |
| --- | --- |
| [Présentation du projet](https://mlops.jakat-inc.fr/ci-docs/presentation.html) | Point d'entrée · décideurs |
| [Flux MLOps](https://mlops.jakat-inc.fr/ci-docs/flux_mlops.html) | Public non technique |
| [Architecture](https://mlops.jakat-inc.fr/ci-docs/architecture.html) · [Guide administrateur](https://mlops.jakat-inc.fr/ci-docs/guide_administrateur.html) | Équipe technique |
| [Mécanismes de résilience](https://mlops.jakat-inc.fr/ci-docs/resilience_mechanisms.html) | Gates, rollbacks, interruptions par déclencheur |
| [Monitoring](https://mlops.jakat-inc.fr/ci-docs/monitoring.html) | Prometheus, Loki, Grafana : collecte, indicateurs des 7 dashboards, alertes |
| [Guide Data Scientist](https://mlops.jakat-inc.fr/ci-docs/ds_guide.html) · [Guide hyperparamètres](https://mlops.jakat-inc.fr/ci-docs/hyperparams_guide.html) | Data scientists |
| [Guide MLOps Engineer](https://mlops.jakat-inc.fr/ci-docs/mlops_eng_guide.html) · [Guide MLOps Lead](https://mlops.jakat-inc.fr/ci-docs/mlops_lead_guide.html) | Exploitation |
| [Catalogue des tests](https://mlops.jakat-inc.fr/ci-docs/tests_catalogue.html) · [Catalogue ETL](https://mlops.jakat-inc.fr/ci-docs/etl_catalogue.html) · [Lignée des données](https://mlops.jakat-inc.fr/ci-docs/data_lineage.html) | Qualité et traçabilité |

Les sources de ces pages sont dans [`docs/`](docs/).

## Démarrage rapide (développeur)

```bash
git clone git@github.com:jakatt/cac_mlops.git && cd cac_mlops
python3 -m venv my_env && source my_env/bin/activate
pip install -r requirements.txt "dvc[s3]>=3.0" && pip install -e .
cp .env.example .env
./scripts/ds_session_start.sh      # synchronise la branche + dvc pull des données
pytest tests/unit/ -v              # 76 tests, ceux exécutés par la CI
```

L'accès à MLflow, Prefect et Grafana nécessite de rejoindre le tailnet du projet (Tailscale).

## Règles de contribution

- Travail sur les branches `mlops` (code) ou `DS` (blueprint), jamais directement sur `main`.
- Toute modification passe par une PR vers `main` ; la CI (lint, pip-audit, tests) doit être verte.
- Ne jamais mélanger `config/model_params.yml` et du code source dans la même PR (bloqué par la CI).

---

<sub>Projet réalisé dans le cadre de la formation MLOps DataScientest, à partir du template [DataScientest-Studio/Template_MLOps_accidents](https://github.com/DataScientest-Studio/Template_MLOps_accidents).</sub>
