FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/jakatt/cac_mlops" \
      org.opencontainers.image.description="Serveur MLflow (tracking + model registry) avec pilotes PostgreSQL et S3 (psycopg2, boto3)"

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# mlflow>=3.15.0 : fix CVE-2026-64849 (SSRF non authentifié via webhook,
# CRITICAL — bloquant Trivy en CD). Le pin >=3.9.0 précédent résolvait vers
# une image en cache figée sur 3.14.0 (vulnérable) : Docker ne refait
# jamais un `pip install` déjà en cache, même si une version plus récente
# sort sur PyPI derrière un même pin ouvert — changer la contrainte
# force la résolution fraîche.
# Versions ÉPINGLÉES (plus de >=) : un rebuild pour un simple LABEL (26/09/2026)
# a tiré mlflow 3.16.1 — schéma de base plus récent que celui de PostgreSQL
# (« out-of-date database schema », migration irréversible) — et SQLAlchemy
# 2.1 (pilote psycopg v3 par défaut) : MLflow en crash-loop, prod coupée.
# Toute montée de version = décision explicite + `mlflow db upgrade` planifié.
RUN pip install --no-cache-dir "mlflow==3.15.1" "SQLAlchemy<2.1" psycopg2-binary boto3

EXPOSE 5000
