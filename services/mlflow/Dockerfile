FROM python:3.11-slim

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
RUN pip install --no-cache-dir mlflow>=3.15.0 psycopg2-binary boto3

EXPOSE 5000
