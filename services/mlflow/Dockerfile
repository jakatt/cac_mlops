FROM ghcr.io/mlflow/mlflow:v3.9.0
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir psycopg2-binary boto3
