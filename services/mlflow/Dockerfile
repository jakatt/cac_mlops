FROM python:3.11-slim

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow>=3.9.0 psycopg2-binary boto3

EXPOSE 5000
