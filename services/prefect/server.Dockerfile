FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Minimal Prefect server deps only — no ML stack
RUN pip install --no-cache-dir \
    "prefect>=3.0.0" \
    "aiosqlite>=0.19.0" \
    "alembic>=1.12.0"

ENV PREFECT_HOME=/prefect
EXPOSE 4200

HEALTHCHECK --interval=20s --timeout=5s --retries=10 \
    CMD curl -f http://localhost:4200/api/health || exit 1

CMD ["prefect", "server", "start", "--host", "0.0.0.0"]
