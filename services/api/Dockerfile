FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# kubectl (flows kapsule)
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl && mv kubectl /usr/local/bin/

# scw CLI (flows kapsule)
RUN curl -sL "https://github.com/scaleway/scaleway-cli/releases/latest/download/scaleway-cli_linux_x86_64" \
    -o /usr/local/bin/scw && chmod +x /usr/local/bin/scw

# docker CLI (flow diag — accès via /var/run/docker.sock)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# project source (for src/ imports)
COPY .dvc/config .dvc/config
COPY src/ src/
COPY services/api/app/ services/api/app/
COPY services/monitoring/ services/monitoring/
COPY services/__init__.py services/__init__.py
COPY scripts/ scripts/
COPY config/ config/
COPY prefect.yaml .

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
