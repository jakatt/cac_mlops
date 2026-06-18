FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/prefect/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source (same layout as the api container)
COPY src/ /app/src/

WORKDIR /app
ENV PYTHONPATH=/app

# Entrypoint script handles: pool creation + flow deployment + worker start
COPY services/prefect/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
