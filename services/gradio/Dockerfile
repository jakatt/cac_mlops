FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/jakatt/cac_mlops" \
      org.opencontainers.image.description="Cockpit Gradio — interface admin MLOps (VPS, Tailscale) et interface publique de prédiction / what-if / points noirs"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

COPY services/gradio/requirements.txt /tmp/gradio_req.txt
# Téléchargements PyPI lents (gros paquets : xgboost ~130 Mo) : sans ces
# réglages, pip abandonne après 15 s sans données et fait échouer tout le
# CD (incident 01/10, 2 échecs de suite sur « Read timed out »).
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir -r /tmp/gradio_req.txt

COPY services/gradio/ /app/services/gradio/
COPY src/ /app/src/

ENV PYTHONPATH=/app
ENV GRADIO_PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -sf http://localhost:${GRADIO_PORT}/ || exit 1

CMD ["python", "services/gradio/app.py"]
