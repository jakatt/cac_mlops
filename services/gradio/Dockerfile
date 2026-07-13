FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/gradio/requirements.txt /tmp/gradio_req.txt
RUN pip install --no-cache-dir -r /tmp/gradio_req.txt

COPY services/gradio/ /app/services/gradio/
COPY src/ /app/src/

ENV PYTHONPATH=/app
ENV GRADIO_PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -sf http://localhost:${GRADIO_PORT}/ || exit 1

CMD ["python", "services/gradio/app.py"]
