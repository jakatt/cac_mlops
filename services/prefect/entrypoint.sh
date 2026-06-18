#!/bin/bash
set -e

echo "==> Waiting for Prefect server…"
until prefect version > /dev/null 2>&1; do
  sleep 3
done

echo "==> Creating work pool (idempotent)"
prefect work-pool create "default-process-pool" --type process 2>/dev/null || true

echo "==> Deploying flows"
cd /app && prefect --no-prompt deploy --all

echo "==> Starting worker"
exec prefect worker start --pool "default-process-pool"
