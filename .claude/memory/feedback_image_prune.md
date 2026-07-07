---
name: feedback-image-prune
description: Never use docker image prune -af blindly — other apps share the VPS
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

The VPS runs another app alongside cac_mlops. `docker image prune -af` removes ALL unused images from the host, including images belonging to the other app if its containers happen to be stopped at that moment.

**Why:** User confirmed another web app runs on the same VPS. Aggressive Docker pruning could break it by removing its images.

**How to apply:** In deploy.yml and any maintenance scripts, only remove OUR specific images (`docker image rm ghcr.io/jakatt/cac-mlops-*`) rather than `docker image prune -af`. If a global prune is truly needed, first verify which images belong to other projects with `docker ps --all` and `docker image ls`.
