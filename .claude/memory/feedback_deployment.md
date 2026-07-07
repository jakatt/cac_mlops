---
name: feedback-deployment
description: "Règles de déploiement VPS — mounts docker-compose, prefect deploy, image rebuild"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8980ac59-2dd0-47a0-ab4c-af544d359117
---

Les images Docker (`ghcr.io/jakatt/cac-mlops-api:latest`, `cac-mlops-gradio:latest`) sont baked — elles ne contiennent pas les dernières modifications de code. Pour déployer sans rebuild, monter les fichiers modifiés via docker-compose volumes.

**Fichiers actuellement montés en override (à maintenir) :**
- `./src:/app/src:ro` → flows Prefect dans prefect-worker
- `./prefect.yaml:/app/prefect.yaml:ro` → deployments Prefect dans prefect-worker
- `./services/gradio/app.py:/app/services/gradio/app.py:ro` → cockpit Gradio

**Why:** Sans ces mounts, `prefect deploy --all` lit l'ancien `prefect.yaml` baked, et le Gradio affiche l'ancienne interface. Découvert le 2026-06-25 après avoir constaté que les nouveaux flows n'apparaissaient pas et que le cockpit n'avait pas l'onglet Pipeline.

**How to apply:** Après chaque modification de `src/flows/`, `prefect.yaml`, ou `services/gradio/app.py`, un simple `git pull + docker compose restart <service>` suffit. Pas besoin de rebuilder l'image.

**Pour `prefect deploy --all` :** Le `prefect.yaml` doit être à jour dans le conteneur (via mount). Sinon, faire `docker compose cp prefect.yaml prefect-worker:/app/prefect.yaml` avant de déployer.

**Cockpit Gradio — Gradio 6.0 breaking change :** Les paramètres `css` et `theme` doivent être dans `launch()`, pas dans `gr.Blocks()`. Corrigé le 2026-06-25.

**Gradio derrière nginx (root_path) :** Gradio 6.x embed l'IP du conteneur Docker (`172.x.x.x`) dans `window.gradio_config.root` si `root_path` n'est pas défini. Le browser essaie de se connecter à cette IP interne → "Loading..." indéfiniment. Fix : passer `root_path=os.getenv("GRADIO_PUBLIC_URL", "")` dans `demo.launch()` ET mettre `GRADIO_PUBLIC_URL=https://mlops.jakat-inc.fr` dans docker-compose. IMPORTANT : doit être HTTPS depuis PR #72 — une valeur `http://` sur une page HTTPS = mixed content bloqué par le browser. Ne pas oublier de `docker compose up -d` (pas juste `restart`) pour que la nouvelle env var soit prise en compte.

**gradio-public : fichiers à monter en override :**
- `./services/gradio/app_public.py:/app/services/gradio/app_public.py:ro`
- `./services/gradio/scenarios.py:/app/services/gradio/scenarios.py:ro` — ajouté PR #74 (baked image n'avait pas les scénarios vélo/moto)
- Modifiable sans rebuild — `docker compose restart gradio-public` suffit.

**Deploy direct VPS (sans attendre GH Actions) :**
Pour les fichiers bind-montés (`src/flows/`, `services/gradio/`) → `scp` + `docker compose restart` immédiats.
Pour un Dockerfile modifié → build sur VPS via `nohup bash -c 'docker build ... > /tmp/build.log 2>&1; echo BUILD_DONE >> /tmp/build.log' &` puis poll `/tmp/build.log`, puis `docker compose stop/rm -f/up -d <service>`.
**Why:** GH Actions prend 20-30 min (build + push + Trivy). Pour les urgences ou itérations rapides, déployer en direct. Mais créer quand même une PR pour syncer `main`.
**How to apply:** Quand le user dit "déploie en direct" → SCP + SSH sans passer par GH Actions. Toujours créer une PR après pour garder `main` à jour.
