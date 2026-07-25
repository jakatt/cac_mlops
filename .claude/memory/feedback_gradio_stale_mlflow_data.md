---
name: feedback-gradio-stale-mlflow-data
description: "Cockpit gradio affiche des données MLflow périmées/fantômes (versions inexistantes) — root cause non trouvée, restart marche mais n'explique rien"
metadata:
  type: project
---

**Symptôme récurrent (2 occurrences : 2026-07-24 et 2026-07-25) :** le tableau
"Modèles — Versions et promotion" du Cockpit (`_load_models_data()` dans
`services/gradio/app.py`) affiche des versions MLflow qui **n'existent plus
réellement** dans le backend au moment de l'affichage — vérifié systématiquement
via 3 canaux indépendants qui donnent TOUS la même réponse correcte (jamais
celle affichée par le navigateur) :
1. `curl` direct sur l'API REST MLflow (`/api/2.0/mlflow/model-versions/search`)
2. Client Python `mlflow.tracking.MlflowClient()` exécuté frais (`docker exec ... python3 -c`)
3. Le MÊME client Python exécuté **depuis l'intérieur du conteneur gradio**
   (`docker exec cac_mlops-gradio-1 python3 -c ...`), donc même réseau/mêmes
   variables d'env que le process gradio réel — donne quand même le résultat
   CORRECT, contrairement à ce que montre le navigateur au même moment.

**2026-07-25, cas précis :** après un reset_flow complet + full_retrain_flow
(4 cycles), le backend MLflow montre exactement 4 versions par modèle
(1,2,3,4) — vérifié 3 fois. Le navigateur (après clic "Rafraichir" ET rechargement
de page) montre `lgbm:v1` à `v4` **ET AUSSI** `lgbm:v28` à `v31` (avec les
métriques exactes du run réel promu la veille, avant le reset — donc ce ne
sont pas des valeurs aléatoires, mais un vrai état antérieur qui persiste).

**Ce qui a été définitivement éliminé comme cause :**
- Cache navigateur / websocket périmé — un hard refresh (Cmd+Shift+R) et un
  nouvel onglet ne changent rien (contrairement à un incident similaire mais
  différent du 2026-07-23, où un hard refresh AVAIT suffi — donc PAS le même
  bug malgré la ressemblance de surface).
- Duplication dans le code : `ALL_MODEL_NAMES` ne contient que 3 entrées
  uniques, pas de double câblage du bouton "Rafraichir" (une seule occurrence
  de `.click(fn=refresh_models, ...)`).
- Le conteneur `mlflow` n'a pas été recréé (`docker inspect` : `Created` =
  2026-07-09, inchangé depuis — élimine l'hypothèse "DNS/IP périmé après un
  recreate du conteneur mlflow").
- Pas de caching explicite dans le code Python (`_load_models_data()` crée un
  `MlflowClient()` neuf à chaque appel, aucun `@lru_cache`/`functools.cache`
  trouvé dans tout `app.py`).

**Ce qui RESTE à investiguer demain (pas encore fait) :**
- Le process gradio a-t-il plusieurs workers internes (uvicorn/gunicorn avec
  `--workers > 1`) ? Si oui, un worker pourrait servir une réponse mise en
  cache par un mécanisme qu'on n'a pas identifié pendant qu'un AUTRE worker
  (celui qu'on teste via docker exec, ou celui qui a démarré après coup)
  donnerait la bonne réponse. Vérifier `ps`/`/proc` DANS le process gradio
  actif (pas un nouveau docker exec) pour voir s'il y a plusieurs processus
  Python.
- Est-ce que MLflow (serveur) ou une couche intermédiaire (pas de proxy connu
  entre gradio et mlflow en théorie, à reconfirmer) envoie des en-têtes de
  cache HTTP (`Cache-Control`, `ETag`) que le client `requests`/`mlflow`
  pourrait honorer d'une façon qu'on n'a pas anticipée ?
  Vérifier `requirements.txt` de gradio pour `requests-cache` ou équivalent
  (pas trouvé au premier coup d'œil mais à re-vérifier explicitement).
- Est-ce spécifique à `reset_flow`/`full_retrain_flow` (les 2 occurrences ont
  eu lieu juste après ce genre d'opération lourde sur MLflow/Postgres) ? Si
  oui, chercher ce que ces flows font PRÉCISÉMENT à la base Postgres
  (`clear_postgres_full`, VACUUM, changement de connexion) qui pourrait
  invalider une connexion/pool que le process gradio garde ouverte plus
  longtemps qu'il ne devrait (ex: `psycopg2`/SQLAlchemy connection pooling
  côté MLflow lui-même, PAS côté gradio directement, mais gradio interroge
  MLflow qui lui-même interroge Postgres — la staleness pourrait être
  côté MLflow-vers-Postgres plutôt que gradio-vers-MLflow, à vérifier en
  refaisant le test #1 (curl direct) IMMÉDIATEMENT après un reset, sans
  attendre, pour voir si MÊME l'API REST directe est parfois stale juste
  après un reset avant de se stabiliser).

**Ce qui marche à chaque fois (mais n'explique rien) :** `docker compose
restart gradio`. Le user a explicitement refusé cette solution "1 shot" le
2026-07-25 et demande la vraie root cause avant d'appliquer quoi que ce soit
— **ne pas se contenter de redémarrer sans creuser davantage la prochaine
fois que ça arrive.**

**How to apply:** Si ce bug réapparaît, commencer par les pistes non encore
testées ci-dessus AVANT tout restart. Si le restart doit être fait en urgence
(prod bloquée), le faire, mais continuer l'investigation séparément plutôt
que de considérer le problème résolu.
