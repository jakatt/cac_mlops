---
name: feedback-gradio-stale-mlflow-data
description: "RÉSOLU (PR220) — Cockpit affichait des versions MLflow périmées : value= figé au démarrage du process, jamais recalculé par session"
metadata:
  type: project
---

**RÉSOLU 2026-07-26, PR #220.** Root cause trouvée après 2 incidents (2026-07-24,
2026-07-25) et une investigation multi-session (pistes éliminées : cache
navigateur, workers multiples, `psycopg2` inutilisé, recreate du conteneur
mlflow, caching explicite Python — voir historique de ce fichier avant PR220
si besoin de détail sur ce qui a été écarté).

**Root cause réelle :** dans `services/gradio/app.py`, l'onglet "Modèles —
Versions et promotion" appelait `_load_models_data()` **une seule fois, au
moment de la construction du layout `gr.Blocks()`** (donc une seule fois par
démarrage du process gradio, ligne ~2499) :
```python
_init_df, _init_choices = _load_models_data()
models_table = gr.Dataframe(value=_init_df, ...)
```
Ce résultat était figé dans la valeur statique du composant. Toute session,
onglet ou navigateur qui se connectait ENSUITE recevait ce même instantané —
même si le registre MLflow avait changé depuis (promotion, `reset_flow`,
`full_retrain_flow`). Seul un restart complet du process gradio recalculait
cet instantané (ré-exécution du code au niveau module), ce qui explique
parfaitement pourquoi seul un `docker compose restart gradio` "réglait" le
problème sans qu'on comprenne pourquoi, et pourquoi un simple hard-refresh ou
un changement de navigateur ne suffisait jamais (ce n'est pas un problème
client, c'est le SERVEUR qui sert la même donnée figée à tout le monde).

**Comment ça a été confirmé, sans ambiguïté :** `curl http://<tailscale-ip>:7860/config`
(endpoint natif Gradio qui expose la config JSON du layout, y compris les
`value=` statiques de chaque composant) — a permis de lire directement la
valeur figée côté serveur, indépendamment de tout navigateur.

**Fix (PR220) :** ajout d'un hook `demo.load(fn=refresh_models, outputs=[models_table,
promote_dd])` — recharge les données MLflow à chaque connexion/session au
lieu de dépendre d'un clic manuel sur "Rafraîchir" ou d'un restart du
container.

**How to apply :** si un autre onglet du Cockpit montre le même symptôme
(données figées jusqu'à restart), chercher le même anti-pattern : un
`gr.Dataframe`/`gr.Dropdown`/etc. dont le `value=` initial est calculé une
fois hors d'un callback, sans hook `demo.load()` associé. [[feedback_verify_before_asserting]]
