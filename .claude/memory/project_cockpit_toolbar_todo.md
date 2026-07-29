---
name: project-cockpit-toolbar-todo
description: "Backend confirmé correct (2026-07-29) — si le bouton 'Fermer tous les accordéons' semble encore cassé, c'est un cache navigateur, pas un bug Python"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

**MàJ 2026-07-29 — root cause isolée, backend disculpé.** `gradio_client`
n'est pas fiable pour tester cet event : ses fonctions `predict()`/`view_api()`
sont conçues pour des composants "porteurs de valeur" (Number, Textbox…) et
ne représentent pas correctement les mises à jour `gr.Accordion(open=...)` —
un test via `gradio_client.predict(api_name=...)` renvoie `()` même quand le
serveur répond correctement (faux négatif, m'a fait perdre du temps).

Test fiable : requête brute sur le protocole queue de Gradio (celui que le
navigateur utilise réellement) —
```python
requests.post(f"{base}/gradio_api/queue/join", json={
    "data": [], "event_data": None, "fn_index": <N>, "trigger_id": None,
    "session_hash": session_hash,
})
# puis GET /gradio_api/queue/data?session_hash=... (SSE)
```
Résultat obtenu en direct sur le VPS (gradio-1, code à jour, fn_index=31 pour
`collapse_all_btn`) : `process_completed` avec
`"data":[{"open":false,"__type__":"update"} ×6]` — **exactement correct**.
Le backend Python envoie la bonne réponse à chaque fois.

**Conclusion : le bug n'est PAS dans `services/gradio/app.py`.** Si
l'utilisateur constate encore le problème, c'est très probablement un bundle
JS Gradio mis en cache par le navigateur (ce toolbar a changé de forme sur
plusieurs PR consécutives cette semaine — PR237/238/239) — aucune directive
de cache agressive côté nginx (`services/nginx/nginx.conf`) n'explique un
cache serveur, donc cache navigateur par défaut. **Premier réflexe avant de
retoucher le code : hard refresh (Cmd+Shift+R / Ctrl+Shift+R) ou navigation
privée.** Ne plus retenter de fix Python sur cette base tant que ce test n'a
pas été fait.

Le bouton toolbar "⊟" (fermer tous les accordéons, `collapse_all_btn` dans
`services/gradio/app.py`) reste inopérant sur l'onglet Cockpit en prod
(kapsule/VPS), constaté par l'utilisateur le 2026-07-28 **après déploiement
de PR239** — donc la 2e tentative de fix n'a pas non plus résolu le
problème.

**Historique des tentatives (toutes deux dans PR239, commit `2b7b08e`)** :
1. Suppression du chevauchement CSS risqué (margin négative + overlay qui
   avait cassé toute la navigation par onglets dans PR237/238).
2. `[gr.Accordion(open=False)] * len(_ALL_ACCORDIONS)` (objet partagé)
   remplacé par une list comprehension générant une instance par sortie.
3. Ajout de `queue=False` sur `collapse_all_btn.click(...)` et
   `home_btn.click(...)`, hypothèse : le clic restait bloqué derrière les
   polls Prefect (timer 20s) partageant la même queue Gradio.

Aucune de ces 3 corrections n'a résolu le symptôme signalé. Le bouton
`home_btn` (icône accueil "⌂"), lui, n'a jamais été signalé comme cassé —
le problème semble spécifique à `collapse_all_btn`/aux accordéons.

**Pistes non encore explorées pour la prochaine session** :
- Vérifier si `_ALL_ACCORDIONS` référence bien les 6 objets `Accordion`
  réellement montés dans le DOM (pas des références obsolètes suite à un
  refactor antérieur — cf. les blocs `if not IS_KAPSULE:` autour de
  `acc_validation`/`acc_orchestration`/`acc_drift`/`acc_modeles`, `IS_KAPSULE`
  étant du code mort toujours `False` en pratique, mais à re-vérifier).
- Tester en dehors du navigateur (ex. `gradio_client`) si l'event
  `collapse_all_btn.click` renvoie bien une réponse avec les 6 updates
  `open=False`, pour isoler si c'est un bug backend (Python/Gradio) ou
  purement un problème de rendu front (le composant Accordion ignore la
  mise à jour `open` reçue).
- Vérifier la version de Gradio réellement utilisée dans l'image buildée
  (`gradio>=4.40.0` dans `services/gradio/requirements.txt` — la version
  exacte résolue au build peut avoir un comportement différent pour les
  updates de composant `Accordion`).

**Why** : signalé explicitement par l'utilisateur ("Ajoute au todo que le
bouton 'Fermer tous les accordéons' ne fonctionne pas dans l'onglet
cockpit") après avoir vérifié le déploiement de PR239 — à netraiter en
priorité la prochaine session plutôt que de retenter une correction à
l'aveugle.
