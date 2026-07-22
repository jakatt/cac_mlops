---
name: feedback-verify-before-asserting
description: "Le user challenge les affirmations de root-cause non vérifiées empiriquement — toujours prouver via inspection directe (SSH, logs, git history) avant de conclure, et distinguer 'confirmé' de 'théorie plausible'"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Le 2026-07-22, en diagnostiquant l'échec silencieux de `dvc_push_task` sur le VPS : première affirmation ("aucun montage du .git hôte vers le conteneur") faite sans vérification complète — le user a directement demandé "tu es sur de toi sur ce diagnostic ?". Rincidence quelques échanges plus tard : le user a aussi jugé bancale une proposition de fix (redémarrage SSH ciblé) qui n'avait pas assez anticipé les effets de bord, menant à une meilleure conception (fetch S3 + clone git jetable) après discussion.

**Why:** Ce user vérifie activement les diagnostics techniques et les designs proposés plutôt que de les accepter tels quels — il pousse sur "es-tu sûr ?" / "ça me semble bancal, d'autres options ?" quand quelque chose sent l'affirmation non prouvée ou la solution de facilité. Il valorise la rigueur empirique (SSH, logs Prefect, `git log`/`git show`, `docker compose config` résolu) plus qu'un raisonnement plausible mais non vérifié.

**How to apply:**
- Avant d'affirmer une root cause, vérifier directement (SSH, logs, inspection de fichiers réels) plutôt que déduire depuis la lecture de code seule.
- Quand une affirmation ne peut pas être prouvée à 100% (ex. reconstitution d'un historique), le dire explicitement — distinguer "confirmé empiriquement" de "théorie plausible, cohérente avec les preuves mais non certifiée".
- Quand on propose un fix, anticiper les effets de bord structurels (ex. self-référence : une tâche qui recrée le conteneur dans lequel elle tourne) avant de le présenter comme solution — pas juste "ça devrait marcher".
- Si le user pousse back avec une question directe sur la certitude d'un diagnostic, ne pas se contenter de réaffirmer — creuser plus profondément avec de nouvelles preuves (ex. `git log --format='%an <%ae>'`, `docker compose config` résolu, timestamps de création de conteneur) jusqu'à une conclusion réellement solide.
