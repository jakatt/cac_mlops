---
name: feedback-icloud-sync
description: Le repo vit dans ~/Documents synchronisé iCloud Drive — des fichiers peuvent devenir "dataless" et bloquer indéfiniment les commandes git
metadata:
  type: feedback
---

Le working directory (`~/Documents/IA & Python/cac_mlops/cac_mlops`) est dans un dossier synchronisé iCloud Drive avec l'optimisation de stockage activée. Des fichiers du repo peuvent devenir "dataless" (placeholder, contenu resté dans le cloud, visible via `ls -la@O` → flag `dataless`). Toute commande qui lit leur contenu (`git status`, `git log`, `file`, `cat`...) déclenche un téléchargement iCloud silencieux et **bloque sans erreur ni timeout visible** tant que la sync n'est pas terminée.

**Symptôme observé** : `git status`/`git log` timeout après 2min+ sans sortie, plusieurs process `git status` zombies accumulés (visibles via `ps aux | grep git`), `lsof -p <pid>` montre le process bloqué en lecture sur un fichier précis (FD ouvert, pas de progression).

**Why:** Ça s'est produit sur ce repo précisément (juillet 2026) après un changement de Mac — la sync iCloud entre les deux machines n'était pas terminée, laissant des fichiers dataless côté local.

**How to apply:** Si des commandes git (ou autres lectures de fichiers) semblent bloquées sans raison apparente sur ce projet, vérifier en premier si c'est un problème de sync iCloud avant de creuser côté git/hooks/réseau : `ls -la@O <fichier>` doit afficher `dataless` si c'est le cas. Ne pas forcer un téléchargement en masse sans accord de l'utilisateur (il gère la sync iCloud lui-même) — attendre que la sync se termine puis retester.
