---
name: user-vps-access
description: "Accès SSH au VPS Scaleway — utilisateur deploy, IP, fail2ban"
metadata: 
  node_type: memory
  type: user
  originSessionId: 8980ac59-2dd0-47a0-ab4c-af544d359117
---

Le compte SSH du VPS est **`deploy`**, pas `root`. L'accès root est refusé.

- IP publique : `51.159.187.132`
- IP Tailscale : `100.117.99.62` (Tailscale doit être connecté des deux côtés)
- Répertoire du projet : `/data/cac_mlops`
- Répertoire volumes Docker : `/data` (block storage Scaleway 80 GB)

**Clé SSH** : La clé ed25519 du Mac de Jacques (`~/.ssh/id_ed25519`) a été ajoutée à `/home/deploy/.ssh/authorized_keys` le 2026-06-25. Si ça ne marche plus, vérifier que le VPS n'a pas été recréé.

**fail2ban** : Le VPS a fail2ban actif. Après plusieurs tentatives SSH échouées, l'IP est bannie. Pour débanner : `sudo fail2ban-client set sshd unbanip <ip>`. Mon IP Mac = `78.243.65.17` (peut changer).

**Ports** : Les ports admin (7860, 4200, 3000, etc.) sont liés à `VPS_TAILSCALE_IP=100.117.99.62`, pas à localhost. Pour tester depuis le VPS lui-même : `curl http://100.117.99.62:7860/`.

**Organisation équipe (2 personnes)**
- Jacques (cet utilisateur) → branche `jacques`
- Collègue → branche `noel`
- Workflow : chacun ouvre une PR vers `main` et peut la merger lui-même dès que le CI passe
- Branch protection main : CI obligatoire uniquement — pas de review requise
