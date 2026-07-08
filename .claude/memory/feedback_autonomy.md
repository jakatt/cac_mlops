---
name: feedback-autonomy
description: "User delegates fully — work alone, never interrupt to ask questions"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

User explicitly delegates full autonomy: "fais tout toi-meme je suis perdu pour le moment" and "vas y seul sans rien me demander, tu as carte blanche".

**Why:** User trusts Claude to handle multi-step infra/CI tasks end-to-end without hand-holding. Gets frustrated when blocked mid-task and asked to do things manually.

**How to apply:** When given a complex task (deploy, fix, migrate, test), carry it through to completion autonomously. Only surface blockers if genuinely impossible without user action (e.g., adding a GitHub Secret that doesn't exist). Never pause mid-pipeline to ask for confirmation on the next step.
