---
name: project-setup
description: "Origin and structure of this repo — fork of DataScientest MLOps accidents template, venv location, past cleanup"
metadata: 
  node_type: memory
  type: project
  originSessionId: c1965358-1fae-477b-8340-fdec26e82732
---

This repo (`IA & Python/cac_mlops/cac_mlops/`) is a fork of `https://github.com/DataScientest-Studio/Template_MLOps_accidents`, used for an MLOps training project on road accident data.

Setup history (as of 2026-06-16):
- Originally cloned into a messy nested structure (`cac_mlops/Template_MLOps_accidents` clone + a separately-moved venv named `cac`), then re-organized into the current flat repo with git history reset (fresh `Initial commit` + a `chore: remove duplicate files from template copy` commit) rather than keeping the original GitHub history.
- The working venv is `my_env/` (Python 3.13.1) directly inside this repo root, with `requirements.txt` already installed (numpy, pandas, scikit-learn, imbalanced-learn, flake8, etc.). Activate with `source my_env/bin/activate`.
- Cleaned up on 2026-06-16: removed leftover empty " 2"-suffixed duplicate folders (artifacts of a Finder copy) and the old duplicate clone `cac_mlops/Template_MLOps_accidents` (516MB) that lived one level up — that clone was the original git clone with real GitHub history/remote, now gone in favor of this fresh-history repo.

**Why:** Claude Code sessions are tied to the exact working directory path, so changing folders (e.g. `cac` → `cac_mlops` → `cac_mlops/cac_mlops`) starts a brand-new session with no memory of prior chat. Prior session transcripts are still recoverable from `~/.claude/projects/<encoded-old-path>/*.jsonl` if needed again.

**How to apply:** If asked to recover lost history again, check sibling encoded-path folders under `~/.claude/projects/` for `.jsonl` transcripts from the old working directory. Don't recreate the cleanup investigation — this memory already covers the current clean state.
