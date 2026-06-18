# Claude Code Setup Guide

A first-time setup guide distilled from my workflow on NYU HPC. Generic — works for any new project. Copy/adapt as needed.

---

## 1. CLAUDE.md Hierarchy

Use **two levels** of `CLAUDE.md`:

### Workspace-level (`/scratch/<netid>/CLAUDE.md`)
Shared facts that apply to everything in scratch:
- HPC environment (SLURM accounts, partitions, log dirs)
- Conda envs and their paths
- Singularity container inventory (path + use case table)
- Cross-project conventions (W&B entity, cache dirs)
- GPU monitoring helpers (e.g. `check_gpu.sh`)

### Project-level (`<project>/CLAUDE.md`)
Project-specific:
- One-paragraph project overview
- Architecture / pipeline stages
- Core scripts + their relationships (call graph in prose)
- Common commands (sbatch jobs, data prep, eval)
- Data locations table (path + size)
- Known gotchas (AMP NaN, NFS sockets, pretrained-init requirements, etc.)
- Latest results / open problems

**Rule:** project CLAUDE.md *adds* to workspace CLAUDE.md, never duplicates.

---

## 2. Mandatory Code Verification (Codex)

In every project CLAUDE.md, include a "Mandatory Code Verification" section:

> Before committing or submitting SLURM jobs, feed changed files to `mcp__codex__codex` with `@file.py` syntax, ask for bugs/logic/wiring issues, fix Critical/High, then proceed. Use `approval-policy: "never"`. Skip only for typos / comment-only edits.

This is non-optional and has caught real bugs before jobs ran.

---

## 3. Auto-Memory System (Index-as-Graph)

Directory: `~/.claude/projects/<project-id>/memory/`

### Core idea
`MEMORY.md` is **not** where memories live — it's a **one-line-per-file index** that acts like a graph node pointing to many specialized memory files. This lets Claude scan ~200 lines of hooks and pull only the relevant files into context, instead of loading everything.

### Types (each memory file is typed)
- **user** — role, preferences, expertise
- **feedback** — rules to follow (with **Why:** and **How to apply:** lines)
- **project** — active state: who/what/why/deadlines
- **reference** — stable pointers (W&B, Ray API, dataset paths, etc.)

### File structure
Each memory file has YAML frontmatter:
```markdown
---
name: hebo torch conflict
description: HEBO pip pulls torch into hebo_pkg; must rm -rf after install
type: feedback
---
<body>
```

### MEMORY.md structure (the graph index)
Pure pointers grouped by type — **no memory content ever lives here**:

```markdown
# <Project> Memory Index

## User
- [user_preferences.md](user_preferences.md) — model selection, memory style

## Feedback (rules to follow)
- [feedback_numpy_pickle.md](feedback_numpy_pickle.md) — CRITICAL: build datasets inside container
- [feedback_slurm_gpu_util.md](feedback_slurm_gpu_util.md) — l40s_public cancels <50% util jobs
- [feedback_hf_precache.md](feedback_hf_precache.md) — pre-download HF models BEFORE HF_HUB_OFFLINE=1

## References (stable facts)
- [reference_wandb.md](reference_wandb.md) — W&B project, entity, API key
- [reference_slurm.md](reference_slurm.md) — accounts, partitions, GPU specs
- [reference_datasets.md](reference_datasets.md) — dataset paths, sample counts

## Projects (current state)
- **Active session**: 013 spatial-loss-scaling (2026-04-19) — paused awaiting submit
- [project_active_runs.md](project_active_runs.md) — job status snapshot
- [project_todo.md](project_todo.md) — pending work
```

### Rules
- One line per entry, under ~150 chars
- Descriptions are **specific hooks** ("HEBO pip pulls torch" not "HEBO note") so matching is reliable
- Prefix with `CRITICAL:` for rules that cause data loss / silent bugs
- Keep the whole index under ~200 lines (truncated after)
- Organize by **topic**, not chronologically
- Update/remove stale entries; never leave dangling pointers

### What NOT to save
- Things derivable from the code or `git log`
- Fix recipes (the commit is authoritative)
- Ephemeral in-progress state (use plans/tasks instead)

---

## 4. Session Management

Sessions are numbered (`session 013 spatial-loss-scaling`) with a lightweight convention:
- `project_active_runs.md` — running/recent jobs
- `project_todo.md` — pending work
- `project_<topic>.md` — one per active thread

Resume via "resume session NNN" or "resume <name>" — the `resume-session` skill handles state load, job status check, and pending work summary.

---

## 5. Container-First Execution

**Never run training on login nodes.** Keep a container table in workspace CLAUDE.md:

| Container | Use Case |
|-----------|----------|
| base + CUDA | training/testing |
| base + Ray + Optuna | HP search |
| base + domain libs (e.g. ONE-api) | data processing |

Use `--nv` for GPU passthrough, `--writable-tmpfs` for in-container pip installs.

---

## 6. SLURM Conventions

- Log dir: `/scratch/<netid>/slurm_logs/` — consistent across projects
- `$SLURM_TMPDIR` for fast local SSD (stage data here)
- `RAY_TMPDIR=/tmp` when using Ray (NFS breaks Unix sockets)
- Set `#SBATCH --requeue` for long jobs so preemption recovers
- Partitions with GPU-utilization floors (e.g. `l40s_public` cancels <50% util for 2h) — budget accordingly

---

## 7. Useful Skills to Install

Project-scoped skills (under `.claude/skills/`):
- `submit-job` — sbatch + tail + verify start
- `slurm-status` — job status + GPU util + log tail
- `check-wandb` — run metrics / compare experiments
- `resume-session` — load session notes + check pending work
- `scan-dataset` — NaN/zero/Inf/session-overlap checks
- `ray-results` — top trials, best HPs, probe accuracy
- `init` — generate initial CLAUDE.md for a new repo

Generic helpers:
- `update-config` — settings.json / hooks / permissions
- `fewer-permission-prompts` — auto-allowlist from transcripts
- `review`, `security-review` — PR review passes

---

## 8. Execution Rules (What I've Learned)

Common pitfalls worth hard-coding in feedback memories:
- Always build datasets *inside* the container (numpy pickle versions diverge)
- Always load npz with `allow_pickle=True` + try/except
- Always re-rank ALL trials before declaring a "best config"
- Pre-download HF models BEFORE setting `HF_HUB_OFFLINE=1`
- When loading SSL wrapper ckpts: extract `backbone_state_dict`, never `strict=False` blindly
- Coord→voxel uses `int()` truncation, not `np.round()`
- Report per-axis R² for regression, not just mean Euclidean

Each of these should live as its own `feedback_*.md` file with **Why:** and **How to apply:**.

---

## 9. W&B Integration

Keep a `reference_wandb.md` with project name + entity + API key location. Cache to `/scratch/<netid>/wandb/` to avoid home quota.

---

## 10. Quick-Start Checklist for a New Project

1. Create `<project>/CLAUDE.md` with overview, pipeline, commands, gotchas
2. Add project to workspace CLAUDE.md if new conventions introduced
3. Create `.claude/skills/` with project-relevant skills
4. Seed `~/.claude/projects/<id>/memory/MEMORY.md` with user + reference memories
5. Set up `slurm_logs/` + W&B cache + container path in env
6. First real edit → run Codex verification end-to-end to confirm the loop works
7. Start session 001 with clear name (topic-scope)
