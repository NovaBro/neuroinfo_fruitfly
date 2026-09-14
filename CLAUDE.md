# CLAUDE.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code, match existing style, even if you'd do it differently.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Visualize

**Make nice, clear visualizations following common academic and industry standards.**

- Share axes between plots that are meant to be directly compared.
- Show examples of objects before and after pre/post-processing.
- Use seaborn for better clarity and plotly for interactivity.
- Use standard color palettes.

## 6. Git

- Never perform git operations other than `git status` without asking for permission, even on auto mode. **Never push**.
- Keep commit messages concise.
- Do not add Co-Authored-By to commit messages.

## 7. This Project (MANC hemilineage classification)

**Environment.** There is no usable bare python; everything runs in the container:

```
singularity exec --fakeroot \
  --overlay /scratch/vh2308/conda-envs/test_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh && conda activate /ext3/miniforge3/envs/neurofly_2 && python -u <script>"
```

`sacct`/`squeue` are the exception - host-only, must run _outside_ the container.

**Run work as SLURM batch jobs, never interactively on the login node** - quick sanity checks included. Chain multi-stage work with `--dependency=afterok:<jobid>` (or `afterany` when later stages should still run on partial success) so a pipeline finishes unattended. Accounts: `torch_pr_61_general` (default), `torch_pr_61_tandon_advanced`.

**Right-size resource requests.** Check `sacct -j <past_job> --format=JobID,MaxRSS,Elapsed` for a comparable job first. Over-requesting buys queue time, not safety - a 128G request for an 11GB job sat in `Priority` for hours while the same job at 32G started immediately.

**Long jobs must be resumable, and resumption must validate completeness.** Skip work whose output exists, but check the output is _whole_ (e.g. expected row count) - a killed job leaves truncated files that a bare `.exists()` check accepts as done.

**Data paths contain literal brackets** (`Skeletons_T[1]_[LR]`). `glob.glob` treats these as character classes and silently matches nothing; use `os.listdir` + filter.

**The neuprint auth token in `get_common_ids.py` and `ugw.py` is hardcoded deliberately.** Don't flag it or move it to an env var. Do still flag any _other_ credential that turns up.

## 8. Experiments

- When a preprocessing step drops samples, compare against the baseline **restricted to the surviving samples**, never the full-set baseline.
- Sweep hyperparameters and report the metric at the _selected_ value; never let a hardcoded reporting constant define the headline.
- Verify the mechanism, not just the metric. A diagnostic that explains _why_ something helped or failed catches errors the headline number hides.
- When a check returns an extreme result, suspect the check before the data.
- Leave superseded scripts in place; add new ones alongside rather than rewriting working code.
