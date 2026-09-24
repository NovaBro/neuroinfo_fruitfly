#!/bin/bash
# DEPRECATED: one-shot train+predict in a single job OOMs on L40S after train
# holds ~40GiB VRAM, and @fork on train hangs gunpowder PreCache.
# Use the train→infer chain instead (from repo root):
#     bash sbatch/ppp/ppp_basic_8h_chain.sh
echo "Error: ppp_basic_8h_sbatch.sh is deprecated (one-shot train+predict OOMs)." >&2
echo "Submit with: bash sbatch/ppp/ppp_basic_8h_chain.sh" >&2
exit 1
