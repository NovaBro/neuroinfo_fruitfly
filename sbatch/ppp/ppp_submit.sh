#!/bin/bash
# Retired launcher. Train and inference are separate submitters.
echo "ppp_submit.sh has been split. Use:" >&2
echo "  ./sbatch/ppp/ppp_submit_train.sh [--dry-run] <preset> [flags…]" >&2
echo "  ./sbatch/ppp/ppp_submit_test.sh [--dry-run] <mode> <preset> [flags…]" >&2
echo "Test modes: array | run | evaluate | ckpt-sweep | vi-sweep" >&2
echo "Train, then infer: pass the train job id as --dependency on the test command." >&2
exit 1
