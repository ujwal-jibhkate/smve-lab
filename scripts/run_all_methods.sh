#!/usr/bin/env bash
# Run every first-stage method on one dataset (stage 3c), then the first-stage report.
#   bash scripts/run_all_methods.sh nfcorpus
# Needs the dataset's BGE-M3 embeddings (scripts/encode_beir.py). Steps that already
# have results are skipped where the script supports it (--skip-existing).
set -euo pipefail
DS="${1:?usage: run_all_methods.sh <dataset>}"
cd "$(dirname "$0")/.."
py() { uv run python "$@"; }
py scripts/evaluate_maxsim.py --dataset "$DS"
py scripts/evaluate_bgem3_single_vector.py --dataset "$DS"
py scripts/evaluate_bm25.py --dataset "$DS"
# SMVE: a representative grid (the SciFact sweep showed k matters most, then w)
py scripts/evaluate_smve.py --dataset "$DS" --w 4096 16384 65536 --k 16 32 64 --center off on --light --skip-existing
# MUVERA: the region that was competitive on SciFact
py scripts/evaluate_muvera.py --dataset "$DS" --reps 20 40 --k-sim 4 5 --d-proj 16 32 --light --skip-existing
py scripts/compare_first_stages.py --dataset "$DS"
echo "DONE $DS"
