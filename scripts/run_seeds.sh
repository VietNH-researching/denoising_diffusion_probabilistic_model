#!/usr/bin/env bash
# Seed sweep: run every (experiment, seed) script one after another.
#
# Each run writes to results/<experiment>/seed<N>/: log.txt, figures/,
# stdout.log (the script's full output) and reverse_sde/metrics.json. A failed
# run is reported and the sweep moves on to the next one.
#
# Usage (from the repo root):
#   scripts/run_seeds.sh
#
# Overrides (environment variables):
#   EXPERIMENTS="sine"   experiments, in run order (notebooks/<name>/<name>.py)
#   SEEDS="0 1 2"        seeds per experiment
#   GPU=0                physical GPU index (CUDA_VISIBLE_DEVICES)
#   PYTHON=python        interpreter with the ddpm package installed

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS="${EXPERIMENTS:-sine}"
SEEDS="${SEEDS:-0 1 2}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
failed=0

for exp in $EXPERIMENTS; do
    nb_dir="$ROOT/notebooks/$exp"
    if [[ ! -f "$nb_dir/$exp.py" ]]; then
        echo "[$(stamp)] $exp: no notebooks/$exp/$exp.py, skipping"
        continue
    fi
    for seed in $SEEDS; do
        out_dir="$ROOT/results/$exp/seed$seed"
        mkdir -p "$out_dir"
        echo "[$(stamp)] $exp seed$seed: start"
        t0=$SECONDS
        ( cd "$nb_dir" && exec env \
            CUDA_VISIBLE_DEVICES="$GPU" RUN_SEED="$seed" MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
            "$PYTHON" "$exp.py" ) > "$out_dir/stdout.log" 2>&1 < /dev/null
        code=$?
        (( code != 0 )) && failed=$(( failed + 1 ))
        echo "[$(stamp)] $exp seed$seed: exit=$code after $(( SECONDS - t0 )) s"
    done
done

echo "[$(stamp)] sweep end, $failed failed run(s)"
exit $(( failed > 0 ))
