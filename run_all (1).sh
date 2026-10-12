#!/bin/bash
set -euo pipefail

EXTRA_ARGS="${@}"

echo "[1/4] Running BASELINE condition"
python "run_experiment.py" --condition baseline $EXTRA_ARGS

echo ""
echo "[2/4] Running MPS condition"
python "run_experiment.py" --condition mps $EXTRA_ARGS

echo ""
echo "[3/4] Running SPATIAL condition"
python "run_experiment.py" --condition spatial $EXTRA_ARGS

echo ""
echo "[4/4] Analyzing results"
python "analyze.py"

echo ""
echo "Done"