#!/bin/bash
#SBATCH --job-name=cs_causal
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Discovery (SLURM): set partition / account for your lab, e.g.:
#SBATCH --partition=gpu
# #SBATCH --account=YOUR_ACCOUNT
#
# Distinguish runs by tag + job id under results/ and checkpoints/
# Override when submitting:
#   EXP_NAME=my_ablation sbatch jobs/train_causal_eval_discovery.sbatch

set -euo pipefail

EXP_NAME="${EXP_NAME:-bidir_train_causal_eval}"
RUN_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${SLURM_SUBMIT_DIR}/results/${EXP_NAME}_${RUN_ID}"
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/${EXP_NAME}_${RUN_ID}"

mkdir -p logs "${OUT_DIR}" "${CKPT_DIR}"

# --- Conda (edit paths for your cluster) ---
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate cs

cd "${SLURM_SUBMIT_DIR}"

export HF_HOME="${HF_HOME:-${SLURM_SUBMIT_DIR}/.hf_cache}"
export PYTHONUNBUFFERED=1

echo "EXP_NAME=${EXP_NAME} RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"

python scripts/train.py \
  --data data/preprocessed.pkl \
  --checkpoint "${CKPT_DIR}/best_xlmr.pt" \
  --results "${OUT_DIR}/train_results.pkl" \
  --results-json "${OUT_DIR}/train_results.json" \
  --causal-at-eval

echo "Done. Artifacts under ${OUT_DIR} and ${CKPT_DIR}"
