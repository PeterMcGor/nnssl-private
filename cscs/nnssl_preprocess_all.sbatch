#!/bin/bash
#SBATCH --account=abs34
#SBATCH --job-name=nnssl_pp666
#SBATCH --partition=normal
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/nnssl_pp666_%A_%a.out
#SBATCH --environment=/iopsstor/scratch/cscs/petermcgor/nnssl-private/cscs/nnssl.toml

# Full FOMO300K preprocessing, reusing the pretraining plan (nnsslPlans.json).
# The collection is split into SLURM_ARRAY_TASK_COUNT non-overlapping parts;
# each array task preprocesses one part. A failed/killed task can be rerun
# alone with:  sbatch --array=<id> nnssl_preprocess_all.sbatch
set -euo pipefail

echo "START $(date -Is) part=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} host=$(hostname)"
echo "raw=${nnssl_raw}  pp=${nnssl_preprocessed}"

  -total_parts "${SLURM_ARRAY_TASK_COUNT}" -part "${SLURM_ARRAY_TASK_ID}" \
  -np 96 --verbose

echo "END $(date -Is) part=${SLURM_ARRAY_TASK_ID}"
