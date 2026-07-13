#!/bin/bash

#$ -M aogunley@nd.edu        # Email address for job notification
#$ -m ae                     # Send mail when job ends and aborts
#$ -q gpu@@crc_a10           # A10 (Ampere) nodes only — Pascal Titan Xp nodes fail with CUDA 13
#$ -l gpu_card=1             # Number of GPUs
#$ -pe smp 4                 # CPU cores for the data loaders
#$ -N evaluate_full          # Job name

# Use the CRC-provided PyTorch module (the conda env 'myenviroment' is empty).
module load pytorch/2.9.1

# Show which GPU we actually got, for the log
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Score best_model_full.pt (the checkpoint JOB_train_quality_full.sh produces)
# on its exact held-out validation set (best_model_full.val_files.txt, 7,000
# images the model never trained on) and on a 7,000-image sample of the
# training set. --plot names the validation scatter; --train-plot defaults to
# "auto", which derives the training scatter's name from it automatically
# (<plot>_train.png), so only one name needs to be typed.
#
# Produces:
#   - eval_scatter_full.png        (validation scatter: predicted vs true score)
#   - eval_scatter_full_train.png  (training-set scatter, auto-named from the above)
#   - console metrics: MAE, RMSE, Pearson r, Spearman rho, R^2
#
# This is the actual, reproducible source of both files. Run it again any
# time to regenerate identical plots from the same checkpoint as proof they
# still work.
python evaluate.py \
    --model best_model_full.pt \
    --csv ffhq_all_results.csv \
    --root . \
    --plot eval_scatter_full.png

echo "Evaluation finished at: $(date)"
