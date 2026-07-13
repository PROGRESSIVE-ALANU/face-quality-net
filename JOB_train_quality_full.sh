#!/bin/bash

#$ -M aogunley@nd.edu        # Email address for job notification
#$ -m ae                     # Send mail when job ends and aborts
#$ -q gpu@@crc_a10           # A10 (Ampere) nodes only — Pascal Titan Xp nodes fail with CUDA 13
#$ -l gpu_card=1             # Number of GPUs
#$ -pe smp 8                 # 8 CPU cores so the data loaders keep the GPU fed on 70k images
#$ -N train_quality_full     # Job name

# Use the CRC-provided PyTorch module (the conda env 'myenviroment' is empty).
module load pytorch/2.9.1

# Show which GPU we actually got, for the log
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Re-train on the COMPLETE OFIQ output (all images scored by the time this runs).
# The script auto-filters to rows whose image exists and drops any failed/NaN rows,
# so it transparently handles ffhq_MISSING_list.txt cases.
# Uses SmallResNet (resnet_small, ~0.5M params) with spatial dropout and
# UnifiedQualityScore.native (the raw linear measure) to reduce overfitting.
#
# --img-size 256 is FFHQ's native resolution -- train_quality.py's --img-size
# now DEFAULTS to 256 too, so this flag is redundant with the code's own
# default rather than overriding a mismatched one. Left explicit here on
# purpose, as a second, visible confirmation of what resolution is used.
#
# --patience 0 (early stopping OFF) per Spencer Giddens: early stopping
# wasn't measurably helping or hurting on this data, so simplest is to just
# run the full requested epoch count every time. Also now the code's own
# default, kept explicit here for the same reason as --img-size above.
#
# RandomResizedCrop removed from the augmentation pipeline (see
# train_quality.py's "strong" augmentation block) per Spencer Giddens:
# OFIQ's score depends on the whole image, so randomly cropping before
# training mismatches the label with what the model actually sees.
python train_quality.py \
    --csv ffhq_all_results.csv \
    --root . \
    --target UnifiedQualityScore.native \
    --arch resnet_small \
    --loss combined \
    --grad-clip 1.0 \
    --lr 1e-3 \
    --epochs 40 \
    --patience 0 \
    --batch-size 64 \
    --workers 8 \
    --img-size 256 \
    --out best_model_full.pt \
    --log train_log_full.csv \
    --curve train_curve_full.png

echo "FULL retrain finished at: $(date)"
