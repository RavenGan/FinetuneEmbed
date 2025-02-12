#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N max_grad_norm      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"

# Use the model GIST-small-Embedding-v0 only based on the rank in MTEB
### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/max_grad_norm/DosageSensitivity_finetune_gradnorm10_auc.csv \
    --output_path ./res/2025_0212_finetune/max_grad_norm/Sensitivity_model_gradnorm10_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --max_grad_norm 1

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/max_grad_norm/DosageSensitivity_finetune_gradnorm09_auc.csv \
    --output_path ./res/2025_0212_finetune/max_grad_norm/Sensitivity_model_gradnorm09_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --max_grad_norm 0.9

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/max_grad_norm/DosageSensitivity_finetune_gradnorm08_auc.csv \
    --output_path ./res/2025_0212_finetune/max_grad_norm/Sensitivity_model_gradnorm08_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --max_grad_norm 0.8

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/max_grad_norm/DosageSensitivity_finetune_gradnorm07_auc.csv \
    --output_path ./res/2025_0212_finetune/max_grad_norm/Sensitivity_model_gradnorm07_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --max_grad_norm 0.7

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/max_grad_norm/DosageSensitivity_finetune_gradnorm06_auc.csv \
    --output_path ./res/2025_0212_finetune/max_grad_norm/Sensitivity_model_gradnorm06_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --max_grad_norm 0.6