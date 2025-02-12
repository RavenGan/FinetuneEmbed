#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N weight_decay      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"

# Use the model GIST-small-Embedding-v0 only based on the rank in MTEB
### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/weight_decay/DosageSensitivity_finetune_decay00_auc.csv \
    --output_path ./res/2025_0212_finetune/weight_decay/Sensitivity_model_decay00_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --weight_decay 0

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/weight_decay/DosageSensitivity_finetune_decay01_auc.csv \
    --output_path ./res/2025_0212_finetune/weight_decay/Sensitivity_model_decay01_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --weight_decay 0.1

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/weight_decay/DosageSensitivity_finetune_decay02_auc.csv \
    --output_path ./res/2025_0212_finetune/weight_decay/Sensitivity_model_decay02_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --weight_decay 0.2

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/weight_decay/DosageSensitivity_finetune_decay03_auc.csv \
    --output_path ./res/2025_0212_finetune/weight_decay/Sensitivity_model_decay03_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --weight_decay 0.3

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/weight_decay/DosageSensitivity_finetune_decay04_auc.csv \
    --output_path ./res/2025_0212_finetune/weight_decay/Sensitivity_model_decay04_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --weight_decay 0.4