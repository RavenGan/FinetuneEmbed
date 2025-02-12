#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N warmup_ratio      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"

# Use the model GIST-small-Embedding-v0 only based on the rank in MTEB
### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/warmup_ratio/DosageSensitivity_finetune_warmup000_auc.csv \
    --output_path ./res/2025_0212_finetune/warmup_ratio/Sensitivity_model_warmup000_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --warmup_ratio 0

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/warmup_ratio/DosageSensitivity_finetune_warmup005_auc.csv \
    --output_path ./res/2025_0212_finetune/warmup_ratio/Sensitivity_model_warmup005_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --warmup_ratio 0.05

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/warmup_ratio/DosageSensitivity_finetune_warmup010_auc.csv \
    --output_path ./res/2025_0212_finetune/warmup_ratio/Sensitivity_model_warmup010_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --warmup_ratio 0.1

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/warmup_ratio/DosageSensitivity_finetune_warmup015_auc.csv \
    --output_path ./res/2025_0212_finetune/warmup_ratio/Sensitivity_model_warmup015_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --warmup_ratio 0.15

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/warmup_ratio/DosageSensitivity_finetune_warmup020_auc.csv \
    --output_path ./res/2025_0212_finetune/warmup_ratio/Sensitivity_model_warmup020_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --warmup_ratio 0.2