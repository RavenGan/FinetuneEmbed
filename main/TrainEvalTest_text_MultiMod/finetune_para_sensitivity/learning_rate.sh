#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=0     # Run on 0 GPU card
#$ -N learning_rate      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"

# Use the model GIST-small-Embedding-v0 only based on the rank in MTEB
### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/learning_rate/DosageSensitivity_finetune_lr1e4_auc.csv \
    --output_path ./res/2025_0212_finetune/learning_rate/Sensitivity_model_lr1e4_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --learning_rate 1e-4

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/learning_rate/DosageSensitivity_finetune_lr5e5_auc.csv \
    --output_path ./res/2025_0212_finetune/learning_rate/Sensitivity_model_lr5e5_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --learning_rate 5e-5

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/learning_rate/DosageSensitivity_finetune_lr1e5_auc.csv \
    --output_path ./res/2025_0212_finetune/learning_rate/Sensitivity_model_lr1e5_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --learning_rate 1e-5

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/learning_rate/DosageSensitivity_finetune_lr5e6_auc.csv \
    --output_path ./res/2025_0212_finetune/learning_rate/Sensitivity_model_lr5e6_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --learning_rate 5e-6

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/learning_rate/DosageSensitivity_finetune_lr1e6_auc.csv \
    --output_path ./res/2025_0212_finetune/learning_rate/Sensitivity_model_lr1e6_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --learning_rate 1e-6