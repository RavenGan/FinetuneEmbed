#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N num_train_epochs      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"

# Use the model GIST-small-Embedding-v0 only based on the rank in MTEB
### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/num_train_epochs/DosageSensitivity_finetune_epoch5_auc.csv \
    --output_path ./res/2025_0212_finetune/num_train_epochs/Sensitivity_model_epoch5_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_train_epochs 5

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/num_train_epochs/DosageSensitivity_finetune_epoch10_auc.csv \
    --output_path ./res/2025_0212_finetune/num_train_epochs/Sensitivity_model_epoch10_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_train_epochs 10

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/num_train_epochs/DosageSensitivity_finetune_epoch15_auc.csv \
    --output_path ./res/2025_0212_finetune/num_train_epochs/Sensitivity_model_epoch15_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_train_epochs 15

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/num_train_epochs/DosageSensitivity_finetune_epoch20_auc.csv \
    --output_path ./res/2025_0212_finetune/num_train_epochs/Sensitivity_model_epoch20_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_train_epochs 20

python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2025_0212_finetune/num_train_epochs/DosageSensitivity_finetune_epoch25_auc.csv \
    --output_path ./res/2025_0212_finetune/num_train_epochs/Sensitivity_model_epoch25_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_train_epochs 25