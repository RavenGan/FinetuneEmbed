#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N Sensitivity      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
### GIST-small-Embedding-v0---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/GIST-small-Embedding-v0/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/GIST-small-Embedding-v0/Sensitivity/Sensitivity_model_ \
#     --model_name avsolatorio/GIST-small-Embedding-v0

### NoInstruct-small-Embedding-v0 (not working)---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/NoInstruct-small-Embedding-v0/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/NoInstruct-small-Embedding-v0/Sensitivity/Sensitivity_model_ \
#     --model_name avsolatorio/NoInstruct-small-Embedding-v0

### stella-base-zh-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2024_1209/stella-base-zh-v2/Sensitivity/DosageSensitivity_finetune_auc.csv \
    --output_path ./res/2024_1209/stella-base-zh-v2/Sensitivity/Sensitivity_model_ \
    --model_name infgrad/stella-base-zh-v2