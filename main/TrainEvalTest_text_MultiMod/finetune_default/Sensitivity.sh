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
### NoInstruct-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/DosageSensitivity \
    --csv_dir ./res/2024_1209/NoInstruct-small-Embedding-v0/Sensitivity/DosageSensitivity_finetune_auc.csv \
    --output_path ./res/2024_1209/NoInstruct-small-Embedding-v0/Sensitivity/Sensitivity_model_ \
    --model_name avsolatorio/NoInstruct-small-Embedding-v0

### GIST-small-Embedding-v0---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/GIST-small-Embedding-v0/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/GIST-small-Embedding-v0/Sensitivity/Sensitivity_model_ \
#     --model_name avsolatorio/GIST-small-Embedding-v0

### stella-base-en-v2---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/stella-base-en-v2/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/stella-base-en-v2/Sensitivity/Sensitivity_model_ \
#     --model_name infgrad/stella-base-en-v2

### bge-small-en-v1.5---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/bge-small-en-v1.5/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/bge-small-en-v1.5/Sensitivity/Sensitivity_model_ \
#     --model_name BAAI/bge-small-en-v1.5

### MedEmbed-small-v0.1---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/MedEmbed-small-v0.1/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/MedEmbed-small-v0.1/Sensitivity/Sensitivity_model_ \
#     --model_name abhinand/MedEmbed-small-v0.1

### gte-small---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/gte-small/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/gte-small/Sensitivity/Sensitivity_model_ \
#     --model_name thenlper/gte-small

### e5-small-v2---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/e5-small-v2/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/e5-small-v2/Sensitivity/Sensitivity_model_ \
#     --model_name intfloat/e5-small-v2

### GIST-all-MiniLM-L6-v2---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/GIST-all-MiniLM-L6-v2/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/GIST-all-MiniLM-L6-v2/Sensitivity/Sensitivity_model_ \
#     --model_name avsolatorio/GIST-all-MiniLM-L6-v2

### e5-small---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/e5-small/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/e5-small/Sensitivity/Sensitivity_model_ \
#     --model_name intfloat/e5-small

### gte-tiny---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/DosageSensitivity \
#     --csv_dir ./res/2024_1209/gte-tiny/Sensitivity/DosageSensitivity_finetune_auc.csv \
#     --output_path ./res/2024_1209/gte-tiny/Sensitivity/Sensitivity_model_ \
#     --model_name TaylorAI/gte-tiny