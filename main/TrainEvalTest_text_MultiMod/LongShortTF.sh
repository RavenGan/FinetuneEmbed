#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N LongShortTF      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
### NoInstruct-small-Embedding-v0 (not working)---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/long_vs_shortTF \
#     --csv_dir ./res/2024_1209/NoInstruct-small-Embedding-v0/LongShortTF/long_vs_shortTF_finetune_auc.csv \
#     --output_path ./res/2024_1209/NoInstruct-small-Embedding-v0/LongShortTF/LongShortTF_model_ \
#     --model_name avsolatorio/NoInstruct-small-Embedding-v0

### GIST-small-Embedding-v0---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/long_vs_shortTF \
#     --csv_dir ./res/2024_1209/GIST-small-Embedding-v0/LongShortTF/long_vs_shortTF_finetune_auc.csv \
#     --output_path ./res/2024_1209/GIST-small-Embedding-v0/LongShortTF/LongShortTF_model_ \
#     --model_name avsolatorio/GIST-small-Embedding-v0

### stella-base-zh-v2---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/long_vs_shortTF \
#     --csv_dir ./res/2024_1209/stella-base-zh-v2/LongShortTF/long_vs_shortTF_finetune_auc.csv \
#     --output_path ./res/2024_1209/stella-base-zh-v2/LongShortTF/LongShortTF_model_ \
#     --model_name infgrad/stella-base-zh-v2

### bge-small-en-v1.5---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2024_1209/bge-small-en-v1.5/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2024_1209/bge-small-en-v1.5/LongShortTF/LongShortTF_model_ \
    --model_name BAAI/bge-small-en-v1.5