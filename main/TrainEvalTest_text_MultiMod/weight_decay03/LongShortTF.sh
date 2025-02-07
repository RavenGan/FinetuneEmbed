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
### NoInstruct-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/NoInstruct-small-Embedding-v0/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/NoInstruct-small-Embedding-v0/LongShortTF/LongShortTF_model_ \
    --model_name avsolatorio/NoInstruct-small-Embedding-v0 \
    --weight_decay 0.3

### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/GIST-small-Embedding-v0/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/GIST-small-Embedding-v0/LongShortTF/LongShortTF_model_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --weight_decay 0.3

### stella-base-en-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/stella-base-en-v2/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/stella-base-en-v2/LongShortTF/LongShortTF_model_ \
    --model_name infgrad/stella-base-en-v2 \
    --weight_decay 0.3

### bge-small-en-v1.5---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/bge-small-en-v1.5/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/bge-small-en-v1.5/LongShortTF/LongShortTF_model_ \
    --model_name BAAI/bge-small-en-v1.5 \
    --weight_decay 0.3

### MedEmbed-small-v0.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/MedEmbed-small-v0.1/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/MedEmbed-small-v0.1/LongShortTF/LongShortTF_model_ \
    --model_name abhinand/MedEmbed-small-v0.1 \
    --weight_decay 0.3

### gte-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/gte-small/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/gte-small/LongShortTF/LongShortTF_model_ \
    --model_name thenlper/gte-small \
    --weight_decay 0.3

### e5-small-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/e5-small-v2/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/e5-small-v2/LongShortTF/LongShortTF_model_ \
    --model_name intfloat/e5-small-v2 \
    --weight_decay 0.3

### GIST-all-MiniLM-L6-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/GIST-all-MiniLM-L6-v2/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/GIST-all-MiniLM-L6-v2/LongShortTF/LongShortTF_model_ \
    --model_name avsolatorio/GIST-all-MiniLM-L6-v2 \
    --weight_decay 0.3

### e5-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/e5-small/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/e5-small/LongShortTF/LongShortTF_model_ \
    --model_name intfloat/e5-small \
    --weight_decay 0.3

### gte-tiny---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF \
    --csv_dir ./res/2025_0206_finetune/weight_decay03/gte-tiny/LongShortTF/long_vs_shortTF_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay03/gte-tiny/LongShortTF/LongShortTF_model_ \
    --model_name TaylorAI/gte-tiny \
    --weight_decay 0.3