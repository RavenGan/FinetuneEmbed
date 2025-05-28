#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N MultiClass_Name      # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
### biobert-base-cased-v1.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/biobert-base-cased-v1.1/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/biobert-base-cased-v1.1/MultiClass/MultiClass_model_ \
    --model_name dmis-lab/biobert-base-cased-v1.1 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/biobert-base-cased-v1.1/MultiClass

### NoInstruct-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/NoInstruct-small-Embedding-v0/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/NoInstruct-small-Embedding-v0/MultiClass/MultiClass_model_ \
    --model_name avsolatorio/NoInstruct-small-Embedding-v0 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/NoInstruct-small-Embedding-v0/MultiClass

### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/GIST-small-Embedding-v0/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/GIST-small-Embedding-v0/MultiClass/MultiClass_model_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/GIST-small-Embedding-v0/MultiClass

### stella-base-en-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/stella-base-en-v2/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/stella-base-en-v2/MultiClass/MultiClass_model_ \
    --model_name infgrad/stella-base-en-v2 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/stella-base-en-v2/MultiClass

### bge-small-en-v1.5---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/bge-small-en-v1.5/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/bge-small-en-v1.5/MultiClass/MultiClass_model_ \
    --model_name BAAI/bge-small-en-v1.5 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/bge-small-en-v1.5/MultiClass

### MedEmbed-small-v0.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/MedEmbed-small-v0.1/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/MedEmbed-small-v0.1/MultiClass/MultiClass_model_ \
    --model_name abhinand/MedEmbed-small-v0.1 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/MedEmbed-small-v0.1/MultiClass

### gte-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/gte-small/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/gte-small/MultiClass/MultiClass_model_ \
    --model_name thenlper/gte-small \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/gte-small/MultiClass

### e5-small-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/e5-small-v2/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/e5-small-v2/MultiClass/MultiClass_model_ \
    --model_name intfloat/e5-small-v2 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/e5-small-v2/MultiClass

### GIST-all-MiniLM-L6-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/GIST-all-MiniLM-L6-v2/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/GIST-all-MiniLM-L6-v2/MultiClass/MultiClass_model_ \
    --model_name avsolatorio/GIST-all-MiniLM-L6-v2 \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/GIST-all-MiniLM-L6-v2/MultiClass

### e5-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/e5-small/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/e5-small/MultiClass/MultiClass_model_ \
    --model_name intfloat/e5-small \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/e5-small/MultiClass

### gte-tiny---------------------------------------------------
python finetuning.py \
    --data_dir ./data/Multi_class/TrainEvalTestData_Name \
    --csv_dir ./res/2025_0528_finetune/Gene_Name/gte-tiny/MultiClass \
    --output_path ./res/2025_0528_finetune/Gene_Name/gte-tiny/MultiClass/MultiClass_model_ \
    --model_name TaylorAI/gte-tiny \
    --num_classes 15 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Name/gte-tiny/MultiClass