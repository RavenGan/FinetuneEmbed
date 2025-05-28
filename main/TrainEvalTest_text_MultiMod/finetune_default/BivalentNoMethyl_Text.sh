#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N BivalentNoMethyl_Text      # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
### biobert-base-cased-v1.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/biobert-base-cased-v1.1/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/biobert-base-cased-v1.1/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name dmis-lab/biobert-base-cased-v1.1 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/biobert-base-cased-v1.1/BivalentNoMethyl

### NoInstruct-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/NoInstruct-small-Embedding-v0/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/NoInstruct-small-Embedding-v0/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name avsolatorio/NoInstruct-small-Embedding-v0 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/NoInstruct-small-Embedding-v0/BivalentNoMethyl

### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/GIST-small-Embedding-v0/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/GIST-small-Embedding-v0/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/GIST-small-Embedding-v0/BivalentNoMethyl

### stella-base-en-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/stella-base-en-v2/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/stella-base-en-v2/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name infgrad/stella-base-en-v2 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/stella-base-en-v2/BivalentNoMethyl

### bge-small-en-v1.5---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/bge-small-en-v1.5/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/bge-small-en-v1.5/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name BAAI/bge-small-en-v1.5 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/bge-small-en-v1.5/BivalentNoMethyl

### MedEmbed-small-v0.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/MedEmbed-small-v0.1/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/MedEmbed-small-v0.1/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name abhinand/MedEmbed-small-v0.1 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/MedEmbed-small-v0.1/BivalentNoMethyl

### gte-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/gte-small/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/gte-small/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name thenlper/gte-small \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/gte-small/BivalentNoMethyl

### e5-small-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/e5-small-v2/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/e5-small-v2/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name intfloat/e5-small-v2 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/e5-small-v2/BivalentNoMethyl

### GIST-all-MiniLM-L6-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/GIST-all-MiniLM-L6-v2/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/GIST-all-MiniLM-L6-v2/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name avsolatorio/GIST-all-MiniLM-L6-v2 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/GIST-all-MiniLM-L6-v2/BivalentNoMethyl

### e5-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/e5-small/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/e5-small/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name intfloat/e5-small \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/e5-small/BivalentNoMethyl

### gte-tiny---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/gte-tiny/BivalentNoMethyl \
    --output_path ./res/2025_0528_finetune/Gene_Text/gte-tiny/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name TaylorAI/gte-tiny \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/gte-tiny/BivalentNoMethyl
