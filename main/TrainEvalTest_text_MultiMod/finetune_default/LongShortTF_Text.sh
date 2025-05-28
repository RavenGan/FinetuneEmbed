#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N LongShortTF_Text      # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
### biobert-base-cased-v1.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/biobert-base-cased-v1.1/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/biobert-base-cased-v1.1/LongShortTF/LongShortTF_model_ \
    --model_name dmis-lab/biobert-base-cased-v1.1 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/biobert-base-cased-v1.1/LongShortTF

### NoInstruct-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/NoInstruct-small-Embedding-v0/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/NoInstruct-small-Embedding-v0/LongShortTF/LongShortTF_model_ \
    --model_name avsolatorio/NoInstruct-small-Embedding-v0 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/NoInstruct-small-Embedding-v0/LongShortTF

### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/GIST-small-Embedding-v0/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/GIST-small-Embedding-v0/LongShortTF/LongShortTF_model_ \
    --model_name avsolatorio/GIST-small-Embedding-v0 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/GIST-small-Embedding-v0/LongShortTF

### stella-base-en-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/stella-base-en-v2/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/stella-base-en-v2/LongShortTF/LongShortTF_model_ \
    --model_name infgrad/stella-base-en-v2 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/stella-base-en-v2/LongShortTF

### bge-small-en-v1.5---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/bge-small-en-v1.5/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/bge-small-en-v1.5/LongShortTF/LongShortTF_model_ \
    --model_name BAAI/bge-small-en-v1.5 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/bge-small-en-v1.5/LongShortTF

### MedEmbed-small-v0.1---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/MedEmbed-small-v0.1/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/MedEmbed-small-v0.1/LongShortTF/LongShortTF_model_ \
    --model_name abhinand/MedEmbed-small-v0.1 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/MedEmbed-small-v0.1/LongShortTF

### gte-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/gte-small/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/gte-small/LongShortTF/LongShortTF_model_ \
    --model_name thenlper/gte-small \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/gte-small/LongShortTF

### e5-small-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/e5-small-v2/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/e5-small-v2/LongShortTF/LongShortTF_model_ \
    --model_name intfloat/e5-small-v2 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/e5-small-v2/LongShortTF

### GIST-all-MiniLM-L6-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/GIST-all-MiniLM-L6-v2/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/GIST-all-MiniLM-L6-v2/LongShortTF/LongShortTF_model_ \
    --model_name avsolatorio/GIST-all-MiniLM-L6-v2 \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/GIST-all-MiniLM-L6-v2/LongShortTF

### e5-small---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/e5-small/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/e5-small/LongShortTF/LongShortTF_model_ \
    --model_name intfloat/e5-small \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/e5-small/LongShortTF

### gte-tiny---------------------------------------------------
python finetuning.py \
    --data_dir ./data/long_vs_shortTF/TrainEvalTestData \
    --csv_dir ./res/2025_0528_finetune/Gene_Text/gte-tiny/LongShortTF \
    --output_path ./res/2025_0528_finetune/Gene_Text/gte-tiny/LongShortTF/LongShortTF_model_ \
    --model_name TaylorAI/gte-tiny \
    --num_classes 2 \
    --ROC_save_dir ./res/2025_0528_finetune/Gene_Text/gte-tiny/LongShortTF