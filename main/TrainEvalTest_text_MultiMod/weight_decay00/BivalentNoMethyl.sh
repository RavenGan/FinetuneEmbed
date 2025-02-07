#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N BivalentNoMethyl      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
# ### NoInstruct-small-Embedding-v0---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/NoInstruct-small-Embedding-v0/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/NoInstruct-small-Embedding-v0/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name avsolatorio/NoInstruct-small-Embedding-v0 \
#     --weight_decay 0

# ### GIST-small-Embedding-v0---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/GIST-small-Embedding-v0/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/GIST-small-Embedding-v0/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name avsolatorio/GIST-small-Embedding-v0 \
#     --weight_decay 0

###stella-base-en-v2---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
    --csv_dir ./res/2025_0206_finetune/weight_decay00/stella-base-en-v2/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
    --output_path ./res/2025_0206_finetune/weight_decay00/stella-base-en-v2/BivalentNoMethyl/BivalentNoMethyl_model_ \
    --model_name infgrad/stella-base-en-v2 \
    --weight_decay 0

# ###bge-small-en-v1.5---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/bge-small-en-v1.5/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/bge-small-en-v1.5/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name BAAI/bge-small-en-v1.5 \
#     --weight_decay 0

# ###MedEmbed-small-v0.1---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/MedEmbed-small-v0.1/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/MedEmbed-small-v0.1/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name abhinand/MedEmbed-small-v0.1 \
#     --weight_decay 0

# ###gte-small---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/gte-small/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/gte-small/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name thenlper/gte-small \
#     --weight_decay 0

# ###e5-small-v2---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/e5-small-v2/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/e5-small-v2/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name intfloat/e5-small-v2 \
#     --weight_decay 0

# ###GIST-all-MiniLM-L6-v2---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/GIST-all-MiniLM-L6-v2/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/GIST-all-MiniLM-L6-v2/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name avsolatorio/GIST-all-MiniLM-L6-v2 \
#     --weight_decay 0

# ###e5-small---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/e5-small/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/e5-small/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name intfloat/e5-small \
#     --weight_decay 0

# ###gte-tiny---------------------------------------------------
# python finetuning.py \
#     --data_dir ./data/MethylationState/bivalent_vs_no_methyl \
#     --csv_dir ./res/2025_0206_finetune/weight_decay00/gte-tiny/BivalentNoMethyl/bivalent_vs_no_methyl_finetune_auc.csv \
#     --output_path ./res/2025_0206_finetune/weight_decay00/gte-tiny/BivalentNoMethyl/BivalentNoMethyl_model_ \
#     --model_name TaylorAI/gte-tiny \
#     --weight_decay 0