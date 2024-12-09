#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N BivalentLys4      # Specify job name

module load conda
source activate base
conda activate Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
### GIST-small-Embedding-v0---------------------------------------------------
python finetuning.py \
    --data_dir ./data/MethylationState/bivalent_vs_lys4 \
    --csv_dir ./res/2024_1209/GIST-small-Embedding-v0/BivalentLys4/bivalent_vs_lys4_finetune_auc.csv \
    --output_path ./res/2024_1209/GIST-small-Embedding-v0/BivalentLys4/BivalentLys4_model_ \
    --model_name avsolatorio/GIST-small-Embedding-v0