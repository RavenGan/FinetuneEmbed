#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q gpu@@li            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N GenePT_Multi      # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --do_cv \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0527"

python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --do_cv \
    --n_PCs 20 \
    --save_root "./res/2025_0527"

python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0527"

python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --n_PCs 20 \
    --save_root "./res/2025_0527"


python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_cv \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0527"


python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_cv \
    --n_PCs 20 \
    --save_root "./res/2025_0527"

python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0527"

python Gene_level_task_GenePT_1536.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --n_PCs 20 \
    --save_root "./res/2025_0527"