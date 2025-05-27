#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q long              # Specify queue
#$ -pe smp 1            # Specify number of cores to use.
#$ -N Multi_class     # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --do_cv \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0528"

python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --do_cv \
    --n_PCs 20 \
    --save_root "./res/2025_0528"

python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0528"

python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData" \
    --embedding_type "text_embedding" \
    --n_PCs 20 \
    --save_root "./res/2025_0528"


python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_cv \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0528"


python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_cv \
    --n_PCs 20 \
    --save_root "./res/2025_0528"

python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_pca \
    --n_PCs 20 \
    --save_root "./res/2025_0528"

python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --n_PCs 20 \
    --save_root "./res/2025_0528"