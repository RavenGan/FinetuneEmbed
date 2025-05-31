#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q long              # Specify queue
#$ -pe smp 1            # Specify number of cores to use.
#$ -N MedEmbed_name     # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"

python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --do_cv \
    --n_PCs 20 \
    --save_root "./res/2025_0531" \
    --model_name 'abhinand/MedEmbed-small-v0.1' \
    --save_mod_name "MedEmbed-small-v0.1"


python MultiMod_Multi_class.py \
    --task "Multi_class" \
    --embedding_data "TrainEvalTestData_Name" \
    --embedding_type "name_embedding" \
    --n_PCs 20 \
    --save_root "./res/2025_0531" \
    --model_name 'abhinand/MedEmbed-small-v0.1' \
    --save_mod_name "MedEmbed-small-v0.1"
