#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe               # Send mail when job begins, ends and aborts
#$ -q long              # Specify queue
#$ -pe smp 1            # Specify number of cores to use.
#$ -N Yufei_Names     # Specify job name

module load conda
source activate base
conda activate /afs/crc/user/d/dgan/.conda/envs/Diff.gene

export PYTHONPATH="/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed:$PYTHONPATH"
# python GeneName_embed.py
# python prepare_15class_TrainEvalTest_text.py
# python prepare_TrainEvalTest_Name.py
# python GeneName_embed_MultiMod.py
# python Yufei_GeneTextHalved_embed_MultiMod.py
python Yufei_GeneName_embed_MultiMod.py