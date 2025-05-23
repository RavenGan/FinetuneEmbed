#!/bin/bash
#$ -M dgan@nd.edu
#$ -m abe
#$ -q long
#$ -pe smp 1                 # Specify number of cores to use.
#$ -N General_embed       # Specify job name

conda activate Diff.gene
python3 hs_general.py