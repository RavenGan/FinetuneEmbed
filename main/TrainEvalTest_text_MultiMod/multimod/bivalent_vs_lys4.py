import os

# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
from mod.multi_mod import *

random_states = list(range(41, 51)) # set up the random seeds

model_names = ['avsolatorio/NoInstruct-small-Embedding-v0',
               'avsolatorio/GIST-small-Embedding-v0',
               'infgrad/stella-base-en-v2',
               'BAAI/bge-small-en-v1.5',
               'abhinand/MedEmbed-small-v0.1',
               'thenlper/gte-small',
               'intfloat/e5-small-v2',
               'avsolatorio/GIST-all-MiniLM-L6-v2',
               'intfloat/e5-small',
               'TaylorAI/gte-tiny']

save_mod_names = ['NoInstruct-small-Embedding-v0',
               'GIST-small-Embedding-v0',
               'stella-base-en-v2',
               'bge-small-en-v1.5',
               'MedEmbed-small-v0.1',
               'gte-small',
               'e5-small-v2',
               'GIST-all-MiniLM-L6-v2',
               'e5-small',
               'gte-tiny']

do_cv = True

for i in range(len(model_names)):
     model_name = model_names[i]
     save_mod_name = save_mod_names[i]

     ## Bivalent vs. lys4
     data_dir = "./data/MethylationState/bivalent_vs_lys4"
     save_csv_dir = "./res/2024_1221/" + save_mod_name + "_bivalent_vs_lys4_auc.csv"
     smallmod_multiple_run_TrainTest(data_dir, save_csv_dir, random_states, model_name, do_cv)
