import os
import sys
# sys.path.append('/Users/david/Desktop/FinetuneEmbed')
sys.path.append('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')

# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
from mod.multi_mod import *

random_states = list(range(41, 51)) # set up the random seeds

model_names = ["dmis-lab/biobert-base-cased-v1.1",
               'avsolatorio/NoInstruct-small-Embedding-v0',
               'avsolatorio/GIST-small-Embedding-v0',
               'infgrad/stella-base-en-v2',
               'BAAI/bge-small-en-v1.5',
               'abhinand/MedEmbed-small-v0.1',
               'thenlper/gte-small',
               'intfloat/e5-small-v2',
               'avsolatorio/GIST-all-MiniLM-L6-v2',
               'intfloat/e5-small',
               'TaylorAI/gte-tiny']

save_mod_names = ["biobert-base-cased-v1.1",
                  'NoInstruct-small-Embedding-v0',
                  'GIST-small-Embedding-v0',
                  'stella-base-en-v2',
                  'bge-small-en-v1.5',
                  'MedEmbed-small-v0.1',
                  'gte-small',
                  'e5-small-v2',
                  'GIST-all-MiniLM-L6-v2',
                  'e5-small',
                  'gte-tiny']

task = "long_vs_shortTF"  # Specify the task
embedding_data = "TrainEvalTestData"

data_dir = f"./data/{task}/{embedding_data}"

embedding_type = "text_embedding"  # Specify the type of embedding

do_cv = False
do_pca = False
n_PCs = 20 

if do_cv==True and do_pca==True:
     folder_name = f"PCA_CV_{embedding_type}"
elif do_cv==True and do_pca==False:
     folder_name = f"PCA_NoCV_{embedding_type}"
elif do_cv==False and do_pca==True:
     folder_name = f"NoPCA_CV_{embedding_type}"
elif do_cv==False and do_pca==False:
     folder_name = f"NoPCA_NoCV_{embedding_type}"

save_root = "./res/2025_0525"

for i in range(len(model_names)):
     model_name = model_names[i]
     save_mod_name = save_mod_names[i]

     ## Long- vs short- range TFs
     # The input data used here are downloaded from Chen et al. (2020) 
     # (link: https://www-nature-com.stanford.idm.oclc.org/articles/s41467-020-16106-x).
     save_csv_dir = f"{save_root}/{folder_name}/" + save_mod_name + f"_{task}_NumRes.csv"
     os.makedirs(os.path.dirname(save_csv_dir), exist_ok=True)

     ROC_save_dir = f"{save_root}/{folder_name}/{task}/" + save_mod_name + "/"

     smallmod_multiple_run_TrainTest(data_dir, save_csv_dir, random_states, model_name, do_cv,
                                     ROC_save_dir, do_pca=do_pca, n_PCs=n_PCs)