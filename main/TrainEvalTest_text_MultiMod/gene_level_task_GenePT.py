import os
import pickle

# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
from mod.utils import *

# with open(f"./data/embeddings/GPT_3_5_gene_embeddings_fromGenePT.pickle", "rb") as fp:
#     GPT_3_5_gene_embeddings = pickle.load(fp)

### Use text-embedding-3-large generated embeddings
# with open(f"/afs/crc.nd.edu/user/d/dgan/TransferGPT/data_embedding/GPT_3_large_human_embeddings_filtered.pickle", "rb") as fp:
#     GPT_3_5_gene_embeddings = pickle.load(fp)

with open(f"./data/embeddings/GenePT_gene_embedding_ada_text_fromGenePT.pickle", "rb") as fp:
     GPT_3_5_gene_embeddings = pickle.load(fp)

random_states = list(range(41, 51)) # set up the random seeds
do_pca = False
do_cv = True

## Long- vs short- range TFs
# The input data used here are downloaded from Chen et al. (2020) 
# (link: https://www-nature-com.stanford.idm.oclc.org/articles/s41467-020-16106-x).
data_dir = "./data/long_vs_shortTF"
save_csv_dir = "./res/2024_1220/GenePT_1536/long_vs_shortTF_auc.csv"

multiple_run_TrainTest(data_dir, save_csv_dir, random_states, GPT_3_5_gene_embeddings, 
                       do_pca, do_cv)

## Dosage sensitive vs insensitive TFs
data_dir = "./data/DosageSensitivity"
save_csv_dir = "./res/2024_1220/GenePT_1536/DosageSensitivity_auc.csv"
multiple_run_TrainTest(data_dir, save_csv_dir, random_states, GPT_3_5_gene_embeddings, 
                       do_pca, do_cv)

## Bivalent vs. lys4
data_dir = "./data/MethylationState/bivalent_vs_lys4"
save_csv_dir = "./res/2024_1220/GenePT_1536/bivalent_vs_lys4_auc.csv"
multiple_run_TrainTest(data_dir, save_csv_dir, random_states, GPT_3_5_gene_embeddings, 
                       do_pca, do_cv)

## Bivalent vs. no methyl
data_dir = "./data/MethylationState/bivalent_vs_no_methyl"
save_csv_dir = "./res/2024_1220/GenePT_1536/bivalent_vs_no_methyl_auc.csv"
multiple_run_TrainTest(data_dir, save_csv_dir, random_states, GPT_3_5_gene_embeddings, 
                       do_pca, do_cv)