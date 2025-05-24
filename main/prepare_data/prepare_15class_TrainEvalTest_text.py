import os
import sys
# sys.path.append('/Users/david/Desktop/FinetuneEmbed')
sys.path.append('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
import json
import pickle
import pandas as pd

# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
# load necessary functions
from mod.utils import *

# load gene name
### 15 class genes----
gene_class = pd.read_csv('./data/Multi_class/gene_info_table_15classes.csv')
gene_class_ls = list(gene_class['gene_name'])

# load gene ncbi descriptions
with open("./data/gene_text/hs_ncbi_gene_text.json", "r") as file:
    gene_descriptions = json.load(file)

with open(f"./data/embeddings/GPT_3_5_gene_embeddings_fromGenePT.pickle", "rb") as fp:
    GPT_3_5_gene_embeddings = pickle.load(fp)

random_states = list(range(41, 51)) # set up the random seeds
eva_size = 0.1 # validation data proportions
test_size = 0.1 # test data proportions

overlap_genes = list(set(gene_class_ls) & set(gene_descriptions.keys())) # find the intersected genes
overlap_genes = list(set(overlap_genes) & set(GPT_3_5_gene_embeddings.keys()))

gene_class_sub = gene_class[gene_class['gene_name'].isin(overlap_genes)]

genes = list(gene_class_sub['gene_name'])
labels = list(gene_class_sub['gene_type'])
save_dir = "./data/Multi_class/TrainEvalTestData"
# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Split the data multiple times
for random_state in random_states:
    TrainEvalTest_split(genes, labels, gene_descriptions, save_dir, 
                        test_size=test_size, eval_size=eva_size, random_state=random_state)
    
