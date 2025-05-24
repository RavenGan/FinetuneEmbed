import os
import sys
# sys.path.append('/Users/david/Desktop/FinetuneEmbed')
sys.path.append('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
import json
import pickle
import mygene
import pandas as pd
# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
# load necessary functions
from mod.utils import *

# load the gene name embeddings
with open(f"./data/embeddings/GeneName_embed.pickle", "rb") as fp:
    gene_name_embeddings = pickle.load(fp)

random_states = list(range(41, 51)) # set up the random seeds
eva_size = 0.1 # validation data proportions
test_size = 0.1 # test data proportions

### 15 class genes----------------------------------------------------------------
gene_class = pd.read_csv('./data/Multi_class/gene_info_table_15classes.csv')
gene_class_ls = list(gene_class['gene_name'])

overlap_genes = list(set(gene_class_ls) & set(gene_name_embeddings.keys())) # find the intersected genes
gene_class_sub = gene_class[gene_class['gene_name'].isin(overlap_genes)]

genes = list(gene_class_sub['gene_name'])
labels = list(gene_class_sub['gene_type'])
save_dir = "./data/Multi_class/TrainEvalTestData_Name"
# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Split the data multiple times
for random_state in random_states:
    TrainEvalTest_Name_split(genes, labels, save_dir, 
                        test_size=test_size, eval_size=eva_size, random_state=random_state)


### Long- vs short- range TFs----------------------------------------------------------------
# The input data used here are downloaded from the geneformer paper Hugging Face website 
# (link: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification).
with open("./data/long_vs_shortTF/example_input_files_gene_classification_tf_regulatory_range_tf_regulatory_range.pickle", "rb") as f:
    check_data = pickle.load(f)

long_range_tf_gene = check_data['long_range']
short_range_tf_gene = check_data['short_range']

# convert gene id to gene symbols
mg = mygene.MyGeneInfo()
long_range_query = mg.querymany(long_range_tf_gene, species='human')
short_range_query = mg.querymany(short_range_tf_gene, species='human')
long_range_gene_name = [x['symbol'] for x in long_range_query]
short_range_gene_name = [x['symbol'] for x in short_range_query if 'symbol' in x]

long_range_tf_gene = list(set(long_range_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes
short_range_tf_gene = list(set(short_range_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes

genes = long_range_tf_gene + short_range_tf_gene
# 1 for long-range TF, 0 for short-range TF
labels = [1] * len(long_range_tf_gene) + [0] * len(short_range_tf_gene) 
save_dir = "./data/long_vs_shortTF/TrainEvalTestData_Name"
# Create the directory if it does not exist
os.makedirs(save_dir, exist_ok=True)

# Split the data multiple times
for random_state in random_states:
    TrainEvalTest_Name_split(genes, labels, save_dir, 
                        test_size=test_size, eval_size=eva_size, random_state=random_state)
    

### Dosage sensitive vs insensitive TFs----------------------------------------------------------------
# link_file = "https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/raw/main/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sens_tf_labels.csv"
with open(f"./data/DosageSensitivity/example_input_files_gene_classification_dosage_sensitive_tfs_dosage_sensitivity_TFs.pickle", "rb") as fp:
    dosage_tfs = pickle.load(fp)
sensitive = dosage_tfs["Dosage-sensitive TFs"]
insensitive = dosage_tfs["Dosage-insensitive TFs"]

# convert gene id to gene symbols
mg = mygene.MyGeneInfo()
sensitive_query = mg.querymany(sensitive, species='human')
in_sensitive_query = mg.querymany(insensitive, species='human')
sensitive_gene_name = [x['symbol'] for x in sensitive_query]
insensitive_gene_name = [x['symbol'] for x in in_sensitive_query if 'symbol' in x]

sensitive_gene = list(set(sensitive_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes
insensitive_gene = list(set(insensitive_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes

genes = sensitive_gene + insensitive_gene
 # 1 for sensitive, 0 for insensitive
labels = [1] * len(sensitive_gene) + [0] * len(insensitive_gene)
save_dir = "./data/DosageSensitivity/TrainEvalTestData_Name"

# Split the data multiple times
for random_state in random_states:
    TrainEvalTest_Name_split(genes, labels, save_dir, 
                        test_size=test_size, eval_size=eva_size, random_state=random_state)
    

###  Methylation state prediction---------------------------------------------------------------
# The csv files are downloaded from https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/
## Bivalent vs. lys4
with open(f"./data/MethylationState/example_input_files_gene_classification_bivalent_promoters_bivalent_vs_lys4_only.pickle", "rb") as fp:
    bivalent_vs_lys4 = pickle.load(fp)

bivalent_gene_labels = bivalent_vs_lys4['bivalent']
lysine_gene_labels = bivalent_vs_lys4['lys4_only']

# convert gene id to gene symbols
mg = mygene.MyGeneInfo()
bivalent_query = mg.querymany(bivalent_gene_labels, species='human')
lysine_query = mg.querymany(lysine_gene_labels, species='human')
bivalent_gene_name = [x.get('symbol', '') for x in bivalent_query]
lysine_gene_name = [x.get('symbol', '') for x in lysine_query if 'symbol' in x]

bivalent_gene = list(set(bivalent_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes
lysine_gene = list(set(lysine_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes

genes = bivalent_gene + lysine_gene
# 1 for bivalent_gene, 0 for lysine_gene
labels = [1] * len(bivalent_gene) + [0] * len(lysine_gene)
save_dir = "./data/MethylationState/bivalent_vs_lys4/TrainEvalTestData_Name"

# Split the data multiple times
for random_state in random_states:
    TrainEvalTest_Name_split(genes, labels, save_dir, 
                        test_size=test_size, eval_size=eva_size, random_state=random_state)

## Bivalent vs. no methyl
with open(f"./data/MethylationState/example_input_files_gene_classification_bivalent_promoters_bivalent_vs_no_methyl.pickle", "rb") as fp:
    bivalent_vs_no_methyl = pickle.load(fp)

bivalent_gene_labels = bivalent_vs_no_methyl['bivalent']
no_methylation_gene_labels = bivalent_vs_no_methyl['no_methylation']

# convert gene id to gene symbols
mg = mygene.MyGeneInfo()
bivalent_query = mg.querymany(bivalent_gene_labels, species='human')
no_methylation_query = mg.querymany(no_methylation_gene_labels, species='human')
bivalent_gene_name = [x.get('symbol', '') for x in bivalent_query]
no_methylation_gene_name = [x.get('symbol', '') for x in no_methylation_query if 'symbol' in x]

bivalent_gene = list(set(bivalent_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes
no_methylation_gene = list(set(no_methylation_gene_name) & set(gene_name_embeddings.keys())) # find the intersected genes

genes = bivalent_gene + no_methylation_gene
 # 1 for bivalent_gene, 0 for no_methylation_gene
labels = [1] * len(bivalent_gene) + [0] * len(no_methylation_gene)
save_dir = "./data/MethylationState/bivalent_vs_no_methyl/TrainEvalTestData_Name"

# Split the data multiple times
for random_state in random_states:
    TrainEvalTest_Name_split(genes, labels, save_dir, 
                        test_size=test_size, eval_size=eva_size, random_state=random_state)