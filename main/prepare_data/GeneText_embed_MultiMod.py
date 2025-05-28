import os
import sys
import json
import pickle
sys.path.append('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
# sys.path.append('/Users/david/Desktop/FinetuneEmbed')

os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
from mod.multi_mod import *
print(os.getcwd())

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

model_dim = [768, 384, 384, 768, 384, 384, 384, 384, 384, 384, 384] # the dimension of each model

# load gene name list
with open('./data/embeddings/gene_name_list.pickle', 'rb') as f:
    loaded_list = pickle.load(f)

# load gene ncbi descriptions
with open("./data/gene_text/hs_ncbi_gene_text.json", "r") as file:
    gene_descriptions = json.load(file)

overlap_genes = list(set(loaded_list) & set(gene_descriptions.keys())) # find the intersected genes

# Subset the dictionary
gene_text_dict = {k: gene_descriptions[k] for k in overlap_genes if k in gene_descriptions}

for i in range(len(model_names)):
    print(f"Processing model: {model_names[i]}")
    # Get embeddings
    embedding = get_embeddings(model_name=model_names[i], 
                                    gene_text_dict=gene_text_dict,
                                    model_dim=model_dim[i])
    
    # Save
    with open(f'./data/embeddings/GeneText2_{save_mod_names[i]}_embed.pickle', 'wb') as f:
        pickle.dump(embedding, f)


