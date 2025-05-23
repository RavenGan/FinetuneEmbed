import numpy as np
import os
import sys
import pickle
from dotenv import load_dotenv
sys.path.append('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
# sys.path.append('/Users/david/Desktop/FinetuneEmbed')

from tqdm import tqdm
from openai import OpenAI
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
print(os.getcwd())

# set up the OpenAI API key CONFIDENTIAL
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

with open('./data/embeddings/gene_name_list.pickle', 'rb') as f:
    loaded_list = pickle.load(f)

# initiate openai client
client = OpenAI(api_key=openai_api_key)

# use the embedding model text-embedding-ada-002
def get_gpt_embedding(client, text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

gpt_gene_name_to_embedding_clean_text = {}
GPT_DIM = 3072 # fix GPT embeddings
for name in tqdm(loaded_list):
    if name not in gpt_gene_name_to_embedding_clean_text:
        # print('key', key)
        if name == '': 
            # if the dictionary does not have information about a gene
            gpt_gene_name_to_embedding_clean_text[name] = np.zeros(GPT_DIM) # it's hard coded to be 0
        else:
            gpt_gene_name_to_embedding_clean_text[name] = get_gpt_embedding(client, name) 

# Save
with open('./data/embeddings/GeneName_embed.pickle', 'wb') as f:
    pickle.dump(gpt_gene_name_to_embedding_clean_text, f)
