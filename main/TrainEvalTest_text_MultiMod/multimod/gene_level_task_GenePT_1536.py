import os
import sys
import pickle
import argparse
# sys.path.append('/Users/david/Desktop/FinetuneEmbed')
sys.path.append('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')

# Specify the working directory
# os.chdir('/Users/david/Desktop/FinetuneEmbed')
os.chdir('/afs/crc.nd.edu/group/StatDataMine/dm011/Dailin_Gan/FinetuneEmbed')
from mod.utils import *

# Set up argument parser
parser = argparse.ArgumentParser(description="Embedding evaluation pipeline")

parser.add_argument("--task", type=str, default="long_vs_shortTF", help="Classification task name")
parser.add_argument("--embedding_data", type=str, default="TrainEvalTestData", help="Directory for embedding data")
parser.add_argument("--embedding_type", type=str, default="text_embedding", help="Embedding type")
parser.add_argument("--do_cv", action="store_true", help="Whether to run cross-validation")
parser.add_argument("--do_pca", action="store_true", help="Whether to apply PCA")
parser.add_argument("--n_PCs", type=int, default=20, help="Number of principal components")
parser.add_argument("--save_root", type=str, default="./res/2025_0526", help="Directory to save results")

args = parser.parse_args()

random_states = list(range(41, 51)) # set up the random seeds

# load the gene embedding dictionary
with open(f"./data/embeddings/GenePT_gene_embedding_ada_text_fromGenePT.pickle", "rb") as fp:
     embedding_dict = pickle.load(fp)

task = args.task
embedding_data = args.embedding_data
embedding_type = args.embedding_type
do_cv = args.do_cv
do_pca = args.do_pca
n_PCs = args.n_PCs
save_root = args.save_root

data_dir = f"./data/{task}/{embedding_data}"

if do_cv==True and do_pca==True:
     folder_name = f"PCA_CV_{embedding_type}"
elif do_cv==True and do_pca==False:
     folder_name = f"PCA_NoCV_{embedding_type}"
elif do_cv==False and do_pca==True:
     folder_name = f"NoPCA_CV_{embedding_type}"
elif do_cv==False and do_pca==False:
     folder_name = f"NoPCA_NoCV_{embedding_type}"


model_name = "GenePT_1536"
save_mod_name = "GenePT_1536"

save_csv_dir = f"{save_root}/{folder_name}/" + save_mod_name + f"_{task}_NumRes.csv"
os.makedirs(os.path.dirname(save_csv_dir), exist_ok=True)
ROC_save_dir = f"{save_root}/{folder_name}/{task}/" + save_mod_name + "/"

multiple_run_TrainTest(data_dir, save_csv_dir, random_states, 
                       embedding_dict, ROC_save_dir, do_cv, do_pca, n_PCs)
