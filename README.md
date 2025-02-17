# Small, Open-Source Text-Embedding Models as Substitutes to OpenAI Models for Gene Analysis
## Introduction
While foundation transformer-based models developed for gene expression data analysis can be costly to train and operate, a recent approach known as GenePT offers a low-cost and highly efficient alternative. GenePT utilizes OpenAI’s text-embedding function to encode background information, which is in textual form, about genes. However, the closed-source, online nature of OpenAI’s text-embedding service raises concerns regarding data privacy, among other issues. In this paper, we explore the possibility of replacing OpenAI’s models with open-source transformer-based text-embedding models. We identified ten models from Hugging Face that are small in size, easy to install, and light in compu- tation. Across all four gene classification tasks we considered, some of these models have outperformed OpenAI’s, demonstrating their potential as viable, or even superior, alternatives. Additionally, we find that fine-tuning these models often does not lead to significant improvements in performance.

## Explanations of the foldes
* `./main`: this folder contains code to perform the analysis.
    * `./get_ncbi_gene_text`: this folder provides code to extract and organize text information from the NCBI database.
    * `./prepare_data:` this folder contains code to split the four classification tasks into train, eval, and text data ten times. The links to get the original data were presented in the code. The text embeddings from GenePT were obtained from its Zenodo website.
    * `./TrainEvalTest_text_MultiMod`: this folder contains different approaches to perform the classifications.
        * `./multimod`: contains code to perform the classification with or without cross-validation using logistic regression and random forest for each of the 11 (ten small-LLMs & GenePT) embedding methods.
        * `./finetune_default`: contains code to perform the classification by fine-tuning model parameters for each of the ten small-LLM models.
        * `./finetune_para_sensitivity`: contains code to perform the classification on the Sensitivity data with the model `GIST-small-Embedding-v0` using different sets of initial hyperparameters.
* `./mod`: this folder contains necessary functions to perform the analysis.

## MTEB ranking
The ranks of the small-LLMs on MTEB can be found in the leader board website [https://huggingface.co/spaces/mteb/leaderboard_legacy](https://huggingface.co/spaces/mteb/leaderboard_legacy). The ranks we provided in the paper were extracted on Dec. 09th, 2024. The ranks are always updating on the website. Please note that the website we used has `_legacy` in it. The most recent one does not have that.

## Citation
If you find our work helpful, please cite the following paper:

Gan, D. & Li, J. (2025). Small, Open-Source Text-Embedding Models as Substitutes to OpenAI Models for Gene Analysis.