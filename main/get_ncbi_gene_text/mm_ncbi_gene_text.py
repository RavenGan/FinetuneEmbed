import requests
import pandas as pd
import html2text
import mygene
import json
import os

from tqdm import tqdm
from bs4 import BeautifulSoup
# import pickle

# os.chdir('/Users/david/Desktop/TransferGPT')
os.chdir('/afs/crc.nd.edu/user/d/dgan/TransferGPT')
print(os.getcwd())

mg = mygene.MyGeneInfo()


parts_to_remove = [
    "##  Summary\n",
    "NEW",
    'Try the newGene table',
    'Try the newTranscript table',
    '**',
    "\nGo to the top of the page Help\n"
]

def rough_text_from_gene_name(gene_number):
    
    # get url
    url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_number}"
    # Send a GET request to the URL
    summary_text = ''
    soup = None
    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.Timeout:
        print('time out')
        return((summary_text,soup))
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the "summary" tab content by inspecting the page's structure
        summary_tab = soup.find('div', {'class': 'rprt-section gene-summary'})

        # Check if the "summary" tab content is found
        if summary_tab:
            # Convert the HTML to plain text using html2text
            html_to_text = html2text.HTML2Text()
            html_to_text.ignore_links = True  # Ignore hyperlinks

            # Extract the plain text from the "summary" tab
            summary_text = html_to_text.handle(str(summary_tab))
            # Remove the specified parts from the original text
            for part in parts_to_remove:
                summary_text = summary_text.replace(part, ' ')
                # Replace '\n' with a space
            summary_text = summary_text.replace('\n', ' ')

            # Reduce multiple spaces into one space
            summary_text = ' '.join(summary_text.split())
            # Print or save the extracted text
        else:
            print("Summary tab not found on the page.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return((summary_text,soup))

gene_df = pd.read_csv('./data_embedding/mm_unique_genes.csv')
gene_ls = [gene for gene in gene_df['mm_genes']]

gene_queries = mg.querymany(gene_ls, scopes='symbol', species='mouse') # extract mouse gene info

gene_name_to_tax_id = {}
for result in gene_queries:
    if "_id" in result and "query" in result:
        gene_name_to_tax_id[result['symbol']] = result['_id']

gene_name_to_summary_page = {}

for gene_name, page_id in tqdm(sorted(gene_name_to_tax_id.items())):
    if gene_name not in gene_name_to_summary_page:
        # print('gene_name',gene_name)
        parsed_text, unparsed_html = rough_text_from_gene_name(page_id)
        gene_name_to_summary_page[gene_name] = parsed_text

# save the results
# Specify the filename
filename = './data_embedding/mm_ncbi_gene_text.json'
# Writing JSON data
with open(filename, 'w') as f:
    json.dump(gene_name_to_summary_page, f, indent=4)