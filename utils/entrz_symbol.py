import requests
import xml.etree.ElementTree as ET
import mygene

def my_gene_get_gene_symbol(ensp_ids):
    mg = mygene.MyGeneInfo()
    symbols = mg.querymany(ensp_ids, species='human')
    ensp2symbol = {x['query']:x['symbol'] for x in symbols if 'symbol' in x}
    return ensp2symbol

def ensembl_gene_symbol(ensembl_id):
    # Ensembl REST API endpoint for gene lookup
    url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}"

    # Headers to specify that we want the response in JSON format
    headers = {'Content-Type': 'application/json'}

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # Extract the gene symbol
        gene_symbol = data.get('display_name', 'Not found')
        return gene_symbol
    else:
        print("Failed to retrieve data")
        return None

import requests
import json

def batch_convert_ensembl_to_symbols(ensembl_ids):
    # Ensembl REST API endpoint for bulk lookup for ENSG IDs
    url = "https://rest.ensembl.org/lookup/id"

    # Prepare headers and data for POST request
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = json.dumps({'ids': ensembl_ids})

    # Make the POST request
    response = requests.post(url, headers=headers, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        response_data = response.json()
        gene_symbols = {}
        for ensembl_id, gene_data in response_data.items():
            # Extract the gene symbol
            if gene_data is None:
                gene_symbols[ensembl_id] = 'Not found'
                continue
            gene_symbols[ensembl_id] = gene_data.get('display_name', 'Not found')
        return gene_symbols
    else:
        print("Failed to retrieve data")
        return None


def get_gene_id_from_protein(ensp_id):
    """
    Get the gene ID associated with an Ensembl protein ID.
    """
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{ensp_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)
    
    if response.ok:
        decoded = response.json()
        return decoded.get("Parent")
    return None

def get_gene_symbol_from_gene_id(gene_id):
    """
    Get the gene symbol from a gene ID.
    """
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{gene_id}?expand=0"
    headers = {"Content-Type": "application/json"}
    response = requests.get(server + ext, headers=headers)

    if response.ok:
        decoded = response.json()
        return decoded.get("display_name")
    return None

def convert_ensp_to_gene_symbol(ensp_ids):
    """
    Convert a list of Ensembl protein IDs (ENSP) to gene symbols using Ensembl REST API.
    """

    result = {}
    for ensp_id in ensp_ids:
        gene_id = get_gene_id_from_protein(ensp_id)
        if gene_id:
            gene_symbol = get_gene_symbol_from_gene_id(gene_id)
            if gene_symbol:
                result[ensp_id] = gene_symbol
            else:
                result[ensp_id] = "Gene symbol not found"
        else:
            result[ensp_id] = "Gene ID not found"
    
    return result



def get_gene_info(entrez_id):
    # Base URL for the EFetch utility
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Parameters for the request
    params = {
        'db': 'gene',        # Database
        'id': entrez_id,     # Entrez Gene ID
        'retmode': 'xml'     # Return mode
    }

    # Making the HTTP request
    response = requests.get(efetch_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the XML response
        root = ET.fromstring(response.text)
        
        # Trying different tags to find the gene symbol
        gene_symbol_tags = [
            ".//Gene-ref_locus", 
            ".//Gene-ref_locus-tag", 
            ".//Entrezgene_gene/Gene-ref/Gene-ref_locus"
        ]

        gene_symbol = None
        for tag in gene_symbol_tags:
            gene_symbol_element = root.find(tag)
            if gene_symbol_element is not None:
                gene_symbol = gene_symbol_element.text
                break

        if gene_symbol is None:
            gene_symbol = "Not found"

        return  gene_symbol
    else:
        print("Failed to retrieve data")
        return None, None

