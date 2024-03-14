import os
from time import time
from tqdm import tqdm
from lxml import etree
import faiss
import numpy as np
import openai
import tiktoken
from openai import OpenAI
client = OpenAI()

openai.api_key = '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'  # shahin


"""
MODEL	                ~ PAGES PER DOLLAR	PERFORMANCE ON MTEB EVAL	MAX INPUT
text-embedding-3-small	62,500	            62.3%	                    8191
text-embedding-3-large	9,615               64.6%	                    8191
text-embedding-ada-002	12,500              61.0%	                    8191
"""
selected_embedding_model = 'text-embedding-3-small'


def _extract_abstracts_from_xml(file_path):
    """
    Extract abstract text only from the unzipped xml file(s) downloaded from
    https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    Args:
        file_path:
    Returns:

    """
    # Parse the XML file
    tree = etree.parse(file_path)
    root = tree.getroot()
    # Adjust the namespace dictionary based on your XML structure if needed
    # ns = {'pm': 'http://www.ncbi.nlm.nih.gov/pubmed'}
    # ns = {'pm': 'http://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd'}
    # Extract abstracts
    abstracts = []
    # for abstract_text in root.findall('.//pm:AbstractText', namespaces=ns):
    for abstract_text in root.findall('.//AbstractText'):
        if abstract_text.text:
            abstracts.append(abstract_text.text.strip())
    return abstracts


def _get_num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def _embed_abstracts(abstracts, file_name):
    """
    Args:
        abstracts: Text, expected to be extracted from PubMed abstracts.
        file_name: Name of embedding to either read if already exists, or write if not.
    Returns: Embedding
    """
    dataset_dir = '../dataset/PubMed_Embeddings'
    file_path = os.path.join(dataset_dir, file_name.rstrip('.xml'))
    binary_file = f'{file_path}.npy'
    if os.path.isfile(binary_file):
        loaded_array = np.load(binary_file)
        return loaded_array
    else:
        st = time()
        # Batch process abstracts for efficiency
        embeddings_list = []
        for abstract in tqdm(abstracts):
            # response = openai.Embedding.create(input=abstract, engine="text-embedding-ada-002")
            # response = openai.Embedding.create(input=abstract, engine='text-embedding-3-small')
            print('here')
            response = client.embeddings.create(input=abstract, model=selected_embedding_model)  # less expensive but small
            embeddings_list.append(response.data[0].embedding)
        print(f'Time taken to embed {file_name} dataset = {round(time() - st,4 )} secs.')
        embeddings_array = np.array(embeddings_list)
        np.save(binary_file, embeddings_array)
    return


# Not sure if I should be replacing newline with white space ?
def get_embedding(text):
    """Note: This function is not used at present"""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=selected_embedding_model).data[0].embedding


def index_pubmed_xml_abstracts_to_faiss(pmc_xml_dir_to_read_in, pubmed_faiss_index_file_to_write):
    """
    Read all PubMed XML files from given source directory.
    Extract abstracts text from each file.
    Embed abstracts in selected OpenAI embedding model (selected at top of script).
    Add the list of embeddings to the Faiss index.
    Args:
        pmc_xml_dir_to_read_in: Source raw PMC xml files to read in and embed in openai FAISS.
        pubmed_faiss_index_file_to_write: Name of .faiss file (of indexed PubMed data) to write.
    Returns:

    """
    # Init FAISS index; adjust dimensionality as needed
    dimension = 1536  # BERT-like embeddings typically are 768
    # "By default, the length of the embedding vector will be 1536 for text-embedding-3-small or 3072 for
    # text-embedding-3-large"""
    index = faiss.IndexFlatL2(dimension)

    # Process each XML file in the directory
    for root_dir_path, dir_names, filenames in os.walk(pmc_xml_dir_to_read_in):
        for file_name in filenames:
            if file_name.endswith('.xml'):
                file_path = os.path.join(root_dir_path, file_name)
                abstracts = _extract_abstracts_from_xml(file_path)
                print(f'There are {len(abstracts)} abstracts in this file')
                if abstracts:  # If there are abstracts extracted
                    embeddings = _embed_abstracts(abstracts, file_name)
                    embeddings = embeddings.astype('float32')
                    index.add(embeddings)  # Add embeddings to FAISS index

    # Save the FAISS index
    faiss.write_index(index, pubmed_faiss_index_file_to_write)


if __name__ == '__main__':

    start = time()
    index_pubmed_xml_abstracts_to_faiss(pmc_xml_dir_to_read_in='../dataset/PUBMED',
                                        pubmed_faiss_index_file_to_write='../FAISS/faiss_index_PMC6388086.faiss')
    print(f'Time taken to index these PubMed abstracts was {round((time() - start) / 60, 2)}  minutes')
