import os
from time import time
from tqdm import tqdm
import pickle
import numpy as np
from lxml import etree
from langchain.docstore.document import Document
import faiss
from langchain_community.vectorstores import FAISS
import openai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
client = OpenAI()
from Datasets import BioASQ

openai.api_key = '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'  # shahin's key

"""
Note: this code does NOT embed the paper title nor publication date, as is done in Yufei's RAG.py code, 
which incorporates these in a "metadata" dict: 

from langchain.docstore.document import Document
docs = []
for record in data:
    metadata = {"year": record.get("pub_date").get('year'),
                "month": record.get("pub_date").get('month'),
                "day": record.get("pub_date").get('day'),
                "title": record.get("article_title")}
    docs.append(Document(page_content=record.get('article_abstract'), metadata=metadata))


RAG.py creates a list of records written as: 
Document(page_content='the abstract text goes here', 
         metadata={'url' : 'http://www.ncbi.nlm.nih.gov/pubmed/34687634', 
                   'title': 'title text goes here'})
            metadata = {"year": record.get("pub_date").get('year'),
                        "month": record.get("pub_date").get('month'),
                        "day": record.get("pub_date").get('day'),
                        "title": record.get("article_title")}


Embedding models available from OpenAI API:

MODEL	                ~ PAGES PER DOLLAR	PERFORMANCE ON MTEB EVAL	MAX INPUT
text-embedding-3-small	62,500	            62.3%	                    8191
text-embedding-3-large	9,615               64.6%	                    8191
text-embedding-ada-002	12,500              61.0%	                    8191
"""

# SELECTED_EMBEDDING_MODEL = 'text-embedding-ada-002'  # medium cost, but worst performance on MTEB ??
# SELECTED_EMBEDDING_MODEL = 'text-embedding-3-large' most expensive, best performance on MTEB.
# SELECTED_EMBEDDING_MODEL = 'text-embedding-3-small'  # cheapest, but medium performance. on MTEB.

# SELECTED_EMBEDDING_MODEL = 'e5'  # Supplied via HuggingFace

SELECTED_EMBEDDING_MODEL = 'faiss.IndexFlatL2'


def _extract_docs_from_xml(file_path):
    """
    Extract abstract text only from the unzipped xml file(s) downloaded from
    https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    Args:
        file_path:
    Returns:
    """
    # Parse XML
    with open(file_path, 'r', encoding='utf-8') as file:  # Assuming UTF-8 encoding
        tree = etree.parse(file)
        root = tree.getroot()

    docs = []

    for record in root.findall('.//PubmedArticle'):
        abstract = record.findtext('.//AbstractText')
        if abstract in [None, '']:  # Don't include records that have no abstract.
            continue
        else:
            pub_date = record.find('.//PubDate')
            year = pub_date.findtext('Year') if pub_date is not None else None
            month = pub_date.findtext('Month') if pub_date is not None else None
            day = pub_date.findtext('Day') if pub_date is not None else None
            title = record.findtext('.//ArticleTitle')
            metadata = {"year": year, "month": month, "day": day, "title": title}
            abstract = abstract.replace('\n', ' ')
            docs.append(Document(page_content=abstract, metadata=metadata))
    return docs
        # for abstract_text in root.findall('.//AbstractText'):
        #     if abstract_text.text:
        #         abstracts.append(abstract_text.text.strip())
        # return abstracts


def _get_index_embeddings():
    """
    Copy-pasted from Yufei's RAG.py
    Returns:
    """
    if SELECTED_EMBEDDING_MODEL == 'e5':
        index = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-unsupervised",
            # model_kwargs={'device': 'cuda'},
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    elif SELECTED_EMBEDDING_MODEL == 'faiss.IndexFlatL2':
        dimension = 1536  # (For e.g. BERT-like embeddings typically are 768)
        # "By default, the length of the embedding vector will be 1536 for text-embedding-3-small
        # or 3072 for text-embedding-3-large"""
        index = faiss.IndexFlatL2(dimension)
    else:
        index = OpenAIEmbeddings()
    return index


#  Different method that builds vectorstore by FAISS.from_documents(chunked_abstract, embeddings)
def _embed_docs(chunked_docs, file_name):
    """
    Args:
        chunked_docs: Text, expected to be extracted from PubMed abstracts.
        file_name: Name of embedding to either read if already exists, or write if not.
    Returns: Embedding
    """
    embeddings_array = None
    dataset_dir = '../dataset/PubMed_Embeddings'
    file_path = os.path.join(dataset_dir, file_name.rstrip('.xml'))
    binary_file = f'{file_path}.npy'

    if os.path.isfile(binary_file):  # if this npy embedding already exists, just read and return it.
        loaded_array = np.load(binary_file)
        return loaded_array
    else:
        st = time()
        embeddings_list = []
        for chunked_doc in tqdm(chunked_docs):
            response = client.embeddings.create(input=chunked_doc, model=SELECTED_EMBEDDING_MODEL)
            embeddings_list.append(response.data[0].embedding)
            embeddings_array = np.array(embeddings_list)
            np.save(binary_file, embeddings_array)
        print(f'Time taken to embed {file_name} dataset = {round(time() - st, 4)} secs.')
    return embeddings_array


def _chunk_docs(docs):
    """
    Split given docs to chunks.
    Copy-pasted from Yufei's RAG.py
    Args:
        docs:
    Returns:
    """
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    return chunks


def _embed_docs_into_index(faiss_index, chunked_docs):
    faiss_index = FAISS.from_documents(documents=chunked_docs, embedding=faiss_index)
    return faiss_index


def _save_indexed_pubmed_to_file(faiss_index, total_num_of_docs):
    dst_dir = f'../FAISS/faissindexflatL2/{total_num_of_docs}docs'
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    dst = os.path.join(dst_dir, 'index.faiss')
    faiss.write_index(faiss_index, dst)
    FAISS


def index_pubmed_xml_abstracts_to_faiss(src_dir_pubmed_xml):
    """
    Read all PubMed XML files from given source directory.
    Extract abstracts text from each file.
    Embed abstracts in selected OpenAI embedding model (selected at top of script).
    Add the list of embeddings to the Faiss index.
    Args:
        src_dir_pubmed_xml: Source raw PMC xml files to read in and embed in openai FAISS.
        dest_dir_pubmed_faiss_index: Name of .faiss file (of indexed PubMed data) to write.
    Returns:
    """
    # Init FAISS index; adjust dimensionality as needed
    # dimension = 1536  # (For e.g. BERT-like embeddings typically are 768)
    # "By default, the length of the embedding vector will be 1536 for text-embedding-3-small
    # or 3072 for text-embedding-3-large"""
    # index = faiss.IndexFlatL2(dimension)

    # Process each XML file in given directory
    total_num_of_docs, num_of_docs = 0, 0
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)

    for root_dir_path, dir_names, filenames in os.walk(src_dir_pubmed_xml):
        for file_name in filenames:
            if file_name.endswith('.xml'):
                file_path = os.path.join(root_dir_path, file_name)
                docs = _extract_docs_from_xml(file_path)
                num_of_docs = len(docs)
                print(f'There are {num_of_docs} articles (abstracts) in this file: {file_name}.')
                chunked_docs = _chunk_docs(docs)
                if chunked_docs:
                    docs_embeddings = _embed_docs(chunked_docs, file_name)
                    docs_embeddings = docs_embeddings.astype('float32')
                    index.add(docs_embeddings)  # Add embeddings array to FAISS index
            total_num_of_docs += num_of_docs
        _save_indexed_pubmed_to_file(index, total_num_of_docs)

"""
if __name__ == '__main__':

    dst_dir = f'../FAISS/{SELECTED_EMBEDDING_MODEL[-7:]}'
    start = time()
    index_pubmed_xml_abstracts_to_faiss(src_dir_pubmed_xml='../dataset/PubMed_XML')
    print(f'Time taken to index these PubMed abstracts was {round((time() - start) / 60, 2)}  minutes')
"""

# # Not used, but may be useful:
# def _get_num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Count number of tokens in given text."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# """
if __name__ == '__main__':
    # Load index (vectorstore) from file
    total_num_of_docs = 15401
    dst_dir = f'../FAISS/faissindexflatL2/{total_num_of_docs}docs'
    # vectorstore = FAISS.load_local(dst_dir, OpenAIEmbeddings())
    vectorstore = faiss.read_index(f'{dst_dir}/index.faiss')

    data = BioASQ(['../dataset/Task11BGoldenEnriched/11B1_golden.json',
                   '../dataset/Task11BGoldenEnriched/11B2_golden.json',
                   '../dataset/Task11BGoldenEnriched/11B3_golden.json',
                   '../dataset/Task11BGoldenEnriched/11B4_golden.json'])
    yesno_questions = data.get_type_questions('yesno')
    matching_ratio = []

    for question in yesno_questions:
        retrieved_docs = vectorstore.similarity_search(question.question_body, k=len(question.docs))
        # k = 4  # Number of nearest neighbors to find
        # xq = question.question_body
        # D, I = vectorstore.search(xq, k=len(question.docs))
        # for i, query_vec in enumerate(xq):
        #     print(f'Query {i}:')
        #     for j, doc_idx in enumerate(I[i]):
        #         print(f'  {j + 1}: Document {doc_idx}, Distance: {D[i][j]}')

        print(question.question_body)
        print(question.docs)
        count = 0
        for doc in retrieved_docs:
            print(doc.metadata['url'])
            if doc.metadata['url'] in question.docs:
                count += 1
        print(count / len(retrieved_docs))
        matching_ratio.append(count / len(retrieved_docs))
    print("\nMatching Ratio (Retrieved URL's Percentage in Golden Refs:", np.mean(matching_ratio))
# """
