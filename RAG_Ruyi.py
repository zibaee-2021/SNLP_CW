# Download all necessary packages
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import json
import os
from Datasets import BioASQ

# OpenAI key
os.environ["OPENAI_API_KEY"] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to load PubMed JSON documents and convert them into Document objects
def load_pubmed_json_docs(pubmed_json_path):
    docs = []
    # Load JSON file
    with open(pubmed_json_path, encoding='UTF-8') as file:
        data = json.load(file)
        # Iterate through 'pages'
        for record in data:
            metadata = {"year": record.get("pub_date").get('year'),
                        "month": record.get("pub_date").get('month'),
                        "day": record.get("pub_date").get('day'),
                        "title": record.get("article_title")}
            # Create Document object and append to list
            docs.append(Document(page_content=record.get('article_abstract'), metadata=metadata))
    return docs

# Function to load retrieved JSON documents and convert them into Document objects
def load_retrieved_json_docs(retrieved_json_path):
    docs = []
    # Load JSON file
    with open(retrieved_json_path, encoding='UTF-8') as file:
        data = json.load(file)
        # Iterate through 'pages'
        for record in data:
            metadata = {"url": record.get("url"),
                        "title": record.get("title")}
            # Create Document object and append to list
            docs.append(Document(page_content=record.get('abstract'), metadata=metadata))
    return docs

# Function to load documents based on their name and file path
def load_docs(document_name, document_file_path):
    # If PubMed, use pubmed path
    if document_name[:6] == 'PubMed':
        docs = load_pubmed_json_docs(document_file_path)
    # If BioASQ, use bioasq path
    elif document_name[:6] == 'BioASQ':
        docs = load_retrieved_json_docs(document_file_path)
    else:
        raise NotImplementedError
    return docs

# Function to get embeddings based on the specified embedding model
def get_embeddings(embedding_model):
    # Use e5 model for RAG
    if embedding_model == 'e5':  # If the specified embedding model is 'e5'
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-unsupervised", # e5 model name
            model_kwargs={'device': 'cpu'},  # Run with CPU
            encode_kwargs={'normalize_embeddings': False}  # Do not normalize embeddings
        )
    # If the specified embedding model is 'OpenAI'
    elif embedding_model == 'OpenAI':
        embeddings = OpenAIEmbeddings() # Initialize OpenAIEmbeddings
    else:
        raise NotImplementedError
    return embeddings

# Function to save a FAISS database and its chunks based on the provided documents and embedding model
def save_faiss_database(documents, document_file_path, embedding_model):
    # Load Document from File Path
    data = load_docs(documents, document_file_path)
    # Split Documents using TokenTextSplitter to chunks
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    # Get the Embeddings Model
    embeddings = get_embeddings(embedding_model)
    # Setup database
    database = FAISS.from_documents(chunks, embeddings)
    database.save_local(f"FAISS/faiss_index_{documents}_{embedding_model}")
    return database

# Function to load a FAISS database based on the specified documents and embedding model
def load_faiss_database(documents, embedding_model):
    # Get embeddings based on the specified embedding model
    embeddings = get_embeddings(embedding_model)
    # Load database from file
    database = FAISS.load_local(f"FAISS/faiss_index_{documents}_{embedding_model}", embeddings)
    return database


if __name__ == '__main__':
    # Flag indicating whether to load an existing FAISS database or create a new one
    load = True
    if load:
        # Load FAISS database using specified documents and embedding model
        retrieval_model = load_faiss_database(documents='BioASQ_11B_test',
                                              embedding_model='OpenAI')
    else:
        # Create new FAISS database and retrieve chunks
        retrieval_model = save_faiss_database(documents='BioASQ_11B_test',
                                              document_file_path='BioASQ_11B_test_yesno.json',
                                              embedding_model='OpenAI')

    # ------------------------------------------------------------------------------------------------------------------

    # Load BioASQ dataset
    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])

    # Retrieve Yes or No questions from the dataset
    yesno_questions = data.get_type_questions('yesno')

    matching_ratio = []
    # Iterate through each Yes or No question
    for question in yesno_questions:
        # Retrieve documents related to the question from the FAISS database
        retrieved_docs = retrieval_model.similarity_search(question.question_body, k=len(question.docs))
        print(question.question_body)
        print(question.docs)
        # Initialize count
        count = 0
        # Check how many retrieved documents match the golden references
        for doc in retrieved_docs:
            print(doc.metadata['url'])
            # If matched, add one in count
            if doc.metadata['url'] in question.docs:
                count += 1

        # Calculate the matching ratio and append to the list
        print(count / len(retrieved_docs))
        matching_ratio.append(count / len(retrieved_docs))
    # Print the average matching ratio
    print("\nMatching Ratio (Retrieved URL's Percentage in Golden Refs:", np.mean(matching_ratio))
