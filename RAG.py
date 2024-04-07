# Download all necessary packages
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json
import os
from Datasets import BioASQ

# OpenAI key
# os.environ["OPENAI_API_KEY"] = 'sk-xxx'
os.environ["OPENAI_API_KEY"] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to calculate the maximum number of consecutive identical words between two sentences
def get_consecutive_identical_words(sentence1, sentence2):
    # Split the sentences into words list
    words1 = sentence1.split()
    words2 = sentence2.split()
    # Initialize variables to store the maximum consecutive identical count
    max_consecutive_identical = 0
    consecutive_identical = 0
    # Iterate through the words of both sentences simultaneously
    for word1, word2 in zip(words1, words2):
        # If the words are the same, add one to count and update max consecutive identical if necessary
        if word1 == word2:
            consecutive_identical += 1
            max_consecutive_identical = max(max_consecutive_identical, consecutive_identical)
        else: # If not equal, reset consecutive identical count
            consecutive_identical = 0
    return max_consecutive_identical

# Function to load documents from a list of MedRAG PubMed JSON file paths
def load_medrag_pubmed_json(json_path_list):
    docs = [] # Initialize an empty list to store documents
    # Iterate through each file path in the provided list
    for file_path in json_path_list:
        # Load JSON file
        with open(file_path, 'r', encoding='UTF-8') as file:
            # Iterate through each line in the file
            for line in file:
                # Load JSON data from the line
                data = json.loads(line)
                # Extract metadata
                metadata = {"id": data.get("id"),
                            "title": data.get("title")}
                # Create Document object and append to list
                docs.append(Document(page_content=data.get('content'), metadata=metadata))
    return docs

# Function to load retrieved JSON documents and convert them into Document objects
def load_retrieved_json_docs(retrieved_json_path):
    docs = []
    # Load JSON file
    with open(retrieved_json_path, encoding='UTF-8') as file:
        data = json.load(file)
        # Iterate through 'pages'
        for record in data:
            # Extract metadata
            metadata = {"url": record.get("url"),
                        "title": record.get("title")}
            # Create Document object and append to list
            docs.append(Document(page_content=record.get('abstract'), metadata=metadata))
    return docs

# Function to get embeddings based on the specified embedding model
def get_embeddings(embedding_model):
    # Use e5 model for RAG
    if embedding_model == 'e5': # If the specified embedding model is 'e5'
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-unsupervised", # e5 model name
            model_kwargs={'device': 'cpu'},  # Run with CPU
            encode_kwargs={'normalize_embeddings': False} # Do not normalize embeddings
        )
    # If the specified embedding model is 'OpenAI'
    elif embedding_model == 'OpenAI':
        embeddings = OpenAIEmbeddings() # Initialize OpenAIEmbeddings
    else:
        raise NotImplementedError
    return embeddings

# Function to save a FAISS database and its chunks based on the provided documents and embedding model
def save_faiss_database(documents, file_path_list, embedding_model):
    # Load Document from File Path
    # data = load_docs(documents, document_file_path)
    docs = load_medrag_pubmed_json(file_path_list)
    # Split Documents using TokenTextSplitter to chunks
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
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

# Function to append documents from a JSON file to an existing FAISS database
def append_golden_faiss_databse(database, json_path, embedding_model):
    # Load Document from File Path
    # data = load_docs(documents, document_file_path)
    docs = load_retrieved_json_docs(json_path)
    # Split Documents using TokenTextSplitter to chunks
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    # Get the Embeddings Model
    embeddings = get_embeddings(embedding_model)
    # Setup database
    database.afrom_documents(chunks, embeddings)
    return database


if __name__ == '__main__':
    # List of file paths for PubMed JSON files
    file_path_list = []
    for i in range(1, 11):
        file_path_list.append('refs/pubmed/chunk/pubmed23n{:04d}.jsonl'.format(i))

    # Flag indicating whether to load an existing FAISS database or create a new one
    load = True
    if load:
        # Load FAISS database using specified documents and embedding model
        retrieval_model = load_faiss_database(documents='BioASQ_11B_test',
                                              embedding_model='OpenAI')
    else:
        # Create new FAISS database and retrieve chunks
        retrieval_model = save_faiss_database(documents='BioASQ_11B_test',
                                              file_path_list=file_path_list,
                                              embedding_model='OpenAI')

    # ------------------------------------------------------------------------------------------------------------------
    # Dictionary to store golden references for BioASQ test questions
    BioASQ_test_golden_ref = {}
    # Load golden references from BioASQ JSON file
    with open('refs/BioASQ_11B_test_yesno.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)
        # Populate golden reference dictionary with given URLs and titles
        for record in data:
            BioASQ_test_golden_ref[record.get("url")] = record.get("title")

    # ------------------------------------------------------------------------------------------------------------------

    # Load FAISS databases for MedRAG and BioASQ 11B test
    rag_database = load_faiss_database(documents='MedRAG_133000', embedding_model='OpenAI')
    rag_database_2 = load_faiss_database(documents='BioASQ_11B_test', embedding_model='OpenAI')
    # Merge BioASQ database into MedRAG database
    rag_database.merge_from(rag_database_2)

    # Load BioASQ dataset
    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])
    # Retrieve yes/no questions from the dataset
    yesno_questions = data.get_type_questions('yesno')

    # Lists to store ROUGE scores and document retrieval accuracies
    average_rouge, accuracy_list = [], []
    rouge = Rouge() # Initialize Rouge object for ROUGE scoring

    # Iterate through each Yes or No question
    for question in yesno_questions:
        # Retrieve documents related to the question from the merged FAISS database
        retrieved_docs = rag_database.similarity_search(question.question_body, k=4)
        # Lists to store ROUGE scores for retrieved documents
        rouge_list = []
        # Initialize count
        count = 0
        # Check how many retrieved documents match the golden references
        for doc in retrieved_docs:
            max_rouge_score = 0
            retrieved = False # Flag to indicate if the document is correctly retrieved
            # Iterate through each golden reference for the question
            for golden_ref in question.refs:
                # Compute ROUGE-1 score between the document content and golden reference
                score = rouge.get_scores(doc.page_content, golden_ref['text'])
                # Extract ROUGE-1 F1 score from the computed scores
                rouge_1_score = score[0]['rouge-1']['f']
                # Update the maximum ROUGE-1 score encountered so far
                max_rouge_score = max(rouge_1_score, max_rouge_score)
                # Compare if the titles are identical using consecutive identical word count
                title_retrieved = doc.metadata['title']
                title_golden = BioASQ_test_golden_ref.get(golden_ref['document'], '')
                # If consecutive identical word count is greater than 5, consider retrieve successful
                if get_consecutive_identical_words(title_retrieved, title_golden) > 5:
                    retrieved = True
                    print(title_retrieved)
                    print(title_golden)
            # Append max ROUGE score to the ROUGE list
            rouge_list.append(max_rouge_score)
            # If the document is correctly retrieved, increment the count
            if retrieved:
                count += 1
        # Append mean ROUGE score and document retrieval accuracy to respective lists
        average_rouge.append(np.mean(rouge_list))
        accuracy_list.append(count / len(retrieved_docs))

    # Print mean average ROUGE-1 score and mean document retrieval accuracy
    print("\nMean Average ROUGE-1 Score: ", np.mean(average_rouge))
    print("\nMean Document Retrieval Accuracy: ", np.mean(accuracy_list))

