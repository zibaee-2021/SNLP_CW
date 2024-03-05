from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json
import os


os.environ["OPENAI_API_KEY"] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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

            docs.append(Document(page_content=record.get('article_abstract'), metadata=metadata))

    return docs


def load_retrieved_json_docs(retrieved_json_path):
    docs = []

    # Load JSON file
    with open(retrieved_json_path, encoding='UTF-8') as file:
        data = json.load(file)

        # Iterate through 'pages'
        for record in data:
            metadata = {"url": record.get("url"),
                        "title": record.get("title")}

            docs.append(Document(page_content=record.get('abstract'), metadata=metadata))

    return docs


def load_docs(document_name, document_file_path):
    if document_name[:6] == 'PubMed':
        docs = load_pubmed_json_docs(document_file_path)
    elif document_name[:6] == 'BioASQ':
        docs = load_retrieved_json_docs(document_file_path)
    else:
        raise NotImplementedError

    return docs


def get_embeddings(embedding_model):
    if embedding_model == 'e5':
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-unsupervised",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )

    elif embedding_model == 'OpenAI':
        embeddings = OpenAIEmbeddings()
    else:
        raise NotImplementedError

    return embeddings


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


def load_faiss_database(documents, embedding_model):
    # Get the Embeddings Model
    embeddings = get_embeddings(embedding_model)

    # Load database from file
    database = FAISS.load_local(f"FAISS/faiss_index_{documents}_{embedding_model}", embeddings)

    return database


if __name__ == '__main__':
    load = True

    if load:
        rag_database = load_faiss_database(documents='BioASQ_11B_test',
                                           embedding_model='OpenAI')
    else:
        rag_database = save_faiss_database(documents='BioASQ_11B_test',
                                           document_file_path='refs/retrieved_BioASQ_test.json',
                                           embedding_model='OpenAI')

    retriever = rag_database.as_retriever(k=2)

    docs_1 = retriever.get_relevant_documents("What is the most common neurological disease published in December 2023")
    print(docs_1)

    docs_2 = retriever.get_relevant_documents("What were the results of the DESTINY-Breast04 Trial?")
    print('\n', docs_2)
