from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json
import os

from Datasets import BioASQ

# os.environ["OPENAI_API_KEY"] = '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'
# os.environ["OPENAI_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_medrag_pubmed_json(json_path_list):
    docs = []

    for file_path in json_path_list:
        # Load JSON file
        with open(file_path, 'r', encoding='UTF-8') as file:
            for line in file:
                data = json.loads(line)

                metadata = {"id": data.get("id"),
                            "title": data.get("title")}

                docs.append(Document(page_content=data.get('content'), metadata=metadata))

    return docs


def get_embeddings(embedding_model):
    if embedding_model == 'e5':
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-unsupervised",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    elif embedding_model == 'OpenAI':
        embeddings = OpenAIEmbeddings()
    else:
        raise NotImplementedError

    return embeddings


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


def load_faiss_database(documents, embedding_model):
    embeddings = get_embeddings(embedding_model)

    # Load database from file
    database = FAISS.load_local(f"FAISS/faiss_index_{documents}_{embedding_model}", embeddings)

    return database


if __name__ == '__main__':
    file_path_list = []
    for i in range(1, 11):
        file_path_list.append('refs/pubmed/chunk/pubmed23n{:04d}.jsonl'.format(i))

    load = True

    if load:
        retrieval_model = load_faiss_database(documents='BioASQ_11B_test',
                                              embedding_model='OpenAI')
    else:
        retrieval_model = save_faiss_database(documents='BioASQ_11B_test',
                                              file_path_list=file_path_list,
                                              embedding_model='OpenAI')

    '''
    # ------------------------------------------------------------------------------------------------------------------

    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])

    yesno_questions = data.get_type_questions('yesno')

    matching_ratio = []

    for question in yesno_questions:
        retrieved_docs = retrieval_model.similarity_search(question.question_body, k=len(question.docs))

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
    '''