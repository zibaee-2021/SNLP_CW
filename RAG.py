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

# os.environ["OPENAI_API_KEY"] = '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'
# os.environ["OPENAI_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_consecutive_identical_words(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()

    max_consecutive_identical = 0
    consecutive_identical = 0

    for word1, word2 in zip(words1, words2):
        if word1 == word2:
            consecutive_identical += 1
            max_consecutive_identical = max(max_consecutive_identical, consecutive_identical)
        else:
            consecutive_identical = 0

    return max_consecutive_identical


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


def append_golden_faiss_databse(database, json_path, embedding_model):
    # Load Document from File Path
    # data = load_docs(documents, document_file_path)
    docs = load_retrieved_json_docs(json_path)

    # Split Documents using TokenTextSplitter to chunks
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Get the Embeddings Model
    embeddings = get_embeddings(embedding_model)

    database.afrom_documents(chunks, embeddings)

    return database


if __name__ == '__main__':
    '''
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

    BioASQ_test_golden_ref = {}

    with open('refs/BioASQ_11B_test_yesno.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)

        for record in data:
            BioASQ_test_golden_ref[record.get("url")] = record.get("title")

    # ------------------------------------------------------------------------------------------------------------------

    rag_database = load_faiss_database(documents='MedRAG_133000', embedding_model='OpenAI')
    rag_database_2 = load_faiss_database(documents='BioASQ_11B_test', embedding_model='OpenAI')
    rag_database.merge_from(rag_database_2)

    # rag_database = append_golden_faiss_databse(rag_database, json_path='refs/BioASQ_11B_test_yesno.json',
    #                                            embedding_model='OpenAI')

    data = BioASQ(['dataset/Task11BGoldenEnriched/11B1_golden.json',
                   'dataset/Task11BGoldenEnriched/11B2_golden.json',
                   'dataset/Task11BGoldenEnriched/11B3_golden.json',
                   'dataset/Task11BGoldenEnriched/11B4_golden.json'])

    yesno_questions = data.get_type_questions('yesno')

    average_rouge, average_bleu, accuracy_list = [], [], []
    rouge = Rouge()

    for question in yesno_questions:
        retrieved_docs = rag_database.similarity_search(question.question_body, k=4)

        rouge_list, bleu_list = [], []
        count = 0

        for doc in retrieved_docs:
            max_rouge_score, max_bleu_score = 0, 0

            retrieved = False

            for golden_ref in question.refs:
                # Compute ROUGE-1 score
                score = rouge.get_scores(doc.page_content, golden_ref['text'])
                rouge_1_score = score[0]['rouge-1']['f']
                max_rouge_score = max(rouge_1_score, max_rouge_score)

                # bleu_score = sentence_bleu(doc.page_content, golden_ref['text'])
                # max_bleu_score = max(bleu_score, max_bleu_score)

                # Compare if title identical
                title_retrieved = doc.metadata['title']
                title_golden = BioASQ_test_golden_ref.get(golden_ref['document'], '')

                if get_consecutive_identical_words(title_retrieved, title_golden) > 5:
                    retrieved = True

                    print(title_retrieved)
                    print(title_golden)

            rouge_list.append(max_rouge_score)
            # bleu_list.append(max_bleu_score)

            if retrieved:
                count += 1

        average_rouge.append(np.mean(rouge_list))
        # average_bleu.append(np.mean(bleu_list))
        accuracy_list.append(count / len(retrieved_docs))

    print("\nMean Average ROUGE-1 Score: ", np.mean(average_rouge))
    # print("\nMean Average BLEU Score: ", np.mean(average_bleu))

    print("\nMean Document Retrieval Accuracy: ", np.mean(accuracy_list))

