from pathlib import Path
from typing import List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json
import os


os.environ["OPENAI_API_KEY"] = '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'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class JSONLoader(BaseLoader):
    def __init__(
            self,
            file_path: Union[str, Path],
            content_key: Optional[str] = None,
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs = []
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)

            # Iterate through 'pages'
            for record in data:
                metadata = {}

                metadata["year"] = record.get("pub_date").get('year')
                metadata["month"] = record.get("pub_date").get('month')
                metadata["day"] = record.get("pub_date").get('day')
                metadata["title"] = record.get("article_title")

                docs.append(Document(page_content=record.get(self._content_key), metadata=metadata))
        return docs


def load_document(document_file_path):
    loader = JSONLoader(
        file_path=document_file_path,
        content_key='article_abstract'
    )

    return loader.load()


class RAG:
    def __init__(self, document_file_path, embedding_model):
        # Load Document from File Path
        self.data = load_document(document_file_path)

        # Split Documents using TokenTextSplitter to chunks
        text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
        self.chunks = text_splitter.split_documents(self.data)

        # Choose the Embeddings Model
        if embedding_model == 'e5':
            self.embeddings = HuggingFaceEmbeddings(
                                    model_name="intfloat/e5-large-unsupervised",
                                    model_kwargs={'device': 'cuda'},
                                    encode_kwargs={'normalize_embeddings': False}
                              )

        elif embedding_model == 'OpenAI':
            self.embeddings = OpenAIEmbeddings()
        else:
            raise NotImplementedError

        # Setup database
        self.database = FAISS.from_documents(self.chunks, self.embeddings)

    def get_retriever(self, k):
        return self.database.as_retriever(k=k)


if __name__ == '__main__':
    rag_pipeline = RAG('refs/pubmed_2023.json', embedding_model='OpenAI')

    retriever = rag_pipeline.get_retriever(k=2)

    docs_1 = retriever.get_relevant_documents("What is the most common neurological disease published in December 2023")
    print(docs_1)

    docs_2 = retriever.get_relevant_documents("What were the results of the DESTINY-Breast04 Trial?")
    print(docs_2)
