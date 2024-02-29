import json
from pathlib import Path
from typing import List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

        documents = []
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)

            # Iterate through 'pages'
            for record in data:
                metadata = {"year": record.get("pub_date").get('year'),
                            "month": record.get("pub_date").get('month'),
                            "day": record.get("pub_date").get('day'),
                            "title": record.get("article_title")}

                documents.append(Document(page_content=record.get(self._content_key), metadata=metadata))

        return documents


def load_document(document_file_path):
    loader = JSONLoader(
        file_path=document_file_path,
        content_key='article_abstract'
    )

    return loader.load()


class RAG:
    def __init__(self, document_file_path):
        self.data = load_document(document_file_path)

        self.database = self.load_e5_database()

    def load_e5_database(self):
        text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
        chunks = text_splitter.split_documents(self.data)

        modelPath = "intfloat/e5-large-unsupervised"

        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # Using faiss index
        db = FAISS.from_documents(chunks, embeddings)

        return db

    def retrieve_document(self, query):
        return self.database.similarity_search(query)

    def get_retriever(self, k):
        return self.database.as_retriever(k=k)


if __name__ == '__main__':
    rag_pipeline = RAG('dataset/pubmed_december-2023.json')

    docs = rag_pipeline.retrieve_document("What is the most common neurological disease published in December 2023")
    print(docs)
