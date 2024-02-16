import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

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

def RAG_DB_setup(data):
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    modelPath = "intfloat/e5-large-unsupervised"
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Using faiss index
    db = FAISS.from_documents(chunks, embeddings)

    return db

if __name__ == '__main__':
    loader = JSONLoader(
        file_path='dataset/pubmed_december-2023.json',
        # jq_schema='.[]',
        content_key='article_abstract'
        # metadata_func=metadata_func
    )
    data = loader.load()

    print(f"{len(data)} pubmed articles are loaded!")

    db = RAG_DB_setup(data)

    query = "What is the most common neurological disease published in December 2023"
    docs = db.similarity_search(query)
    print(docs)
