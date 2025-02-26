from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_chroma import Chroma
from typing import List
from pydantic import Field
import weave


class SimpleRetriever(weave.Model):
    model: str = None
    k: int = 1
    retriever: VectorStoreRetriever = Field(default=None, init=None)
    vector_db: VectorStore = Field(default=None, init=None)

    def __init__(self, docs: List[Document] = None, **data):
        super().__init__(**data)
        self.init_retriever(docs)

    def init_retriever(self, docs: List[Document] = None):
        embedding = OllamaEmbeddings(
            model=self.model,
        )

        self.vector_db = Chroma(
            embedding_function=embedding,
        )
        if docs:
            self.add_docs(docs)

        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": self.k})

    def add_docs(self, docs: List[Document]):
        # Workaround for ChromaDb not supporting list as metadata value - convert to string
        for doc in docs:
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    doc.metadata[key] = ",".join(value)
        self.vector_db.add_documents(docs)

    @weave.op()
    def predict(self, query: str) -> List[Document]:
        relevant_docs = self.retriever.invoke(query)
        return relevant_docs
