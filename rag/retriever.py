from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
from pydantic import Field
import weave


class SimpleRetriever(weave.Model):
    model: str = None
    k: int = 1
    retriever: VectorStoreRetriever = Field(default=None, init=None)

    def __init__(self, docs: List[Document], **data):
        super().__init__(**data)
        self.init_retriever(docs)

    def init_retriever(self, docs: List[Document]):
        embedding = OllamaEmbeddings(
            model=self.model,
        )

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
        )

        self.retriever = vector_db.as_retriever(search_kwargs={"k": self.k})

    @weave.op()
    def predict(self, query: str) -> List[Document]:
        relvant_docs = self.retriever.invoke(query)
        return [doc.page_content for doc in relvant_docs]
