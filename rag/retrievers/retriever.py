from typing import List, Optional

import weave
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from pydantic import Field, PrivateAttr

from rag.chunkers.base import BaseChunkingStrategy
from rag.common.registry import registry
from rag.retrievers.base import BaseRetriever


@registry.register_retriever('SimpleRetriever')
class SimpleRetriever(weave.Model, BaseRetriever): 
    model: str = None
    k: int = 1
    _retriever: VectorStoreRetriever = PrivateAttr(default=None)
    vector_db: Optional[VectorStore] = Field(default=None, init=None)
    chunker: Optional[BaseChunkingStrategy] = Field(default=None, init=None)
    docs: Optional[List[Document]] = None
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def model_post_init(self, __context):
        self.init_retriever(self.docs)

    def init_retriever(self, docs: List[Document] = None):
        embedding = OllamaEmbeddings(
            model=self.model,
        )

        self.vector_db = Chroma(
            embedding_function=embedding,
        )
        if docs is not None:
            self.add_docs(docs)

        if self.similarity_threshold:
            self._retriever = self.vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": self.k, "score_threshold": self.similarity_threshold})
        else:
            self._retriever = self.vector_db.as_retriever(search_kwargs={"k": self.k})

    def add_docs(self, docs: List[Document], chunker: Optional[BaseChunkingStrategy] = None):
        # Workaround for ChromaDb not supporting list as metadata value - convert to string
        if chunker is None:
            chunker = self.chunker
        
        chunked_docs = chunker.chunk(docs)
        self.vector_db.add_documents(filter_complex_metadata(chunked_docs))
    
    def query(self, input: str) -> List[Document]:
        relevant_docs = self._retriever.invoke(input)
        return relevant_docs

    # Use this only for evaluation with weave
    @weave.op()
    def predict(self, input: str) -> List[str]:
        docs = self.query(input)
        return [doc.page_content for doc in docs]
    
    