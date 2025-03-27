from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_chroma import Chroma
from typing import Any, List, Optional
from langchain_community.vectorstores.utils import filter_complex_metadata
from pydantic import Field, PrivateAttr

import weave
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
# print("PATH", os.path.abspath(os.path.join(__file__, '../..')))

from rag.chunkers.base import BaseChunkingStrategy
from rag.retrievers.base import Retriever

class SimpleRetriever(weave.Model, Retriever): 
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
            print('CHunking using default chunker!!!!*****************')
            chunker = self.chunker
        
        chunked_docs = chunker.chunk(docs)
        self.vector_db.add_documents(filter_complex_metadata(chunked_docs))
    
    def query(self, input: str)-> List[Document]:
        relevant_docs = self._retriever.invoke(input)
        return relevant_docs

    # Use this only for evaluation with weave
    @weave.op()
    def predict(self, input: str) -> List[str]:
        docs = self.query(input)
        return [doc.page_content for doc in docs]
    
    

class MMR(weave.Model):
    # Initialize parameters with proper type hints
    model: str
    vector_db: Optional[VectorStore] = None
    fetch_k: int = 1
    k: int = 1
    diversity: float = Field(default=0.5, ge=0.0, le=1.0, description="Diversity parameter (0=max diversity, 1=min diversity)")
    chunker: Optional[BaseChunkingStrategy] = None
    _retriever: Optional[VectorStoreRetriever] = PrivateAttr(default=None)
    docs: Optional[List[Document]] = None

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

        self._retriever = self.vector_db.as_retriever(
            search_type='mmr', search_kwargs={
                'k': self.k,
                'fetch_k': self.fetch_k,
                'lambda_mult': self.diversity
            }
        )

    def add_docs(self, docs: List[Document], chunker: Optional[BaseChunkingStrategy] = None):
        # Workaround for ChromaDb not supporting list as metadata value - convert to string
        if chunker is None:
            print('CHunking using default chunker!!!!*****************')
            chunker = self.chunker
        
        chunked_docs = chunker.chunk(docs)
        self.vector_db.add_documents(filter_complex_metadata(chunked_docs))
    
    def query(self, input: str)-> List[Document]:
        relevant_docs = self._retriever.invoke(input)
        return relevant_docs

    # Use this only for evaluation with weave
    @weave.op()
    def predict(self, input: str) -> List[str]:
        docs = self.query(input)
        return [doc.page_content for doc in docs]

