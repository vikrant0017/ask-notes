from typing import List, Optional

import weave
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from pydantic import Field, PrivateAttr

from rag.chunkers.base import BaseChunkingStrategy
from rag.common.registry import registry
from rag.retrievers.base import BaseRetriever


@registry.register_retriever('reranker')
class Reranker(weave.Model, BaseRetriever): 
    cross_encoding_model: str = Field(default='cross-encoder/ms-marco-MiniLM-L6-v2', description='Hugging face model name')
    model: str = 'nomic-embed-text'
    fetch_k:int = 6
    k: int = 3
    vector_db: Optional[VectorStore] = Field(default=None, init=None)
    chunker: Optional[BaseChunkingStrategy] = Field(default=None, init=None)
    docs: Optional[List[Document]] = None

    _retriever: VectorStoreRetriever = PrivateAttr(default=None)

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

        # https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/
        retriever = self.vector_db.as_retriever(search_kwargs={"k": self.fetch_k})
        model = HuggingFaceCrossEncoder(model_name=self.cross_encoding_model)
        compressor = CrossEncoderReranker(model=model, top_n=self.k)
        self._retriever = ContextualCompressionRetriever( # Combiens and passes docuemtns for compressesion 
            base_compressor=compressor, base_retriever=retriever
        )

    def add_docs(self, docs: List[Document], chunker: Optional[BaseChunkingStrategy] = None):
        # Workaround for ChromaDb not supporting list as metadata value - convert to string
        if chunker is None:
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
