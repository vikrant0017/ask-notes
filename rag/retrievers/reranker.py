from typing import List
from langchain_core.documents import Document
from rag.chunkers.base import BaseChunkingStrategy
from rag.retrievers.base import Retriever
from sentence_transformers import CrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from rag.retrievers.retriever import SimpleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_chroma import Chroma
from typing import Any, List, Optional
from langchain_community.vectorstores.utils import filter_complex_metadata
from pydantic import Field, PrivateAttr
import weave


class Reranker(weave.Model, Retriever): 
    emedding_model: str = Field(default='cross-encoder/ms-marco-MiniLM-L6-v2', description='Hugging face model name')
    cross_encoding_model: str = 'nomic-embed-text'
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
            model=self.emedding_model,
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
