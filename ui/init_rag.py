from rag.rag import RAG
from rag.generators.generator import ResponseGenerator
from rag.ingestors.ingestor import SimpleIngestor
from rag.retrievers.retriever import SimpleRetriever, MMR
from rag.chunkers.base import ByTitleChunking
from rag.retrievers.reranker import Reranker

def initialize_rag():
    generator = ResponseGenerator(model="qwen2.5:3b")
    ingestor = SimpleIngestor()
    retriever = Reranker(
        emedding_model="nomic-embed-text",
        cross_encoding_model='cross-encoder/ms-marco-MiniLM-L6-v2',
        fetch_k=6,
        k=3,
        chunker=ByTitleChunking(
            max_characters=1500, new_after_n_chars=1000, combine_text_under_n_characters=300
        ),
    )
    return RAG(retriever, generator, ingestor)
    
