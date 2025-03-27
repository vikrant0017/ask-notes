from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document

from rag.chunkers.base import BaseChunkingStrategy

# Interface for the Retriver Class
class Retriever(ABC):
    @abstractmethod
    def init_retriever(self): ...
    """
    Put everything requried for initialization. This will be called during method initializiton
    """

    @abstractmethod
    def add_docs(self, docs: List[Document], chunker: BaseChunkingStrategy | None): ...

    @abstractmethod
    def query(self, prompt:str): List[Document]
