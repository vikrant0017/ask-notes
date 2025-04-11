from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents.base import Document


class BaseChunkingStrategy(ABC):
    @abstractmethod  # Similar to raising NotImplementedError
    def chunk(self, docs: List[Document]) -> List[Document]:
        # """Split text into chunks using the implemented strategy."""
        pass

