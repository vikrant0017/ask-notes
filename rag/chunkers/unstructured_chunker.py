from typing import List

from langchain_core.documents.base import Document
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title

from rag.chunkers.base import BaseChunkingStrategy
from rag.common.registry import registry
from rag.utils import doc_to_el, el_to_doc


@registry.register_chunker('basic_chunking')
class BasicChunking(BaseChunkingStrategy):
    def __init__(self, max_characters=250, overlap=40):
        super().__init__()
        self.max_characters = max_characters
        self.overlap = overlap

    def chunk(self, docs: List[Document]) -> List[Document]:
        elements = doc_to_el(docs)
        print(elements)
        chunks = chunk_elements(
            elements, max_characters=self.max_characters, overlap=self.overlap
        )
        return el_to_doc(chunks)


@registry.register_chunker('by_title_chunking')
class ByTitleChunking(BaseChunkingStrategy):
    def __init__(
        self,
        combine_text_under_n_characters=None,
        max_characters=250,
        overlap=40,
        new_after_n_chars=250,
    ):
        super().__init__()
        self.max_characters = max_characters
        self.overlap = overlap
        self.combine_text_under_n_characters = (
            combine_text_under_n_characters
            if combine_text_under_n_characters is not None
            else max_characters
        )
        self.new_after_n_chars = new_after_n_chars

    def chunk(self, docs: List[Document]) -> List[Document]:
        elements = doc_to_el(docs)
        # for element in elements:
        #     print(element.metadata.to_dict())
        chunks = chunk_by_title(
            elements,
            max_characters=self.max_characters,
            overlap=self.overlap,
            new_after_n_chars=self.new_after_n_chars,
            combine_text_under_n_chars=self.combine_text_under_n_characters,
        )

        return el_to_doc(chunks)
