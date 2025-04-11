import glob
import os
from unstructured.partition.auto import partition
from typing import List
from rag.utils import el_to_doc
from rag.common.registry import registry
from langchain_core.documents import Document

@registry.register_ingestor('SimpleIngestor')
class SimpleIngestor:
    """
    This class leverages the Unstructured open-source library to parse files.
    Unlike LangChain's Unstructured integration, this implementation provides
    greater flexibility by decoupling chunking from partitioning. This design
    allows the chunking strategy to be part of the retriever class, which is
    responsible for storing and retrieving embedded chunks.

    While chunking could be disabled by setting a sufficiently high max chunk
    size, LangChain's integration returns data as a Document type, which is
    incompatible with the chunking methods provided by the Unstructured library.
    To address this, a helper function was created to facilitate conversion
    between formats.
    """

    def _get_file_paths(self, dir_path: str | List[str]): 
        dir_paths = [dir_path] if isinstance(dir_path, str) else dir_path

        for dir in dir_paths:
            if not os.path.isdir(dir):
                raise Exception(f"Path - {dir} is not a directory")

        valid_extensions = ('md', 'txt', 'pdf')
        files = []
        for ext in valid_extensions:
            files.extend(glob.glob(f'{dir_path}/**/*.{ext}', recursive=True))
        return files
        

    def load_dir(self, dir_path: str | List[str]) -> List[Document]: 
        files = self._get_file_paths(dir_path)
        docs = []
        for file in files:
            docs.extend(self.load_file(file))
        return docs

    def _validate_filetype(self, file_path: str | List[str]):
        if isinstance(file_path, str):
            ext = file_path.split(".")[-1]
            if ext not in ("txt", "md", "pdf"):
                raise Exception("Invalid file extension")

        if isinstance(file_path, list):
            for f in file_path:
                ext = f.split(".")[-1]
                if ext not in ("txt", "md", "pdf"):
                    raise Exception("Invalid file extension")

    def load_file(self, file_path: str | List[str] = None, file=None) -> List[Document]:
        """Load and chunk all files provided by the paths. The path can be a single file path
        or a list of file paths."""
        if file_path is not None:
            self._validate_filetype(file_path)
            if isinstance(file_path, list):
                elements = []
                for path in file_path:
                    elements.extend(partition(path))

            if isinstance(file_path, str):
                elements = partition(file_path)

        if file is not None:
            elements = partition(file=file)
        return el_to_doc(elements)
