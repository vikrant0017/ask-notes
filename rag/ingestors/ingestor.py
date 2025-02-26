from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from typing import List

"""
The UnstructuredLoader is a wrapper for the unstructured package. Refer to the package documentation for 
features and usage details.
"""

class SimpleIngestor:
    def load_dir(self, dirname):
        """Load and chunk all types of files in the provided directory"""
        dir_loader = DirectoryLoader(
            path=dirname,
            glob=["**/*.md", "**/*.txt"],
            loader_cls=UnstructuredLoader,
            loader_kwargs={
                "chunking_strategy": "basic",
                "max_characters": 100000,
                "include_orig_elements": False,
            },
        )
        docs = dir_loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=1, chunk_overlap=0, separator="."
        )
        chunks = splitter.split_documents(docs)
        return chunks

    # https://python.langchain.com/docs/integrations/providers/unstructured/
    def load_file(self, file_path: str | List[str]):
        """Load and chunk all files provided by the paths. The path can be a single file path
        or a list of file paths"""
        # https://docs.unstructured.io/open-source/core-functionality/chunking
        # The parameters below are directly from unstructured chunking api. It is a basic chunking
        # strategy which only chunks the raw texts and not semantic elemeents and returns the entire
        # document as a single chunk

        if isinstance(file_path, str):
            ext = file_path.split(".")[-1]
            if ext not in ("txt", "md"):
                raise Exception("Invalid file extension")

        if isinstance(file_path, list):
            for f in file_path:
                ext = f.split(".")[-1]
                if ext not in ("txt", "md"):
                    raise Exception("Invalid file extension")

        unstructured_loader = UnstructuredLoader(
            file_path=file_path,
            chunking_strategy="basic",
            max_characters=100000,
            include_orig_elements=False,
        )

        docs = unstructured_loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=1, chunk_overlap=0, separator="."
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def load_text(self, text: str):
        splitter = CharacterTextSplitter(
            chunk_size=1, chunk_overlap=0, separator="."
        )
        docs = [Document(text)]
        chunks = splitter.split_documents(docs)
        return chunks
