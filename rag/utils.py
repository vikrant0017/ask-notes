import os
import sys
import importlib.util
import inspect
from typing import Any, Callable, List, Optional


from typing import Any, Iterator, List, cast
from langchain_core.documents import Document
from unstructured.documents.elements import (
    TYPE_TO_TEXT_ELEMENT_MAP,
    ElementType,
    Element,
    ElementMetadata,
    Text,
    DataSourceMetadata
)

# Usage example for both classes and functions
def get_callable_from_name(
    callable_name: str, search_dirs: list[str]
) -> Optional[Callable]:
    """
    Find a callable (function or class) by name from the specified directories.

    Args:
        callable_name: Name of the function or class to find
        search_dirs: List of directories to search through

    Returns:
        The callable object if found, None otherwise
    """
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            print(f"Warning: Directory {search_dir} does not exist")
            continue

        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)
                    module_name = os.path.splitext(file)[0]
                    unique_module_name = (
                        f"dynamic_import_{os.path.basename(root)}_{module_name}"
                    )

                    try:
                        spec = importlib.util.spec_from_file_location(
                            unique_module_name, file_path
                        )
                        if spec is None or spec.loader is None:
                            continue

                        module = importlib.util.module_from_spec(spec)
                        sys.modules[unique_module_name] = module
                        spec.loader.exec_module(module)

                        if hasattr(module, callable_name):
                            obj = getattr(module, callable_name)
                            if inspect.isfunction(obj):
                                return obj
                            elif inspect.isclass(obj):
                                return obj
                    except (ImportError, ModuleNotFoundError, AttributeError) as e:
                        print(f"Warning: Error importing {file_path}: {str(e)}")
                        continue

    return None



def execute_callable(
    callable_name: str, search_dirs: List[str], **params,
) -> Any:


    callable_obj = get_callable_from_name(callable_name, search_dirs)
    if callable_obj is None:
        raise ValueError(f"Callable {callable_name} not found")

    # Instantiate class or call function
    return callable_obj(**params)  




def el_to_doc(elements) -> List[Document]:
    """Load file."""
    elements_json = [element.to_dict() for element in elements]
    docs = []
    for element in elements_json:
        # metadata = self._get_metadata()
        metadata = dict()
        metadata.update(element.get("metadata"))  # type: ignore
        metadata.update({"category": element.get("category") or element.get("type")})
        metadata.update({"element_id": element.get("element_id")})
        docs.append(Document(page_content=cast(str, element.get("text")), metadata=metadata))
    
    return docs


# def get_metadata(self) -> dict[str, Any]:
#     """Get file_path metadata if available."""
#     return {"source": self.file_path} if self.file_path else {}


def convert_elements_to_dicts(self, elements: list[Element]) -> list[dict[str, Any]]:
    return [element.to_dict() for element in elements]


def doc_to_el(documents: List[Document]) -> List[Element]:
    """
    Convert a list of Langchain Document objects to unstructured Element objects.
    The element category is determined from the document's metadata['category'] field.

    Args:
        documents: List of Langchain Document objects

    Returns:
        List of unstructured Element objects
    """
    elements = []

    for doc in documents:
        # Get the category from metadata, default to NarrativeText if not specified
        category = doc.metadata.get("category", ElementType.NARRATIVE_TEXT)

        # Get the appropriate Element class based on category
        if category in TYPE_TO_TEXT_ELEMENT_MAP:
            ElementClass = TYPE_TO_TEXT_ELEMENT_MAP[category]
        else:
            # Default to Text if category is not recognized
            ElementClass = Text

        # Create metadata object
        metadata = ElementMetadata()

        # Copy document metadata to Element metadata
        for key, value in doc.metadata.items():
            if key != "category":  # Skip category as we've already used it
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        # Handle common metadata fields specifically
        if "source" in doc.metadata:
            metadata.filename = doc.metadata.get("source")

        if "page" in doc.metadata:
            metadata.page_number = doc.metadata.get("page")

        # Create data source if URL exists
        if "url" in doc.metadata:
            metadata.url = doc.metadata.get("url")
            metadata.data_source = DataSourceMetadata(
                url=doc.metadata.get("url"),
                date_created=doc.metadata.get("created", None),
                date_modified=doc.metadata.get("last_modified", None),
            )

        # Create the element
        element = ElementClass(text=doc.page_content, metadata=metadata, element_id=doc.metadata.get('element_id'))

        elements.append(element)

    return elements
