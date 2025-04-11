import glob
import importlib
import os
from typing import List


def setup_imports(exclude_modules: List[str]):
    """
    Import modules for each of the components, which allows the registry decorators
    to run and register the decorated components
    """
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    retrievers_path = os.path.join(root_dir, 'retrievers')
    retrievers_pattern = os.path.join(retrievers_path, '**', '*.py')

    generators_path = os.path.join(root_dir, 'generators')
    generators_pattern = os.path.join(generators_path, '**', '*.py')

    ingestors_path = os.path.join(root_dir, 'ingestors')
    ingestors_pattern = os.path.join(ingestors_path, '**', '*.py')

    chunkers_path = os.path.join(root_dir, 'chunkers')
    chunkers_pattern = os.path.join(chunkers_path, '**', '*.py')

    scorers_path = os.path.join(root_dir, 'scorers')
    scorers_pattern = os.path.join(scorers_path, '**', '*.py')

    files = (
        glob.glob(retrievers_pattern, recursive=True)
        + glob.glob(ingestors_pattern, recursive=True)
        + glob.glob(generators_pattern, recursive=True)
        + glob.glob(chunkers_pattern, recursive=True)
        + glob.glob(scorers_pattern, recursive=True)
    )

    for f in files:
        f = os.path.normpath(f)
        if f.endswith('.py') and not f.endswith('__init__.py'):
            f_splits = f[f.find('rag/'):f.find('.py')].split(os.sep)
            module = '.'.join(f_splits)
            if module not in exclude_modules:
                importlib.import_module(module)