import os
import sys
import importlib.util
import inspect
from typing import Any, Callable, List, Optional


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