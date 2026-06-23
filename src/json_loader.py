from typing import List, Any, Tuple, Dict
from src.models.functions_definiton import FunctionDefintion
from src.models.prompts import Prompt
import json


def load_function_definition(path: str) -> List[FunctionDefintion]:
    """
    Loads function definitions from a JSON file.

    Args:
        path (str): The file path to the JSON definitions.

    Returns:
        List[FunctionDefintion]: A list of validated function definition
        objects.

    Raises:
        RuntimeError: If the file is not found or contains invalid JSON.
    """
    def _no_duplicates_object_pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f"Duplicate key '{k}' in JSON object")
            d[k] = v
        return d

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_no_duplicates_object_pairs_hook)
        return [FunctionDefintion(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON in {path}")
    except ValueError as e:
        # propagate duplicate-key detection with a clear message
        raise RuntimeError(f"Invalid JSON in {path}: {e}")


def load_prompts(path: str) -> List[Prompt]:
    """
    Loads natural language prompts from a JSON file.

    Args:
        path (str): The file path to the JSON prompts.

    Returns:
        List[Prompt]: A list of validated prompt objects.

    Raises:
        RuntimeError: If the file is not found or contains invalid JSON.
    """
    def _no_duplicates_object_pairs_hook(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f"Duplicate key '{k}' in JSON object")
            d[k] = v
        return d

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=_no_duplicates_object_pairs_hook)

        # Validate shape: expect a list of objects each containing exactly one
        # key: 'prompt'. Reject ambiguous or malformed entries (e.g. duplicates
        # or multiple keys in one object).
        if not isinstance(data, list):
            raise RuntimeError(f"Prompts file {path} must contain a JSON list")

        prompts: List[Prompt] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError(f"Item at index {idx} in {path} is not an object")
            if 'prompt' not in item:
                raise RuntimeError(f"Item at index {idx} in {path} missing 'prompt' key")
            if len(item) != 1:
                # This covers cases where an object contains multiple keys,
                # including duplicated 'prompt' entries which are detected
                # earlier by the object_pairs_hook.
                raise RuntimeError(
                    f"Item at index {idx} in {path} contains extra keys; "
                    "expected only 'prompt'"
                )
            prompts.append(Prompt(**item))

        return prompts
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON in {path}")
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}")
