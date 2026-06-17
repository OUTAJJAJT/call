from typing import List
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
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [FunctionDefintion(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON in {path}")


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
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON in {path}")
