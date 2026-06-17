from typing import List, Set, Optional, Dict
from llm_sdk import Small_LLM_Model
from src.models.functions_definiton import FunctionDefintion
import json


def extract_complete_json(text: str) -> Optional[str]:
    """
    Extracts the first complete JSON object from a string using brace counting.

    Args:
        text (str): The input text containing a potential JSON object.

    Returns:
        Optional[str]: The extracted JSON string if complete, otherwise None.
    """
    start = text.find("{")

    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        if text[i] == "}":
            brace_count -= 1

        if brace_count == 0:
            return text[start:i+1]

    return None


def get_best_valid_token(logits: List[float], valid_ids: Set[int]) -> int:
    """
    Selects the token ID with the highest logit value among a set
    of allowed token IDs.

    Args:
        logits (List[float]): The logit distribution from the model.
        valid_ids (Set[int]): The set of allowed token IDs.

    Returns:
        int: The ID of the best valid token.
    """
    return max(valid_ids, key=lambda i: logits[i])


def build_json_valid_ids(vocab: Dict[str, int]) -> Set[int]:
    """
    Filters the model vocabulary to find tokens that only
    contain characters safe for JSON.

    Args:
        vocab (Dict[str, int]): The model's token
        vocabulary (mapping string to ID).

    Returns:
        Set[int]: A set of token IDs that are "JSON-safe".
    """
    json_safe = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789*_.,:-+/\'!?()[]{}"ĠĊ'
    )
    valid = set()
    for token_str, token_id in vocab.items():
        if token_str and all(c in json_safe for c in token_str):
            valid.add(token_id)
    return valid


def load_vocabulary(model: Small_LLM_Model) -> Dict[str, int]:
    """
    Loads the vocabulary from the model's tokenizer file.

    Args:
        model (Small_LLM_Model): The model instance to load vocabulary from.

    Returns:
        Dict[str, int]: The vocabulary mapping token strings to IDs.
    """
    vocab_path = model.get_path_to_tokenizer_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    raw_vocab = tok_data.get("model", {}).get("vocab", {})
    return raw_vocab


def build_system_prompt(functions: List[FunctionDefintion]) -> str:
    """
    Constructs a strict system prompt containing
    the available function definitions.

    Args:
        functions (List[FunctionDefintion]):
        The list of functions to include in the prompt.

    Returns:
        str: The formatted system prompt.
    """
    lines = [
        "STRICT SYSTEM RULE: Use ONLY a matching\
function from the list below.",
        "If NO function matches the user's intent\
(even if types match), set name: \"none\".",
        "Never use an unrelated function for a different task.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(
            f"{name}: {info.type}"
            for name, info in fn.parameters.items()
        )
        lines.append(f"  -{fn.name}({params}): {fn.description}")

    lines.append('\nOutput ONLY valid JSON:\
{"name": "<fn>", "args": {<args>}}')

    return "\n".join(lines)
