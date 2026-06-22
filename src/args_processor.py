from typing import List, Dict, Any
from src.models.functions_definiton import FunctionDefintion


def build_fn_param_types(
    functions: List[FunctionDefintion]
) -> Dict[str, Dict[str, str]]:
    """
    Builds a lookup dictionary mapping each function name to its
    parameter names and their expected types.

    Args:
        functions (List[FunctionDefintion]): The list of loaded functions.

    Returns:
        Dict[str, Dict[str, str]]: A mapping of function name to
        {param_name: param_type}.

    Example:
        {
            "fn_add_numbers":  {"a": "number", "b": "number"},
            "fn_is_even":      {"n": "integer"},
            "fn_greet":        {"name": "string"},
        }
    """
    return {
        fn.name: {name: param.type for name, param in fn.parameters.items()}
        for fn in functions
    }


def cast_args(
    raw_args: Dict[str, Any],
    param_types: Dict[str, str]
) -> Dict[str, Any]:
    """
    Casts each argument to its expected type based on the function definition.
    Also filters out any hallucinated parameters not in the definition.

    Args:
        raw_args (Dict[str, Any]): The raw arguments parsed
        from the model output.
        param_types (Dict[str, str]): The expected types for each parameter.

    Returns:
        Dict[str, Any]: The cleaned and type-cast arguments.
    """
    typed_args = {}
    for k, v in raw_args.items():
        if k not in param_types:
            continue
        expected_type = param_types.get(k, "")
        if expected_type == "number":
            typed_args[k] = float(v)
        elif expected_type == "integer":
            typed_args[k] = int(v)
        else:
            typed_args[k] = v
    return typed_args
