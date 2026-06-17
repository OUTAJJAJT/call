from pydantic import BaseModel
from typing import Dict


class Parameter(BaseModel):
    """
    Represents a parameter in a function definition.

    Attributes:
        type (str): The expected data type of the parameter
        (e.g., 'float', 'int', 'str').
    """
    type: str


class ReturnType(BaseModel):
    """
    Represents the return type of a function.

    Attributes:
        type (str): The data type returned by the function.
    """
    type: str


class FunctionDefintion(BaseModel):
    """
    Represents the full schema of a function available for the LLM to call.

    Attributes:
        name (str): The unique name of the function.
        description (str): A brief description of what the function does.
        parameters (Dict[str, Parameter]): A mapping of
        parameter names to their definitions.
        returns (ReturnType): The definition of the return value.
    """
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: ReturnType
