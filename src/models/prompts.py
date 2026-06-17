from pydantic import BaseModel


class Prompt(BaseModel):
    """
    Represents a natural language prompt to be processed by the LLM.

    Attributes:
        prompt (str): The text content of the prompt.
    """
    prompt: str