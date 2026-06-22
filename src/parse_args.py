import argparse


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the function calling pipeline.

    Returns:
        argparse.Namespace: The parsed arguments including input/output paths
        and model name.
    """
    parse = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls"
        " using constrained decoding."
    )

    parse.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing user prompts."
    )

    parse.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing available function definitions."
    )

    parse.add_argument(
        "--output",
        type=str,
        default="data/output/functions_results.json",
        help="Path where the generated function calls will be saved."
    )

    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Identifier of the model to use from Hugging Face."
    )

    return parse.parse_args()
