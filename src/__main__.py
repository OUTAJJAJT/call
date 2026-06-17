import argparse
from src.json_loader import load_function_definition, load_prompts
from src.constrained_decoding import build_system_prompt
from llm_sdk import Small_LLM_Model
from src.constrained_decoding import (load_vocabulary,
                                      build_json_valid_ids,
                                      get_best_valid_token,
                                      extract_complete_json)
import time
import json
import os


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the function calling pipeline.

    Returns:
        argparse.Namespace: The parsed arguments including input/output paths
        and model name.
    """
    parse = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls \
using constrained decoding."
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


def main() -> None:
    """
    Main execution pipeline:
    1. Loads data and model.
    2. Builds the constrained vocabulary.
    3. Iterates through prompts, generating valid JSON
    function calls via logit masking.
    4. Saves results and displays performance metrics.
    """
    args = parse_args()
    functions = load_function_definition(args.functions_definition)
    if not functions:
        raise RuntimeError(
            "No function definition found. Please provide at least one."
        )
    prompts = load_prompts(args.input)
    if not prompts:
        raise RuntimeError(
            "No prompts found in input file. Please provide at least one."
        )

    system = build_system_prompt(functions)

    print(f"Loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        raise RuntimeError(
            f"Model {args.model} not found or failed to download"
        )
    vocab = load_vocabulary(model)
    valid_ids = build_json_valid_ids(vocab)

    all_results = []
    start_time = time.time()

    print("Processing prompts...\n")
    for p in prompts:
        prompt = p.prompt
        print(f"Processing prompt: {prompt}")
        full_prompt = f"{system}\n\nUser prompt: {prompt}\nAssistant:"
        input_ids = model.encode(full_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated = []
        clean_json = None

        # Pre-injecting the start of the JSON to guide the model
        all_generated.extend(model.encode('{"name": "')[0].tolist())

        for _ in range(50):
            logits = model.get_logits_from_input_ids(generated_ids +
                                                     all_generated)
            next_id = get_best_valid_token(logits, valid_ids)
            all_generated.append(next_id)
            text = model.decode(all_generated)
            clean_json = extract_complete_json(text)
            if clean_json:
                try:
                    parsed = json.loads(clean_json)
                    break
                except Exception:
                    pass

        if not clean_json:
            parsed = {"name": "none", "args": {}}

        raw_args = parsed.get("args", {})

        float_args = {}
        for k, v in raw_args.items():
            if isinstance(v, (int, float)):
                float_args[k] = float(v)
            else:
                float_args[k] = v
        all_results.append({
            "prompt": prompt,
            "name": parsed.get("name", "none"),
            "parameters": float_args
        })

        name = parsed.get("name", "none")
        parameter = parsed.get("args", {})

        if name == "none":
            print("-> ❌ No matching function found.")
        elif not parameter and name != "none":
            print(f"-> ⚠️ {name} called but args are empty.")
        else:
            print(f"-> ✅ {name}({parameter})")

    total_time = time.time() - start_time
    all_parsed_result = []
    for result in all_results:
        if result["name"] != "none":
            all_parsed_result.append(result)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_parsed_result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {args.output}")
    print(f"Total time: {total_time:.2f} seconds")
    if prompts:
        print(f"Average per prompt: {total_time/len(prompts):.2f} seconds")
        print(f"Success rate: {len(all_parsed_result)}/{len(prompts)} \
({len(all_parsed_result)/len(prompts)*100:.2f}%)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser stopped the program.")
    except Exception as error:
        print(f"Error: {error}")
