"""
Main evaluation script for DSPy extraction system.

This script provides DSPy-native evaluation using dspy.Evaluate framework
and maintains backward compatibility with JSON file comparison.
"""

import argparse
from pathlib import Path

import dspy
import polars as pl

from extract import Acquisition, Merger, Other, extract_first_n_sentences


def get_gold_pydantic_model(gold_data: dict) -> Merger | Acquisition | Other:
    """Reconstruct a Pydantic model from gold JSON data."""
    article_type = gold_data.get("article_type")

    if article_type == "merger":
        return Merger(**gold_data)
    elif article_type == "acquisition":
        return Acquisition(**gold_data)
    else:  # article_type == "other"
        return Other(**gold_data)


def load_json_file(file_path: Path):
    """Load JSON file with error handling."""
    import json

    with open(file_path, "r") as f:
        return json.load(f)


def get_column_values(df, column_name: str):
    """Safely extract column values with None fallback."""
    if column_name in df.columns:
        return df.get_column(column_name).to_list()
    return [None] * len(df)


def load_and_join_data(gold_file: str, result_file: str):
    """Load and join gold and prediction data."""
    gold_df = pl.read_json(gold_file)
    output_df = pl.read_json(result_file)
    comparison_df = gold_df.join(output_df, on="article_id", suffix="_pred")

    gold_columns = [col for col in gold_df.columns if col != "article_id"]
    all_fields = sorted(set(gold_columns))

    return comparison_df, all_fields


def metric(gold_val, pred_val, trace=None):  # trace unused but required for DSPy
    """
    Define a DSPy metric for the optimizer.
    This one calculates an exact match score.
    """
    # Handle None/empty equivalence
    if gold_val in [None, []] and pred_val in [None, []]:
        return 1
    return 1 if gold_val == pred_val else 0


# --- For live evaluation (running dspy.Evaluate and making new LM calls) ---


def create_dspy_examples(data_path: Path = Path("./data"), num_sentences: int = 5):
    """
    Create dspy.Example objects from articles.json and gold.json.

    Args:
        data_path: Path to the data directory containing articles.json and gold.json
        num_sentences: Number of sentences to include from article content (default: 5)

    Returns:
        List of dspy.Example objects with text input and expected Pydantic model output
    """
    # Load articles and gold data
    articles = load_json_file(data_path / "articles.json")
    gold_data = load_json_file(data_path / "gold.json")

    # Create a mapping from article_id to gold data for quick lookup
    gold_by_id = {item["article_id"]: item for item in gold_data}

    examples = []
    for article in articles:
        article_id = article["id"]

        # Skip if no gold data available for this article
        if article_id not in gold_by_id:
            continue

        # Create text input using the same logic as extract.py:157
        text = (
            article["title"]
            + "\n"
            + extract_first_n_sentences(article["content"], num_sentences)
        )

        # Obtain the expected Pydantic model from gold data
        expected_output = get_gold_pydantic_model(gold_by_id[article_id])

        # Create dspy.Example with inputs marked
        example = dspy.Example(
            text=text, article_id=article_id, expected_output=expected_output
        ).with_inputs("text", "article_id")

        examples.append(example)

    return examples


def validate_answer(
    example: dspy.Example,
    pred: Merger | Acquisition | Other,
    trace=None,  # trace unused but required for DSPy
):
    """DSPy-compatible metric function for evaluating extraction results."""
    expected = example.expected_output

    # Type mismatch or Other type handling
    if not isinstance(pred, type(expected)):
        return 0.0
    if isinstance(expected, Other):
        return 1.0

    # Compare all model fields using Pydantic's field info
    expected_dict = expected.model_dump()
    pred_dict = pred.model_dump()

    # Calculate field-level accuracy
    matches = [
        metric(expected_dict[field], pred_dict[field]) for field in expected_dict
    ]

    return sum(matches) / len(matches) if matches else 0.0


def run_evaluation(
    data_path: Path = Path("./data"),
    limit: int = 2,
    num_sentences: int = 5,
    num_threads: int = 1,
    optimized_module_path: str | None = None,
):
    """Run DSPy evaluation and detailed mismatch analysis."""
    from dspy.adapters.baml_adapter import BAMLAdapter

    from extract import Extract, lm

    # Configure DSPy
    dspy.configure(lm=lm, adapter=BAMLAdapter())

    # Create examples from gold data
    examples = create_dspy_examples(data_path, num_sentences)
    examples = examples[:limit]
    print(f"\nLoaded {len(examples)} examples for evaluation")

    # Initialize Extract module (base or optimized)
    extract_module = Extract()
    if optimized_module_path:
        extract_module.load(optimized_module_path)
        print("✅ Optimized module loaded successfully")

    # Run DSPy evaluation
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=validate_answer,
        num_threads=num_threads,
        display_progress=True,
        display_table=False,  # No pandas dependency, it's bloat when we already have Polars
    )

    results = evaluator(extract_module)

    # Get predictions and create temporary output file for analysis
    predictions = []
    for example in examples:
        pred = extract_module(text=example.text, article_id=example.article_id)
        pred_dict = pred.model_dump()
        pred_dict["article_id"] = example.article_id
        predictions.append(pred_dict)

    # Create temporary files and run mismatch analysis
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_pred:
        json.dump(predictions, temp_pred, indent=2)
        temp_pred_path = temp_pred.name

    # Run detailed comparison
    evaluate_results(str(data_path / "gold.json"), temp_pred_path)

    # Clean up
    import os

    os.unlink(temp_pred_path)

    return results


def calculate_field_accuracy(comparison_df, field: str):
    """Calculate accuracy and mismatches for a single field."""
    pred_field = f"{field}_pred"

    gold_values = get_column_values(comparison_df, field)
    pred_values = get_column_values(comparison_df, pred_field)
    article_ids = get_column_values(comparison_df, "article_id")

    field_scores = []
    mismatches = []

    for i, (gold_val, pred_val) in enumerate(zip(gold_values, pred_values)):
        score = metric(gold_val, pred_val)
        field_scores.append(score)

        if score == 0:
            mismatches.append(article_ids[i])

    field_accuracy = sum(field_scores) / len(field_scores) if field_scores else 0
    return field_accuracy, mismatches, field_scores


def print_mismatched_fields(field_mismatches: dict):
    """Print field mismatch results and which articles they are from."""
    print("\n" + "-" * 40)
    print("Mismatches ")
    print("-" * 40)

    for field in sorted(field_mismatches.keys()):
        mismatches = field_mismatches[field]
        if mismatches:
            print(f"{field:<25} ❌ Article IDs: {mismatches[:10]}")
        else:
            print(f"{field:<25} ✅ (no mismatches)")


def evaluate_results(gold_file: str, result_file: str):
    """Evaluate prediction results against gold standard."""
    comparison_df, all_fields = load_and_join_data(gold_file, result_file)

    field_accuracies = {}
    field_mismatches = {}
    total_comparisons = 0
    total_correct = 0

    for field in all_fields:
        field_accuracy, mismatches, field_scores = calculate_field_accuracy(
            comparison_df, field
        )

        field_accuracies[field] = field_accuracy
        field_mismatches[field] = mismatches
        total_comparisons += len(field_scores)
        total_correct += sum(field_scores)

    print_mismatched_fields(field_mismatches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate DSPy extraction system with mismatch analysis"
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=1000,
        help="Limit the number of examples to evaluate",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=Path,
        default=Path("./data"),
        help="Path to the data directory",
    )
    parser.add_argument(
        "--module",
        "-m",
        type=str,
        default=None,
        help="Path to a locally saved, optimized module file (e.g., 'optimized_extract.json')",
    )

    args = parser.parse_args()

    if args.module:
        print(f"Running DSPy evaluation with user-specified saved module: {args.module}")

    run_evaluation(
        data_path=args.data_path,
        limit=args.limit,
        optimized_module_path=args.module,
    )
