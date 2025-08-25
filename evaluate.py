import argparse
from pathlib import Path
from typing import Any

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


def create_dspy_examples(
    data_path: Path = Path("./data"), num_sentences: int = 5
) -> list[dspy.Example]:
    """
    Create dspy.Example objects from articles.json and gold.json.

    Args:
        data_path: Path to the data directory containing articles.json and gold.json
        num_sentences: Number of sentences to include from article content (default: 5)

    Returns:
        List of dspy.Example objects with text input and expected Pydantic model output
    """
    import json

    # Load articles and gold data
    with open(data_path / "articles.json", "r") as f:
        articles = json.load(f)

    with open(data_path / "gold.json", "r") as f:
        gold_data = json.load(f)

    # Create a mapping from article_id to gold data for quick lookup
    gold_by_id = {item["article_id"]: item for item in gold_data}

    examples = []
    for article in articles:
        article_id = article["id"]

        # Skip if no gold data available for this article
        if article_id not in gold_by_id:
            continue

        # Create text input using the same logic as extract.py:157
        text = article["title"] + "\n" + extract_first_n_sentences(
            article["content"], num_sentences
        )

        # Pbtain the expected Pydantic model from gold data
        expected_output = get_gold_pydantic_model(gold_by_id[article_id])

        # Create dspy.Example with inputs marked
        example = dspy.Example(
            text=text, article_id=article_id, expected_output=expected_output
        ).with_inputs("text", "article_id")

        examples.append(example)

    return examples


def metric(gold_val: Any, pred_val: Any, trace=None) -> int:
    """
    Define a DSPy metric for the optimizer.
    This one calculates an exact match score.
    """
    # Handle None/empty equivalence
    if gold_val in [None, []] and pred_val in [None, []]:
        return 1
    return 1 if gold_val == pred_val else 0


def validate_with_metric(
    example: dspy.Example, pred: Merger | Acquisition | Other, trace=None
) -> float:
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


def evaluate_results(gold_file: str, result_file: str):
    # Load both datasets
    gold_df = pl.read_json(gold_file)
    output_df = pl.read_json(result_file)

    # Join on article_id to align records
    comparison_df = gold_df.join(output_df, on="article_id", suffix="_pred")

    # Get all field names except article_id, sorted alphabetically
    gold_columns = [col for col in gold_df.columns if col != "article_id"]
    all_fields = sorted(set(gold_columns))

    # Calculate field-level accuracy
    field_accuracies = {}
    field_mismatches = {}
    total_comparisons = 0
    total_correct = 0

    for field in all_fields:
        pred_field = f"{field}_pred"

        # Get values for comparison, handling missing columns
        gold_values = (
            comparison_df.get_column(field).to_list()
            if field in comparison_df.columns
            else [None] * len(comparison_df)
        )
        pred_values = (
            comparison_df.get_column(pred_field).to_list()
            if pred_field in comparison_df.columns
            else [None] * len(comparison_df)
        )

        # Calculate accuracy for this field
        field_scores = []
        mismatches = []

        for i, (gold_val, pred_val) in enumerate(zip(gold_values, pred_values)):
            score = metric(gold_val, pred_val)
            field_scores.append(score)
            total_comparisons += 1
            total_correct += score

            if score == 0:
                article_id = comparison_df.get_column("article_id").to_list()[i]
                mismatches.append(article_id)

        field_accuracy = sum(field_scores) / len(field_scores) if field_scores else 0
        field_accuracies[field] = field_accuracy
        field_mismatches[field] = mismatches

    # Calculate total accuracy
    total_accuracy = (
        (total_correct / total_comparisons * 100) if total_comparisons > 0 else 0
    )

    # Pretty print results
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal Accuracy: {total_accuracy:.1f}%")
    print(f"({total_correct}/{total_comparisons} field comparisons correct)")

    print("\n" + "-" * 40)
    print("FIELD-LEVEL ACCURACY")
    print("-" * 40)

    for field in sorted(field_accuracies.keys()):
        accuracy_pct = field_accuracies[field] * 100
        print(f"{field:<25} {accuracy_pct:>6.1f}%")

    print("\n" + "-" * 40)
    print("MISMATCHES BY FIELD")
    print("-" * 40)

    for field in sorted(field_mismatches.keys()):
        mismatches = field_mismatches[field]
        if mismatches:
            print(f"{field:<25} Article IDs: {mismatches}")
        else:
            print(f"{field:<25} No mismatches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate extraction")
    parser.add_argument("--gold", "-g", default="./data/gold.json", help="Path to the gold standard file")
    parser.add_argument("--result", "-r", default="./data/output.json", help="Path to the result file")

    args = parser.parse_args()
    gold_file = args.gold
    result_file = args.result
    evaluate_results(gold_file, result_file)
