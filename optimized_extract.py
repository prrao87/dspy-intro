import os
import json
from pathlib import Path

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from evaluate import create_dspy_examples
from extract import Extract, read_data

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "openrouter/google/gemini-2.0-flash-001")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model=MODEL_NAME,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm, adapter=BAMLAdapter())

# Create Examples for training/testing
# This is a simple and naive example, so we just train on the first 8 examples
# and test on the remaining
examples = create_dspy_examples(num_sentences=5)

# Start with baseline module
optimized_extract = Extract()
optimized_extract.load("./optimized_extract.json")

# Rerun the optimized module on the full dataset
data_path = Path("./data")
articles = read_data(data_path / "articles.json")
output = []
article_id = 1
for article in examples:
    # Combine the title and the content's first n sentences
    text = article.text
    result = optimized_extract(text=text, article_id=article_id)
    final_result = result.model_dump()
    print(f"Finished processing article {article_id}")
    output.append(final_result)
    article_id += 1

# Write results to file
optimized_result_file = "new_output.json"
with open(data_path / optimized_result_file, "w") as f:
    json.dump(output, f, indent=2)
print(f"Optimized extractor results written to {str(data_path / optimized_result_file)}")