import os
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from evaluate import create_dspy_examples, validate_with_metric
from extract import Extract

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "openrouter/google/gemini-2.0-flash-001")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model=MODEL_NAME,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm, adapter=BAMLAdapter())

# Start with baseline module
baseline_extract = Extract()

# Create Examples for training/testing
# This is a simple and naive example, so we just train on the first 8 examples
# and test on the remaining
examples = create_dspy_examples(num_sentences=5)
marker_id = 8
train_set = examples[:marker_id]

# Use with DSPy optimizer
optimizer = dspy.BootstrapFewShot(metric=validate_with_metric, max_bootstrapped_demos=8, max_rounds=4)
optimized_extract = optimizer.compile(baseline_extract, trainset=train_set)
# Save path for optimized module
optimized_extract.save("./optimized_module.json")
