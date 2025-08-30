import os

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from evaluate import create_dspy_examples, validate_answer
from extract import Extract

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model="openrouter/google/gemini-2.5-flash-lite",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm, adapter=BAMLAdapter())

# Start with baseline module
baseline_extract = Extract()

# Create Examples for training/testing
# Train on samples 1-12 only (the original training data)
train_set = create_dspy_examples(num_sentences=5, article_ids=list(range(1, 13)))

# Optional: Create test set for evaluation after optimization
test_set = create_dspy_examples(num_sentences=5, article_ids=list(range(13, 23)))

# Use with DSPy optimizer
optimizer = dspy.BootstrapFewShot(
    metric=validate_answer, max_bootstrapped_demos=8, max_rounds=4
)
optimized_extract = optimizer.compile(baseline_extract, trainset=train_set)

# Save optimized module for later use
optimized_extract.save("./optimized_extract.json")

# Optional: Evaluate baseline vs optimized performance on test set
print("\nEvaluating performance on test set (samples 13-22)...")
evaluator = dspy.Evaluate(metric=validate_answer, devset=test_set)

print("Evaluating baseline module performance")
baseline_score = evaluator(baseline_extract)

print("Evaluating optimized module performance")
optimized_score = evaluator(optimized_extract)

