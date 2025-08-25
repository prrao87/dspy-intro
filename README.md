# DSPy Financial News Extraction Tutorial

This project demonstrates how to build an information extraction system using [DSPy](https://dspy.ai). We'll extract structured outputs from financial news articles, specifically identifying merger and acquisition deals and their associated information. The goal is to give an introduction to DSPy's core abstractions:
signatures, modules and optimizers to those who are coming from traditional systems that rely on
manual prompt engineering.

Blog post on this coming soon!

## Goals

This project aims to explain the following concepts:

- Understand DSPy's signatures and how they leverage Pydantic types
- Build a compound classification + information extraction pipeline via a custom DSPy module
- Implement evaluation metrics for structured outputs
- Optimize LM performance with bootstrap few-shot examples and in-context learning

## Setup

Install dependencies via [uv](https://docs.astral.sh/uv/getting-started/installation/) as follows:
```bash
uv sync
```
Add any additional dependencies as needed with the `uv add <package_name>` command.

## Usage

There are three key scripts to run for the initial extraction, as shown below.

```bash
# Run information extraction up to a limit of 5 articles
uv run extract.py --limit 5
# Process all 12 articles
uv run extract.py

# Evaluate existing predictions by specifying a result output file
uv run evaluate.py -o output.json

# Run optimization experiment
uv run optimize.py
```

Once the optimized module is available and persisted to disk, you can reload it and run the improved
module using the fourth script:

```bash
uv run optimized_extract.py
```

This outputs the new result to the file `new_outputs.json`, which can then be run through the evaluation
script once more to compare the results vs. the baseline. Depending on the type of optimizer used
and the LM, your mileage may vary.

## How it works

Financial news contains valuable structured information, but it's buried in unstructured text.
The data for this exercise is in the file `data/articles.json`.
Consider the following example:

> "Australia's Newcrest Mining has closed the acquisition of Pretium Resources, which owns the Brucejack mine... for $2.8bn (C$3.5bn)..."

We want to extract the following fields:
- **Type**: Acquisition
- **Parent Company**: Newcrest Mining  
- **Child Company**: Pretium Resources
- **Deal Amount**: 2.8 billion
- **Currency**: USD

We can do this using a DSPy pipeline that has two stages:

1. **Classification**: Is this article about a "merger", "acquisition", or "other" (e.g., failed acquisition)?
2. **Extraction**: Extract structured data based on the classification of "merger" or "acquisition"

## Core Components

This section lists the core components of the codebase.

### 1. Data Models

We use Pydantic to define our structured output so that we can obtain complex types from our LM as output.

```python
class MergerInfo(BaseModel):
    companies: list[str]         # Companies involved in merger
    tickers: list[str]           # Stock ticker symbols  
    deal_amount: float | None    # Deal value in millions/billions
    deal_currency: str | None    # Currency (USD, EUR, etc.)
    article_type: str            # Always "merger"

class AcquisitionInfo(BaseModel):
    parent_company: str          # Acquiring company
    child_company: str           # Target company
    deal_amount: float | None    # Deal value in millions/billions
    deal_currency: str | None    # Currency
    article_type: str            # Always "acquisition"
```

These Pydantic models are used to define the output fields for their respective signatures.

### 2. DSPy Pipeline

The `Extract` class orchestrates three DSPy modules:

- **Classifier**: Determines article type using a DSPy Signature
- **Merger Extractor**: Extracts merger details when applicable  
- **Acquisition Extractor**: Extracts acquisition details when applicable

The latter two modules are branches of the first, i.e., depending on the output of the classifier module,
the appropriate extractor module is called downstream.

### 3. Evaluation System

The evaluation compares predicted vs. ground truth data field-by-field:

- **Total accuracy**: Number of exact matches / total number of samples
- **Field-level accuracy**: Each field is scored individually and the articles that have these mismatches
are listed for debugging purposes.

### 4. Optimization

DSPy's `BootstrapFewShot` optimizer helps improve performance by generating examples from training data
For this simple demo, the gold dataset is split into 8 training and 4 test samples, and the optimizer
works by selecting high-quality examples based on evaluation metrics. The optimized module
is then run via the script `optimized_extract.py` to generate another output, `new_output.json`, which
contains the improved predictions.

Once the optimization process is complete and the new output has been generated, it's trivial to rerun
the evaluation to see if the results have improved:

```bash
uv run evaluate.py -o new_output.json
```