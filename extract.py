import argparse
import json
import os
import re
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv
from dspy.adapters.baml_adapter import BAMLAdapter
from pydantic import BaseModel, Field

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "openrouter/google/gemini-2.0-flash-001")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model=MODEL_NAME,
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm, adapter=BAMLAdapter())


def read_data(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def extract_first_n_sentences(text: str, num_sentences: int = 3) -> str:
    # Define sentence boundary pattern: period followed by space or newline
    pattern = r"\.(?:\s+|\n+)"

    # Split the text into sentences
    sentences = re.split(pattern, text)

    # Filter out empty sentences and join the first 3 with periods
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    first_n = valid_sentences[:num_sentences]

    # Join with periods and spaces
    result = ". ".join(first_n) + "."
    return result


# --- Pydantic models ---


class Merger(BaseModel):
    article_id: int | None = Field(default=None)
    company_1: str | None = Field(description="First company in the merger")
    company_1_ticker: list[str] | None = Field(description="Stock ticker of first company")
    company_2: str | None = Field(description="Second company in the merger")
    company_2_ticker: list[str] | None = Field(description="Stock ticker of second company")
    merged_entity: str | None = Field(description="Name of merged entity")
    deal_amount: str | None = Field(description="Total monetary amount of the deal")
    deal_currency: Literal["USD", "CAD", "AUD", "Unknown"] = Field(
        description="Currency of the merger deal"
    )
    article_type: Literal["merger"] = "merger"


class Acquisition(BaseModel):
    article_id: int | None = Field(default=None)
    parent_company: str | None = Field(description="Parent company in the acquisition")
    parent_company_ticker: list[str] | None = Field(description="Stock ticker of parent company")
    child_company: str | None = Field(description="Child company in the acquisition")
    child_company_ticker: list[str] | None = Field(description="Stock ticker of child company")
    deal_amount: str | None = Field(description="Total monetary amount of the deal")
    deal_currency: Literal["USD", "CAD", "AUD", "Unknown"] = Field(
        description="Currency of the acquisition deal"
    )
    article_type: Literal["acquisition"] = "acquisition"


class Other(BaseModel):
    article_id: int | None = Field(default=None)
    article_type: Literal["other"] = "other"


# --- Signatures ---


class ClassifyArticle(dspy.Signature):
    """
    Analyze the following news article and classify it according to whether it's a "Merger" or "Acquisition".
    If it mentions a potential or failed deal, classify it as "Other".
    """

    text: str = dspy.InputField()
    article_type: Literal["merger", "acquisition", "other"] = dspy.OutputField()


class ExtractMergerInfo(dspy.Signature):
    """
    Extract merger information about companies from the given text.
    """

    text: str = dspy.InputField()
    merger_info: Merger = dspy.OutputField()


class ExtractAcquisitionInfo(dspy.Signature):
    """
    Extract acquisition information about companies from the given text.
    """

    text: str = dspy.InputField()
    acquisition_info: Acquisition = dspy.OutputField()


class Extract(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict(ClassifyArticle)
        self.merger_extractor = dspy.Predict(ExtractMergerInfo)
        self.acquisition_extractor = dspy.Predict(ExtractAcquisitionInfo)

    def classify(self, text: str, num_sentences: int = 3) -> str:
        text = extract_first_n_sentences(text, num_sentences=num_sentences)
        result = self.classifier(text=text)
        article_type = result.article_type
        return article_type

    def forward(self, text: str, article_id: int) -> Merger | Acquisition | Other:
        # Implement extraction logic here
        article_type = self.classify(text)
        if article_type == "merger":
            extracted_result = self.merger_extractor(text=text, num_sentences=5)
            merger_info = extracted_result.merger_info
            merger_info.article_id = article_id
            return merger_info
        elif article_type == "acquisition":
            extracted_result = self.acquisition_extractor(text=text, num_sentences=5)
            acquisition_info = extracted_result.acquisition_info
            acquisition_info.article_id = article_id
            return acquisition_info
        else:
            return Other(article_id=article_id, article_type="other")


def run_sync(data_path: Path, limit: int) -> None:
    articles = read_data(data_path / "articles.json")
    extract = Extract()
    output = []
    for article in articles[:limit]:
        # Combine the title and the content's first n sentences
        article_id = article["id"]
        text = article["title"] + extract_first_n_sentences(article["content"], 5)
        result = extract(text=text, article_id=article_id)
        final_result = result.model_dump()
        print(f"Finished processing article {article_id}")
        output.append(final_result)

    # Write results to file
    with open(data_path / "output.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Extracted results written to {data_path / 'output.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", "-l", type=int, default=15, help="Number of articles to process")
    args = parser.parse_args()

    data_path = Path("./data")
    run_sync(data_path, limit=args.limit)
