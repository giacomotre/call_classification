"""
Prompt Builder for AI classification.

Assembles structured prompts from:
  - Taxonomy (valid categories)
  - Retrieved examples (similar labeled cases)
  - Case context (rich text of the case to classify)

Usage:
    from src.classification.prompt_builder import build_classification_prompt, parse_classification_response

    prompt = build_classification_prompt(case_context, examples, taxonomy_text)
    result = parse_classification_response(llm_response)
"""
import pandas as pd
from config import EXAMPLE_TRUNCATION_LENGTH


def build_classification_prompt(case_context, retrieved_examples, taxonomy_text,
                                truncation_length=None):
    """
    Build a prompt for problem classification.

    Parameters
    ----------
    case_context        : rich text of the case (from build_llm_context)
    retrieved_examples  : list of dicts from retriever.retrieve_examples()
    taxonomy_text       : formatted taxonomy string
    truncation_length   : max chars per example text (defaults to config value)

    Returns the full prompt string ready to send to the LLM.
    """
    if truncation_length is None:
        truncation_length = EXAMPLE_TRUNCATION_LENGTH

    # format retrieved examples, skip any with missing labels
    examples_block = ""
    for i, ex in enumerate(retrieved_examples, 1):
        main = ex["labels"].get("nam_main_category", "unknown")
        sub = ex["labels"].get("nam_sub_category", "unknown")

        # skip examples with null labels
        if main is None or (isinstance(main, float) and pd.isna(main)):
            continue

        sim = ex["similarity"]
        
        # base_text = ex.get("llm_context", ex["text"])
        #text = base_text[:truncation_length] + "..." if len(base_text) > truncation_length else base_text
        base_text = ex.get("llm_context", ex["text"])

        parts = base_text.split("\n\n")

        # keep primary info (signals + structured)
        primary_text = "\n\n".join(parts[:2])

        # optionally append repair as secondary info
        repair_text = ""
        if "Repair Action" in base_text:
            repair_text = "\n(Repair info): " + base_text.split("Repair Action")[-1][:150]

        text = primary_text + repair_text
        #uncomment the two line before and the delete up to here
        examples_block += f"  Example {i} (similarity: {sim:.2f}):\n"
        examples_block += f'    Text: "{text}"\n'
        examples_block += f"    Classification: main_category={main}, sub_category={sub}\n\n"

    prompt = f"""You are an expert MRI service engineer. Your task is to classify the problem of a technical service call into a fixed number of categories.

VALID CATEGORIES:
{taxonomy_text}

SIMILAR CASES (already classified by engineers):
{examples_block}
CASE TO CLASSIFY:
"{case_context}"

Choose exactly ONE main_category and sub_category from the valid categories above.
Do not invent new categories. If unsure, choose the category that best matches the similar cases above.
If the case matches multiple categories, choose the PRIMARY root cause.

RESPOND IN EXACTLY THIS FORMAT (nothing else):
main_category: <value>
sub_category: <value>"""

    return prompt


def parse_classification_response(response_text):
    """
    Parse the LLM response into main_category and sub_category.

    Expected format:
        main_category: software
        sub_category: process_crash

    Returns dict with main_category and sub_category (None if parsing fails).
    """
    result = {"main_category": None, "sub_category": None}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("main_category:"):
            result["main_category"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("sub_category:"):
            result["sub_category"] = line.split(":", 1)[1].strip()

    return result
