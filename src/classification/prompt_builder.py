"""
Prompt Builder for AI classification.

Assembles structured prompts from:
  - Taxonomy (valid categories)
  - Retrieved examples (similar labeled cases)
  - Case text (the document to classify)

Usage:
    from classification.prompt_builder import build_classification_prompt

    prompt = build_classification_prompt(
        case_text="chiller temp high, cooling out of spec",
        retrieved_examples=examples,
        taxonomy_text=formatted_taxonomy,
    )
"""


def build_classification_prompt(case_text, retrieved_examples, taxonomy_text):
    """
    Build a complete prompt for problem classification.

    Parameters
    ----------
    case_text           : the text of the case to classify
    retrieved_examples  : list of dicts from retriever.retrieve_examples()
    taxonomy_text       : formatted taxonomy string from taxonomy.format_taxonomy_for_prompt()

    Returns the full prompt string ready to send to the LLM.
    """
    # format the retrieved examples
    examples_block = ""
    for i, ex in enumerate(retrieved_examples, 1):
        labels = ex["labels"]
        main = labels.get("nam_main_category", "unknown")
        sub = labels.get("nam_sub_category", "unknown")
        sim = ex["similarity"]
        # truncate long texts for the prompt
        text = ex["text"][:300] + "..." if len(ex["text"]) > 300 else ex["text"]
        examples_block += f"  Example {i} (similarity: {sim:.2f}):\n"
        examples_block += f"    Text: \"{text}\"\n"
        examples_block += f"    Classification: main_category={main}, sub_category={sub}\n\n"

    prompt = f"""You are an expert MRI service engineer classifying technical service calls.

TASK:
Classify the service call below into a main_category and sub_category.
You MUST choose from the categories listed below. Do not invent new categories.

TAXONOMY OF VALID CATEGORIES:
{taxonomy_text}

SIMILAR CASES AND THEIR CLASSIFICATIONS:
{examples_block}
CASE TO CLASSIFY:
\"{case_text}\"

INSTRUCTIONS:
- Choose the single best main_category and sub_category from the taxonomy above.
- If the case matches multiple categories, choose the PRIMARY root cause.
- If unsure, choose the category that best matches the similar cases above.
- Use "other" as sub_category only when no specific sub_category fits.

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

    Returns dict with main_category and sub_category, or None values if parsing fails.
    """
    result = {"main_category": None, "sub_category": None}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("main_category:"):
            result["main_category"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("sub_category:"):
            result["sub_category"] = line.split(":", 1)[1].strip()

    return result
