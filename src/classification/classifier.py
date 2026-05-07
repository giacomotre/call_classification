"""
AI Classifier for service calls.

Sends prompts to the LLM API and parses responses.
Retrieval is handled separately — this module only builds prompts and calls the LLM.

Usage:
    from src.classification.classifier import classify_batch

    predictions = classify_batch(case_contexts, all_examples, taxonomy_text)
"""
import pandas as pd
from config import LLM_TEMPERATURE
from src.classification.prompt_builder import build_classification_prompt, parse_classification_response


# ── LLM API call ─────────────────────────────────────────────────────

def call_llm(prompt: str, temperature: float = None) -> str:
    """
    Call the LLM API.

    Uses EU-hosted OpenAI endpoint. Requires OPENAI_API_KEY in environment.
    """
    import os
    from openai import OpenAI

    if temperature is None:
        temperature = LLM_TEMPERATURE

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://eu.api.openai.com/v1",
    )

    resp = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt,
        temperature=temperature,
    )

    return resp.output_text


# ── Single case classification ───────────────────────────────────────

def classify_case(case_context, retrieved_examples, taxonomy_text):
    """
    Classify a single service call.

    Parameters
    ----------
    case_context        : LLM context text (from build_llm_context)
    retrieved_examples  : list of dicts from retrieve_examples()
    taxonomy_text       : formatted taxonomy string

    Returns dict with main_category and sub_category.
    """
    prompt = build_classification_prompt(case_context, retrieved_examples, taxonomy_text)
    response = call_llm(prompt)
    return parse_classification_response(response)


# ── Batch classification ─────────────────────────────────────────────

def classify_batch(case_contexts, all_examples, taxonomy_text, progress=True):
    """
    Classify multiple service calls.

    Retrieval is done beforehand — this function only builds prompts
    and calls the LLM for each case.

    Parameters
    ----------
    case_contexts   : list of LLM context strings (from build_llm_context)
    all_examples    : list of lists from retrieve_batch()
    taxonomy_text   : formatted taxonomy string
    progress        : print progress updates

    Returns DataFrame with columns: main_category, sub_category
    """
    total = len(case_contexts)
    print(f"  Classifying {total} cases...")

    results = []
    for i, (context, examples) in enumerate(zip(case_contexts, all_examples)):
        if progress and (i + 1) % 50 == 0:
            print(f"  Classified {i + 1}/{total}...")

        prompt = build_classification_prompt(context, examples, taxonomy_text)
        response = call_llm(prompt)
        result = parse_classification_response(response)
        results.append(result)

    print(f"  Classification complete: {total} cases processed")
    return pd.DataFrame(results)
