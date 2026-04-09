"""
AI Classifier for service calls.

Sends prompts to the internal LLM API and parses responses.
Currently uses a stub — replace call_llm() when API details arrive.

Usage:
    from classification.classifier import classify_case, classify_batch

    result = classify_case(case_text, index, embedding_model, taxonomy_text)
"""
import pandas as pd
from classification.retriever import retrieve_examples, retrieve_batch
from classification.prompt_builder import build_classification_prompt, parse_classification_response


# ── LLM API call ──────────────────────────────────────────────────────
# Replace this function when you get the API details.
# It should take a prompt string and return the LLM's response string.

def call_llm(prompt):
    """
    Send a prompt to the internal LLM API and return the response.

    TODO: Replace this stub with your actual API call.
    The function should:
      1. Send the prompt to the API
      2. Return the response text as a string

    Example for OpenAI-compatible API:
        import requests
        response = requests.post(
            "https://your-internal-api.company.com/v1/chat/completions",
            headers={"Authorization": "Bearer YOUR_KEY"},
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    """
    # STUB: prints the prompt and returns a placeholder
    print("\n" + "=" * 60)
    print("  LLM API STUB — prompt that would be sent:")
    print("=" * 60)
    print(prompt[:500])
    if len(prompt) > 500:
        print(f"  ... ({len(prompt)} total characters)")
    print("=" * 60)

    return "main_category: STUB\nsub_category: STUB"


# ── Single case classification ────────────────────────────────────────

def classify_case(case_text, index, embedding_model, taxonomy_text, n_examples=5):
    """
    Classify a single service call.

    Steps:
      1. Retrieve n similar labeled cases (local)
      2. Build prompt with taxonomy + examples + case (local)
      3. Call LLM API (remote)
      4. Parse response

    Returns dict with main_category, sub_category, and retrieved examples.
    """
    # step 1: retrieve similar labeled cases
    examples = retrieve_examples(case_text, index, embedding_model, n=n_examples)

    # step 2: build prompt
    prompt = build_classification_prompt(case_text, examples, taxonomy_text)

    # step 3: call LLM
    response = call_llm(prompt)

    # step 4: parse
    result = parse_classification_response(response)
    result["retrieved_examples"] = examples

    return result


# ── Batch classification ──────────────────────────────────────────────

def classify_batch(case_texts, index, embedding_model, taxonomy_text,
                   n_examples=5, batch_size=32, progress=True):
    """
    Classify multiple service calls.

    Embeds all cases in one batch (fast), then calls LLM per case.

    Parameters
    ----------
    case_texts      : list of text strings to classify
    index           : retrieval index from build_index()
    embedding_model : SentenceTransformer model
    taxonomy_text   : formatted taxonomy string
    n_examples      : number of similar cases to retrieve per case
    batch_size      : embedding batch size
    progress        : print progress

    Returns DataFrame with columns: text, main_category, sub_category
    """
    total = len(case_texts)
    print(f"  Classifying {total} cases...")

    # step 1: batch retrieve (embeds all queries at once)
    print(f"  Retrieving {n_examples} examples per case...")
    all_examples = retrieve_batch(
        case_texts, index, embedding_model,
        n=n_examples, batch_size=batch_size,
    )

    # step 2-4: build prompts and call LLM per case
    results = []
    for i, (text, examples) in enumerate(zip(case_texts, all_examples)):
        if progress and (i + 1) % 50 == 0:
            print(f"  Classified {i + 1}/{total}...")

        prompt = build_classification_prompt(text, examples, taxonomy_text)
        response = call_llm(prompt)
        result = parse_classification_response(response)
        result["text"] = text
        results.append(result)

    print(f"  Classification complete: {total} cases processed")

    return pd.DataFrame(results)
