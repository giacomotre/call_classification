def compute_info_score(row):
    """
    Compute how much useful textual information a case contains
    based on extracted fields.
    """
    fields = [
        row.get("extracted_problem_description_remote", ""),
        row.get("extracted_error_remote", ""),
        row.get("extracted_malfunction_area_remote", ""),
        row.get("extracted_diagnostic_remote", ""),
    ]

    # count non-empty characters
    return sum(len(str(f).strip()) for f in fields if f)


def has_enough_information(row, min_chars=40):
    """
    Return True if case has enough information to be classified.
    """
    return compute_info_score(row) >= min_chars