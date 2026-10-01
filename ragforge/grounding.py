import re


def validate_citations(answer: str, source_count: int) -> tuple[bool, str]:
    """Validate that a generated answer cites only available source IDs."""
    if not answer.strip() or source_count == 0:
        return False, "No answer or no evidence was available."
    ids = {int(x) for x in re.findall(r"\[S(\d+)\]", answer)}
    if ids and not ids.issubset(set(range(1, source_count + 1))):
        return False, "Answer contains a citation that does not map to retrieved evidence."
    return True, "ok"
