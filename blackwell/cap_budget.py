"""Select a complete summary by generation order, without seeing a question."""


def first_complete_feasible(candidates, budget):
    """Return the first verified complete, nonempty candidate within token budget.

    candidates: saved drafts in generation order, with independently verified
        completion status and token counts; no prediction scores are inspected.
    budget: maximum allowed tokens, fixed from the original source length.
    """
    for candidate in candidates:
        if (candidate.get('complete_verified') is True
                and isinstance(candidate.get('text'), str)
                and candidate['text'].strip()
                and 0 < candidate['tokens'] <= budget):
            return candidate
    raise ValueError('No complete summary fits the cap; do not truncate or fabricate.')
