"""Length-only policies for an ex-ante budget. No post-truncation or cross-method matching."""
import math
from .compressors.llm_rewrite import SUMMARY_PROMPT

def in_band(tokens, budget):
    return math.ceil(.95*budget) <= tokens <= budget

def choose_candidate(candidates, budget):
    feasible = [c for c in candidates if c['tokens'] <= budget]
    band = [c for c in feasible if in_band(c['tokens'],budget)]
    if not feasible:
        raise ValueError('No complete candidate fits the fixed budget; never truncate.')
    # Length only. A short candidate remains explicit; it cannot lower another method's budget.
    return max(band or feasible,key=lambda c:c['tokens'])

def revision_messages(context, budget, previous, previous_tokens):
    target=math.floor(.975*budget)
    words=max(5,round(len(previous.split())*target/max(1,previous_tokens)))
    system=SUMMARY_PROMPT+' You must follow the requested length range, not default to a brief summary.'
    request=(f'The entire source is below. Produce a detailed factual summary for unknown future questions. '
        f'Target {math.ceil(.95*budget)} to {budget} Qwen tokens, ideally {target}. '
        f'An approximate guide is {words} whitespace-separated words; the token range takes priority. '
        'There is NO other word limit. Do not add headings, meta-comments, invented facts or padding. '
        'Use the available space to preserve specific source details.\n<source>\n'+context+'\n</source>')
    direction='Expand by restoring omitted source details' if previous_tokens<target else 'Shorten by condensing wording and prioritizing source details'
    feedback=(f'The previous draft has {previous_tokens} Qwen tokens, outside the target range. '
        f'{direction}. Aim for {target} tokens (approximately {words} words). '
        'Return ONLY the revised summary. Do not discuss length or answer any question.')
    return [{'role':'system','content':system},{'role':'user','content':request},
            {'role':'assistant','content':previous},{'role':'user','content':feedback}]
