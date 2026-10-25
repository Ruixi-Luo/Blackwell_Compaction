"""Proportional length control: adjust requested length from measured tokenizer feedback."""
import math
from .compressors.llm_rewrite import SUMMARY_PROMPT

def drafting_budget(previous, budget):
    old_request=previous.get('drafting_budget',budget)
    return max(32,min(2*budget,round(old_request*(.975*budget/max(1,previous['tokens'])))))

def feedback_messages(context, requested, previous, measured):
    # The true ex-ante budget stays in the runner. This is an internal controller
    # setting, correcting the writer's systematic token-count error.
    words=max(5,round(len(previous.split())*requested/max(1,measured)))
    system=SUMMARY_PROMPT+' This is a precise length-revision task. Follow the NEW drafting length below.'
    request=(f'Rewrite the previous draft using the original source. NEW drafting length: approximately {requested} tokens '
        f'(roughly {words} whitespace-separated words). This new length overrides earlier length instructions. '
        'Preserve as many specific source facts as this length permits. Return only readable factual prose.\n'
        '<source>\n'+context+'\n</source>')
    direction='Condense wording and remove lower-priority details' if requested<measured else 'Restore omitted facts from the source'
    feedback=(f'The previous draft was measured at {measured} tokens. {direction}. '
        f'Your new drafting target is {requested} tokens, NOT {measured}. '
        'Do not repeat the previous draft unchanged. Output only the revised prose, without a token count or preamble.')
    return [{'role':'system','content':system},{'role':'user','content':request},
            {'role':'assistant','content':previous},{'role':'user','content':feedback}]
