"""不调用 LLM：核验 HotpotQA 与 QASPER 的规模、标注一致性和 Qwen token 长度。"""
import argparse
from collections import Counter
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.datasets import hotpotqa, qasper
from blackwell.storage import file_hash, write_json
from blackwell.tokenization import ContextTokenizer, TOKENIZER_REVISION


def length_stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    # nearest-rank 分位数：定义明确，不依赖 numpy/pandas。
    return {"n": len(values), "mean": sum(values) / len(values), "min": values[0],
            "p50": values[math.ceil(.5 * len(values))-1],
            "p90": values[math.ceil(.9 * len(values))-1], "max": values[-1],
            "bins": {"<4000": sum(x < 4000 for x in values),
                     "4000-7999": sum(4000 <= x < 8000 for x in values),
                     "8000-15999": sum(8000 <= x < 16000 for x in values),
                     ">=16000": sum(x >= 16000 for x in values)}}


def subset_stats(records):
    by_paper = {r['paper_id']: r['context_tokens'] for r in records}
    counts = Counter(r['paper_id'] for r in records)
    return {"questions": len(records), "papers": len(by_paper),
            "yes": sum(r['label'] for r in records), "no": sum(1-r['label'] for r in records),
            "papers_with_multiple_questions": sum(n > 1 for n in counts.values()),
            "tokens_per_question": length_stats([r['context_tokens'] for r in records]),
            "tokens_per_unique_paper": length_stats(list(by_paper.values()))}


def census_qasper(papers, tokenizer):
    exclusions, evidence_counts, annotation_types = Counter(), Counter(), Counter()
    records = []
    total_questions = 0
    for paper_id, paper in papers.items():
        token_count = None
        has_body = any(p.strip() for section in paper['full_text'] for p in section['paragraphs'])
        for question in paper['qas']:
            total_questions += 1
            annotation_types.update(qasper.answer_type(a['answer']) for a in question['answers'])
            status, label = qasper.classify_question(question)
            if status == 'eligible' and not has_body:
                status = 'missing_full_text'
            if status != 'eligible':
                exclusions[status] += 1
                continue
            if token_count is None:
                token_count = tokenizer.count(qasper.build_context(paper))
            evidence = qasper.text_evidence_status(question, paper)
            evidence_counts[evidence] += 1
            records.append({'paper_id': paper_id, 'question_id': question['question_id'],
                            'label': label, 'annotations': len(question['answers']),
                            'context_tokens': token_count, 'evidence_status': evidence})
    conservative = [r for r in records if r['evidence_status'] == 'text_verified']
    assert sum(exclusions.values()) + len(records) == total_questions
    assert sum(evidence_counts.values()) == len(records)
    if len({r['question_id'] for r in records}) != len(records):
        raise ValueError('Duplicate QASPER question IDs.')
    return {'all_papers': len(papers), 'all_questions': total_questions,
            'annotation_types': dict(annotation_types), 'exclusions': dict(exclusions),
            'strict_boolean': subset_stats(records), 'text_verified': subset_stats(conservative),
            'evidence_status': dict(evidence_counts), 'eligible_questions': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.offline and not (ROOT/'data/tokenizer/qwen3-30b-a3b-instruct-2507.json').exists():
        raise RuntimeError('Prepare the tokenizer before offline census.')
    tokenizer = ContextTokenizer()
    pool, provenance = hotpotqa.load_pool(args.offline)
    hotpot = {'provenance': provenance, 'questions': len(pool), 'yes': sum(e['label'] for e in pool),
              'no': sum(1-e['label'] for e in pool), 'types': dict(Counter(e['question_type'] for e in pool)),
              'paragraph_counts': dict(Counter(e['paragraph_count'] for e in pool)),
              'context_tokens': length_stats([tokenizer.count(e['context']) for e in pool])}
    result = {'schema_version': 1, 'tokenizer_revision': TOKENIZER_REVISION,
              'code_sha256': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                              [Path(__file__), ROOT/'blackwell/datasets/qasper.py', ROOT/'blackwell/datasets/hotpotqa.py']},
              'hotpotqa': hotpot, 'qasper': {}}
    for split in ('train', 'dev', 'test'):
        papers, provenance = qasper.load_papers(split, args.offline)
        result['qasper'][split] = {**census_qasper(papers, tokenizer), 'provenance': provenance}
        s = result['qasper'][split]
        print(f"QASPER {split}: {s['all_questions']} total; strict={s['strict_boolean']}; text_verified={s['text_verified']}", flush=True)
    print('HotpotQA:', hotpot, flush=True)
    write_json(ROOT/'docs/dataset_inventory.json', result)


if __name__ == '__main__':
    main()
