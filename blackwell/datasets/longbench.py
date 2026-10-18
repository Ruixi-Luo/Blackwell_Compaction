"""LongBench v1 标准版：只筛原生一致英文 yes/no 答案，不改写题目。"""
import json
import random
import zipfile
from collections import Counter

from ..storage import ROOT, download, fingerprint

SOURCE_URL = 'https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip'
SOURCE_SHA256 = 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
# 标准版全部英文任务；不混入 LongBench-E 或 v2。
ENGLISH_TASKS = ('narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa',
                 'musique', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa',
                 'samsum', 'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p')


def binary_label(answers):
    """LongBench 的参考答案必须全为一致的 yes/no；混合或冲突答案排除。"""
    if not answers or any(not isinstance(a, str) for a in answers):
        return None
    labels = {a.strip().lower().strip('.!?。！？').strip() for a in answers}
    if labels == {'yes'}:
        return 1
    if labels == {'no'}:
        return 0
    return None


def load_longbench(split, sample_size, seed, offline=False):
    if split != 'test':
        raise ValueError('LongBench v1 standard release uses split=test.')
    path = download(SOURCE_URL, ROOT / 'data/longbench/data.zip', SOURCE_SHA256, offline)
    pool, counts = [], {}
    with zipfile.ZipFile(path) as archive:
        for task in ENGLISH_TASKS:
            matches = [name for name in archive.namelist() if name.rsplit('/', 1)[-1] == task + '.jsonl']
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous standard LongBench member: {task}')
            total, eligible = 0, 0
            with archive.open(matches[0]) as handle:
                # 二进制逐行：文本 splitlines() 会把正文内的 Unicode 分隔符当成新记录。
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    total += 1
                    label = binary_label(row['answers'])
                    if label is None:
                        continue
                    if not row['context'].strip() or not row['input'].strip():
                        raise ValueError('Empty eligible LongBench input.')
                    eligible += 1
                    pool.append({'id': f"longbench-{task}-{row['_id']}",
                                 'source_row_zero_based': index, 'source_task': task,
                                 'context': row['context'], 'context_group': fingerprint(row['context']),
                                 'question': row['input'], 'label': label})
            counts[task] = {'all_rows': total, 'eligible_rows': eligible}
    if len({e['id'] for e in pool}) != len(pool):
        raise ValueError('Duplicate LongBench IDs.')
    if not 1 <= sample_size <= len(pool):
        raise ValueError('sample_size exceeds the eligible LongBench pool.')
    indices = random.Random(seed).sample(range(len(pool)), sample_size)
    provenance = {
        'name': 'LongBench v1 English yes/no', 'split': split,
        'dataset_home': 'https://github.com/THUDM/LongBench', 'source_url': SOURCE_URL,
        'source_sha256': SOURCE_SHA256, 'license': 'MIT (LongBench release; retain source dataset attribution)',
        'available_rows': len(pool), 'task_counts': counts, 'sample_seed': seed,
        'selection_rule': 'All reference answers normalize to the same English yes/no. Standard v1 English tasks only; exclude E and v2.',
        'context_definition': 'Unchanged context field; question=input. No reference answers or evidence enter the reader context.',
        'scope': 'Binary subset probability protocol, not official full LongBench scoring. Pooled result describes this fixed task mixture; source tasks reported separately.',
    }
    return [pool[i] for i in indices], provenance
