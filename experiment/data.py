"""从公开数据加载样本池；无需任何历史实验目录。"""
import random
from blackwell.datasets.hotpotqa import load_pool
from blackwell.datasets.qasper import load_papers, convert_papers


def load_examples(name, size, seed, offline):
    """name：数据集；size：题数；seed：抽样种子；offline：禁止下载。"""
    if name == 'hotpotqa':
        pool, source = load_pool(offline)
    elif name == 'qasper':
        papers, source = load_papers('train', offline)
        pool, exclusions = convert_papers(papers, 'train', require_text_evidence=True)
        source = {**source, 'split': 'train', 'excluded': exclusions,
                  'selection_rule': 'All boolean labels agree; all evidence occurs in full text.'}
    else:
        raise ValueError('Unsupported dataset: '+name)
    if not 1 <= size <= len(pool):
        raise ValueError(f'{name}: requested {size}, eligible pool has {len(pool)}.')
    # 按 ID 排序后抽样，复现 mini100 的选题顺序，而非依赖旧预测文件。
    examples = random.Random(seed).sample(sorted(pool, key=lambda e: e['id']), size)
    return examples, {**source, 'eligible_pool': len(pool), 'sample_seed': seed,
                      'sampling': 'uniform sample without replacement from sorted IDs'}
