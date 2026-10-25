"""原文 -> 某一种压缩结果。只做各方法自己的长度控制，不相互对齐。"""
import math
from blackwell.storage import fingerprint, read_json, write_json
from blackwell.compressors import create_compressor
from .summary import generate_summary

LOCAL_METHODS = ('selective_context', 'llmlingua2', 'dac')


def compression_path(folder, context, method):
    """folder：某数据集目录；context+method 决定缓存文件名，配置由 manifest 冻结。"""
    return folder/'compressions'/f'{fingerprint({"context": context, "method": method})}.json'


def load_compression(folder, context, method, budget):
    """读取并校验结果；不存在返回 None，内容不一致则明确报错。"""
    path = compression_path(folder, context, method)
    if not path.exists():
        return None
    entry = read_json(path)
    expected = {'context': context, 'method': method, 'budget_tokens': budget}
    if entry['request'] != expected or fingerprint(entry['result']) != entry['result_sha256']:
        raise ValueError('Compression cache integrity check failed: '+str(path))
    return entry['result']


def save_compression(folder, context, method, budget, result, provenance=None):
    """result：完整压缩结果；provenance：可选的历史导入来源，不影响选择。"""
    entry = {'request': {'context': context, 'method': method, 'budget_tokens': budget},
             'result': result, 'result_sha256': fingerprint(result), 'provenance': provenance}
    write_json(compression_path(folder, context, method), entry)


def create_engine(method, device):
    """method：本地算法名；device：cpu/cuda；按需加载一个模型。"""
    if method == 'dac':
        from blackwell.compressors.dac import DACCompressor
        return DACCompressor(device=device)
    return create_compressor(method, device)


def local_candidates(context, budget, method, engine, tokenizer, retained_ratio=.5):
    """engine：已加载模型；只依据实测长度试探原生保留率，最多 8/4 次。"""
    original = tokenizer.count(context)
    low, high = .001, .999999
    rate = .975*retained_ratio if method == 'dac' else budget/original
    candidates = []
    for _ in range(4 if method == 'dac' else 8):
        text = engine.compress(context, retained_ratio=rate)
        count = tokenizer.count(text)
        candidates.append({'text': text, 'tokens': count, 'native_rate': rate})
        if math.ceil(.95*budget) <= count <= budget:
            break
        if method == 'dac':
            rate = min(.999, max(.001, rate*(.975*budget)/max(1, count)))
        else:
            if count > budget:
                high = rate
            else:
                low = rate
            rate = (low+high)/2
    return candidates


def select_local(candidates, budget):
    """选择最长完整合规候选；没有合规输出就失败，绝不裁剪。"""
    feasible = [c for c in candidates if c['tokens'] <= budget]
    if not feasible:
        raise ValueError('No local output fits the budget; no truncation is allowed.')
    return max(feasible, key=lambda c: c['tokens'])


def compress_one(context, budget, method, tokenizer, engine=None, client=None, retained_ratio=.5):
    """唯一公共压缩边界；只接收原文/预算/方法/执行依赖，不接收整条题目。"""
    if not isinstance(context, str) or not context.strip() or budget < 1:
        raise ValueError('Expected nonempty source text and a positive budget.')
    if method == 'llm_summary':
        selected = generate_summary(context, budget, client, tokenizer)
        candidates = [selected]
    else:
        candidates = local_candidates(context, budget, method, engine, tokenizer, retained_ratio)
        selected = select_local(candidates, budget)
    return {**selected, 'budget_tokens': budget, 'budget_met': selected['tokens'] <= budget,
            'length_repair_used': False, 'post_clip_used': False, 'matching_trim_used': False,
            'compression_seconds': 0., 'candidates': candidates,
            'policy': 'original_first_draft' if method == 'llm_summary' else 'native_length_search'}
