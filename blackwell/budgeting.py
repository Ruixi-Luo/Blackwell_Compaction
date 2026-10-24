"""只根据实际 Qwen token 数选择压缩结果；不读取问题或答案。"""
import math


def select_candidate(candidates, budget, tokenizer):
    """优先选择目标区间内最长候选；否则选距离预算最近者，必要时显式裁剪。"""
    in_band = [c for c in candidates if math.ceil(.95*budget) <= c['tokens'] <= budget]
    selected = (max(in_band, key=lambda c: c['tokens']) if in_band else
                min(candidates, key=lambda c: (abs(c['tokens']-budget), c['tokens'] > budget)))
    text = selected['text']
    clipped = tokenizer.count(text) > budget
    if clipped:
        text = tokenizer.head_tail(text, budget)
    tokens = tokenizer.count(text)
    return {'text': text, 'tokens': tokens, 'budget_tokens': budget,
            'budget_met': tokens <= budget, 'budget_fill': tokens / budget,
            'within_target_band': math.ceil(.95 * budget) <= tokens <= budget,
            'post_clip_used': clipped, 'selected_candidate': candidates.index(selected),
            'length_repair_used': len(candidates) > 1, 'compression_request_hashes': []}


def align_local(engine, context, budget, tokenizer, max_trials=8):
    """用最多 max_trials 次原生参数试探逼近预算；不假定输出长度严格单调。"""
    original = tokenizer.count(context)
    low, high = .001, .999999
    rate = min(high, max(low, budget / original))
    candidates = []
    for _ in range(max_trials):
        text = engine.compress(context, retained_ratio=rate)
        candidate = {'text': text, 'tokens': tokenizer.count(text), 'native_rate': rate}
        candidates.append(candidate)
        if math.ceil(.95 * budget) <= candidate['tokens'] <= budget:
            break
        if candidate['tokens'] > budget:
            high = rate
        else:
            low = rate
        rate = (low + high) / 2
    result = select_candidate(candidates, budget, tokenizer)
    return {**result, 'candidates': candidates,
            'native_retained_ratio': candidates[result['selected_candidate']]['native_rate']}
