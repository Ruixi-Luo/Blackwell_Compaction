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


def match_lengths(compacted, tokenizer):
    """把所有压缩条件对齐到共同可行长度（不超过原文 50%），保留 full/no_context 对照。"""
    methods = [m for m in compacted if m not in {'full', 'no_context'}]
    adjusted = {m: dict(c) for m, c in compacted.items()}
    if not methods:
        return adjusted
    for method in methods:
        adjusted[method]['before_match_tokens'] = adjusted[method]['tokens']
    # 字符边界裁剪可能比预算少一两个 token；逐次降低共同长度直至完全一致。
    # 每次未收敛迭代都严格减少非负整数 token 总数，因此循环一定终止。
    passes = 0
    while True:
        budget = min(adjusted[m]['tokens'] for m in methods)
        if all(adjusted[m]['tokens'] == budget for m in methods):
            break
        for method in methods:
            c = adjusted[method]
            c['text'] = tokenizer.head_tail(c['text'], budget)
            c['tokens'] = tokenizer.count(c['text'])
        passes += 1
    for method in methods:
        c = adjusted[method]
        c.update(matched_budget_tokens=budget, matching_passes=passes,
                 matching_trim_used=c['tokens'] != c['before_match_tokens'],
                 budget_fill=c['tokens']/c['budget_tokens'],
                 within_target_band=math.ceil(.95*c['budget_tokens']) <= c['tokens'] <= c['budget_tokens'])
    return adjusted
