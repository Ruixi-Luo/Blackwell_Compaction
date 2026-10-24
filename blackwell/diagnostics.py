"""预测概率与预算的描述性诊断；不拟合校准器，不改变原始 p_yes。"""
import math

from .metrics import forecast_scores


def probability_diagnostics(probabilities, labels, bins=10):
    """等宽 reliability bins / ECE / 极端错判 / 0.5 堆积；空箱保留但不参与 ECE。"""
    forecast_scores(probabilities, labels)  # 复用严格输入校验。
    if type(bins) is not int or bins < 1:
        raise ValueError('bins must be a positive integer')
    reliability = []
    for index in range(bins):
        indices = [i for i, p in enumerate(probabilities) if min(bins-1, int(p*bins)) == index]
        count = len(indices)
        mean_p = sum(probabilities[i] for i in indices)/count if count else None
        frequency = sum(labels[i] for i in indices)/count if count else None
        reliability.append({'lower': index/bins, 'upper': (index+1)/bins, 'n': count,
                            'mean_p_yes': mean_p, 'observed_yes_rate': frequency})
    clipped_losses = {}
    for epsilon in (1e-12, 1e-6, 1e-3):
        q = [min(1-epsilon, max(epsilon, p)) for p in probabilities]
        clipped_losses[str(epsilon)] = -sum(y*math.log(p)+(1-y)*math.log1p(-p) for p, y in zip(q, labels))/len(q)
    return {'n': len(labels), 'base_rate': sum(labels)/len(labels),
            'mean_p_yes': sum(probabilities)/len(labels),
            'ece_equal_width_10': sum(b['n']/len(labels)*abs(b['mean_p_yes']-b['observed_yes_rate']) for b in reliability if b['n']),
            'reliability': reliability, 'p_half': probabilities.count(.5),
            'p_zero_or_one': sum(p in (0, 1) for p in probabilities),
            'wrong_zero_or_one': sum(p == 1-y for p, y in zip(probabilities, labels)),
            'confident_wrong_95': sum((p >= .95 and y == 0) or (p <= .05 and y == 1) for p, y in zip(probabilities, labels)),
            'distinct_probabilities': len(set(probabilities)),
            'log_loss_sensitivity': clipped_losses}


def retention_diagnostics(conditions):
    ratios = sorted(c['retained_ratio_actual'] for c in conditions)
    return {'min': ratios[0], 'p10': ratios[int(.1*(len(ratios)-1))],
            'median': ratios[(len(ratios)-1)//2], 'p90': ratios[int(.9*(len(ratios)-1))], 'max': ratios[-1],
            'within_target_band': sum(c.get('within_target_band', False) for c in conditions),
            'post_clip_used': sum(c.get('post_clip_used', False) for c in conditions),
            'matching_trims': sum(c.get('matching_trim_used', False) for c in conditions),
            'zero_matched_budget': sum(c.get('matched_budget_tokens') == 0 for c in conditions),
            'underfilled': sum(c.get('budget_fill', 1) < .95 for c in conditions)}
