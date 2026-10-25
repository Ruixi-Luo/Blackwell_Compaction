"""配对概率 -> 分数、双向 InfoGap 和按上下文分组的 bootstrap。"""
from blackwell.metrics import forecast_scores, infogap, bootstrap_gaps, payoff_curve


def analyze(rows, config):
    if len(rows) != config["sample_size"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate samples: refusing to report a silently reduced dataset.")
    methods = config["methods"]
    if any(set(r["conditions"]) != set(methods) for r in rows):
        raise ValueError("Conditions must match the configured methods on every sample.")
    labels = [r["label"] for r in rows]
    groups = [r["context_group"] for r in rows]
    probabilities = {method: [r["conditions"][method]["p_yes"] for r in rows] for method in methods}
    scores = {}
    for method in methods:
        conditions = [r["conditions"][method] for r in rows]
        # 论文的多个问题复用同一压缩结果；耗时不能按题目重复累计。
        unique_compressions = {c['local_compression_hash']: c['compression_seconds']
                               for c in conditions if 'local_compression_hash' in c}
        scores[method] = {
            **forecast_scores(probabilities[method], labels),
            "mean_context_tokens": sum(c["tokens"] for c in conditions) / len(rows),
            "pooled_retained_ratio": sum(c["tokens"] for c in conditions) / sum(r["original_tokens"] for r in rows),
            "over_budget_samples": sum(c["budget_met"] is False for c in conditions),
            "length_repairs": sum(c["length_repair_used"] for c in conditions),
            "unique_compressions": len(unique_compressions),
            "compression_seconds_unique_total": sum(unique_compressions.values()),
        }
    for method in methods:
        for metric in ('accuracy', 'brier', 'log_loss'):
            scores[method][metric + '_delta_vs_full'] = scores[method][metric] - scores['full'][metric]
    matrix = {a: {b: infogap(probabilities[a], probabilities[b], labels) for b in methods} for a in methods}
    uncertainty = {method: bootstrap_gaps(
        probabilities["full"], probabilities[method], labels, groups,
        config["bootstrap_repetitions"], config["bootstrap_seed"],
    ) for method in methods if method != "full"}
    curves = {m: payoff_curve(probabilities['full'], probabilities[m], labels)
              for m in methods if m != 'full'}
    subsets = {}
    for task in sorted({r['source_task'] for r in rows if 'source_task' in r}):
        indices = [i for i, r in enumerate(rows) if r.get('source_task') == task]
        y = [labels[i] for i in indices]
        p = {m: [probabilities[m][i] for i in indices] for m in methods}
        subsets[task] = {'n': len(indices), 'positive_labels': sum(y),
                         'scores': {m: forecast_scores(p[m], y) for m in methods},
                         'infogap_vs_full': {m: {'full_over_method': infogap(p['full'], p[m], y),
                                                'method_over_full': infogap(p[m], p['full'], y)}
                                             for m in methods if m != 'full'}}
    return {"n": len(rows), "positive_labels": sum(labels), "scores": scores,
            "context_clusters": len(set(groups)), "payoff_curves_vs_full": curves, "source_tasks": subsets,
            "infogap": matrix, "bootstrap_vs_full": uncertainty,
            "normalization": "2 * sup_t mean[(t-y)(1[p_A<=t]-1[p_B<=t])]; also save unscaled.",
            "scope": "Small pilot of verbalized probabilities, not true-posterior recovery or a dominance test."}
