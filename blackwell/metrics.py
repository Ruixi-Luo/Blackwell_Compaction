"""InfoGap 核心数学：不依赖 LLM、tokenizer 或网络，可单独审计。"""
import math
import random
from collections import defaultdict


def _validate(probabilities, labels):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Nonempty, equal-length prediction and label arrays are required.")
    if any(type(p) not in {int, float} or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite numbers in [0, 1].")
    if any(type(y) not in {int, bool} or y not in {0, 1} for y in labels):
        raise ValueError("Labels must be binary.")


def infogap(probabilities_a, probabilities_b, labels):
    """估计 G(A,B)：A 相对 B 的最大决策收益优势，通常不对称。

    h_t(p,y) = (t-p)_+ + (p-y) 1[p<=t] = (t-y) 1[p<=t]
    G(A,B) = 2 sup_t mean(h_t(p_A,y) - h_t(p_B,y)).

    系数 2 采用 arXiv:2507.12094v2 定理 4.1 和定理 5.1 的证明；
    定理 5.1 展示的 estimator 漏写/采用不同尺度，故同时保存 unscaled。
    在全部概率断点及左右极限求 supremum，不使用粗网格或四舍五入。
    """
    _validate(probabilities_a, labels)
    _validate(probabilities_b, labels)
    # 对每个概率断点，记录通过该点时线性函数 D(t)=slope*t+intercept 的变化。
    events = defaultdict(lambda: [0, 0])
    for a, b, y in zip(probabilities_a, probabilities_b, labels):
        events[a][0] += 1
        events[a][1] -= y
        events[b][0] -= 1
        events[b][1] += y
    events[0.0]
    events[1.0]
    slope, intercept = 0, 0
    best = {"unscaled": 0.0, "threshold": 0.0, "side": "constant_action"}
    n = len(labels)
    for threshold in sorted(events):
        # 在断点左边，尚未计入 p==threshold 的样本。
        if threshold > 0:
            value = (slope * threshold + intercept) / n
            if value > best["unscaled"]:
                best = {"unscaled": value, "threshold": threshold, "side": "left_limit"}
        slope += events[threshold][0]
        intercept += events[threshold][1]
        # 此处实现论文的 <= 约定；允许概率完全相同或等于 0/1。
        value = (slope * threshold + intercept) / n
        if value > best["unscaled"]:
            best = {"unscaled": value, "threshold": threshold, "side": "at_threshold"}
    return {"value": 2 * best["unscaled"], **best, "n": n, "normalization_factor": 2}


def forecast_scores(probabilities, labels):
    _validate(probabilities, labels)
    n = len(labels)
    # 仅 Log loss 使用 clipping；InfoGap、Brier 和原始输出均不改概率。
    clipped = [min(1 - 1e-12, max(1e-12, p)) for p in probabilities]
    return {
        "accuracy": sum(int(p >= 0.5) == y for p, y in zip(probabilities, labels)) / n,
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n,
        "log_loss": -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(clipped, labels)) / n,
    }


def bootstrap_gaps(a, b, labels, groups, repetitions=500, seed=20261017):
    """按上下文成组、两方法配对重采样；区间仅描述 pilot 波动，不证明支配。"""
    _validate(a, labels)
    _validate(b, labels)
    if len(groups) != len(labels) or repetitions < 2:
        raise ValueError("Invalid bootstrap settings.")
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    keys = list(members)
    rng = random.Random(seed)
    draws_ab, draws_ba = [], []
    for _ in range(repetitions):
        indices = [i for group in rng.choices(keys, k=len(keys)) for i in members[group]]
        sampled_a = [a[i] for i in indices]
        sampled_b = [b[i] for i in indices]
        sampled_y = [labels[i] for i in indices]
        draws_ab.append(infogap(sampled_a, sampled_b, sampled_y)["value"])
        draws_ba.append(infogap(sampled_b, sampled_a, sampled_y)["value"])

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    return {"a_over_b": interval(draws_ab), "b_over_a": interval(draws_ba),
            "repetitions": repetitions, "context_clusters": len(keys),
            "interpretation": "Descriptive paired cluster-bootstrap percentile interval, not a dominance test."}
