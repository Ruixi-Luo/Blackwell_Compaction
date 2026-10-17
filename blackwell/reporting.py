"""生成可离线分享的中英文结果页。所有图和数据都内嵌，不依赖 CDN。"""
import html
import json

from .storage import read_json


def bi(english, chinese):
    return f'<span lang="en">{english}</span><span lang="zh">{chinese}</span>'


STYLE = """
:root{font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;color:#243445;background:#f4f7fa;line-height:1.65}
body{max-width:1180px;margin:auto;padding:28px}header,section{background:white;padding:26px;margin-bottom:22px;border-radius:14px;border:1px solid #dce5ee}
h1{font-size:30px;line-height:1.3}h2{font-size:22px}h3{font-size:17px}button,select{font:inherit;padding:6px 12px;cursor:pointer}
html[data-language=en] [lang=zh],html[data-language=zh] [lang=en]{display:none}
.toolbar{float:right;display:flex;gap:8px}.note{padding:12px 16px;background:#edf4fb;border-left:4px solid #35789a;margin:12px 0}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #dce5ee;text-align:left;white-space:nowrap}th{background:#f0f5f8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{padding:15px;border:1px solid #dce5ee;border-radius:9px}
pre,code{font-family:Consolas,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:14px;font-size:13px}details{border-top:1px solid #dce5ee;padding:12px 0}summary{cursor:pointer;font-weight:600}
a{color:#176b97}svg{width:100%;height:auto}svg text{font-family:Segoe UI,Arial,sans-serif}.muted{color:#5b6c7e;font-size:14px}
@media(max-width:650px){body{padding:10px}header,section{padding:16px}.toolbar{float:none}h1{font-size:25px}}
"""


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def scatter(rows, method):
    points = []
    for row in rows:
        x = 42 + 210 * row["conditions"]["full"]["p_yes"]
        y = 245 - 210 * row["conditions"][method]["p_yes"]
        color = "#267c92" if row["label"] else "#bc7134"
        label = html.escape(f"{row['id']}: y={row['label']}, full={row['conditions']['full']['p_yes']}, {method}={row['conditions'][method]['p_yes']}")
        points.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}" opacity=".7"><title>{label}</title></circle>')
    return '<svg viewBox="0 0 300 295" role="img" aria-label="Probability comparison">' + '<path d="M42 35V245H252" stroke="#789" fill="none"/><path d="M42 245L252 35" stroke="#aab" stroke-dasharray="4 4"/>' + ''.join(points) + f'<text x="130" y="281" font-size="12">Full p(Yes)</text><text x="42" y="20" font-size="12">{method} p(Yes)</text><text x="32" y="262" font-size="11">0</text><text x="249" y="262" font-size="11">1</text><text x="26" y="39" font-size="11">1</text></svg>'


def write_report(run_dir, rows, metrics, config):
    methods = list(metrics["scores"])
    dataset = read_json(run_dir / "manifest.json")["identity"]["dataset"]
    dataset_name = html.escape(dataset["name"])
    ledger_path = run_dir / "api_attempts.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    usage = {key: sum(entry.get("usage", {}).get(key, 0) for entry in ledger) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    sections = []
    sections.append('<section><h2>' + bi('Experiment at a glance', '实验设置') + '</h2>' +
        table([bi('Item','项目'),bi('Value','取值')], [
            ['Model', html.escape(config['model']) + ' · SiliconFlow · thinking off · temperature 0'],
            ['Dataset', f"{dataset_name} {config['dataset_split']} · n={len(rows)} · Yes={metrics['positive_labels']} · seed={config['sample_seed']}"],
            [bi('Compaction budget','压缩预算'), bi('50% of original context tokens; original Qwen tokenizer; marker included.', '原文上下文 token 的 50%；使用 Qwen 官方 tokenizer；截断标记计入预算。') if config['retained_ratio']==0.5 else str(config['retained_ratio'])],
            [bi('API accounting','API 记录'), f"{len(ledger)} attempts · {usage['prompt_tokens']:,} input + {usage['completion_tokens']:,} output tokens"],
        ]) + '<p>' + bi('The scalar p_yes represents Bernoulli(p_yes). These are verbalized probability forecasts, not verified Bayesian posteriors or answer-token log probabilities.', '标量 p_yes 表示 Bernoulli(p_yes)。这里是模型直接报告的概率预测，不预设为真实贝叶斯后验，也不是答案 token 的 log probability。') + '</p></section>')
    if config.get("dataset") == "hotpotqa_yesno":
        sections.append('<section><h2>' + bi('HotpotQA input protocol', 'HotpotQA 输入协议') + '</h2><p>' +
            bi('Native yes/no questions from distractor dev. Every context includes all supplied paragraph titles and texts (usually 10) in the supplied order. Supporting facts are never used to select or reorder paragraphs. Compression receives no question or label. The supplied context itself was retrieved for the question; query-blind compression does not make this dataset query-independent.',
               '使用 distractor dev 的原生 yes/no 题。保留全部所提供段落的标题和正文（通常为 10 段）及其原始顺序，不用 supporting facts 筛选或重排。压缩器看不到问题和标签。数据集提供的上下文本身按题目检索，因此只能说压缩器 query-blind，不能说整个数据生成过程与 query 无关。') + '</p><p>' +
            bi('We evaluate binary probability forecasts with InfoGap, Brier and log loss. Accuracy is auxiliary. Official HotpotQA also evaluates answer EM/F1, supporting-fact EM/F1 and joint scores; this subset run is not a full official benchmark submission.',
               '本实验用 InfoGap、Brier 与 log loss 评估二元概率，Accuracy 为辅助。官方 HotpotQA 还评估答案 EM/F1、证据 EM/F1 及联合指标；这里的子集实验不等于完整官方 benchmark 成绩。') + '</p><a href="https://hotpotqa.github.io/">HotpotQA · CC BY-SA 4.0</a></section>')
    if "llmlingua2" in methods or "selective_context" in methods:
        sections.append('<section><h2>' + bi('Local compression methods', '本地压缩方法') + '</h2><p>' +
            bi('Selective Context: official GPT-2 / phrase filtering. LLMLingua-2: official XLM-R-large MeetingBank token classifier. Both run on CPU using frozen local weights; only the reader calls the API. Both receive only the context and a retained ratio, never the question or label.',
               'Selective Context 使用官方 GPT-2 及短语筛选；LLMLingua-2 使用官方 XLM-R-large MeetingBank 分类器。两者以固定本地权重在 CPU 上运行，只有 reader 调用 API。输入仅有上下文与保留率，不含问题和标签。') + '</p><p>' +
            bi('The native ratio applies to different lexical units. It is not an exact Qwen-token budget. We make one native compression call and report the actual Qwen-token count without post-truncation. This is a nominal-rate comparison, not a strictly matched-length comparison.',
               '两个方法的原生比例作用于不同词法单位，不等于精确的 Qwen token 预算。每条文本只调用一次原生压缩，保留原输出并报告实际长度。本实验比较相同名义保留率，不是严格等长对比。') + '</p><p><a href="https://github.com/liyucheng09/Selective_Context">Selective Context · official code</a> · <a href="https://github.com/microsoft/LLMLingua">LLMLingua-2 · official code</a></p></section>')
    receipt = run_dir / 'cache_import.json'
    if receipt.exists():
        reused = read_json(receipt)
        sections.append('<section><h2>' + bi('Reused predictions', '复用预测') + '</h2><p>' +
            bi(f"Imported {reused['count']} cached API responses from {html.escape(reused['source_run'])}. Reuse requires an identical complete API request. The API accounting above counts only attempts made in this new run; imported historical calls are excluded.",
               f"从 {html.escape(reused['source_run'])} 导入 {reused['count']} 条 API 响应缓存。只有完整请求完全一致才可复用。上方 API 用量只统计本次新增调用，不重复计算历史调用。") + '</p></section>')
    score_rows = []
    for method in methods:
        s = metrics['scores'][method]
        score_rows.append([method, f"{s['accuracy']:.3f}",f"{s['brier']:.4f}",f"{s['log_loss']:.4f}",f"{s['mean_context_tokens']:.1f}",f"{s['pooled_retained_ratio']:.1%}",s['over_budget_samples']])
    sections.append('<section><h2>' + bi('Forecasts and compression', '概率预测与压缩长度') + '</h2>' + table(['Method','Accuracy ↑','Brier ↓','Log loss ↓',bi('Context tokens','上下文 token'),bi('Actual retained','实际保留率'),bi('Over budget','超预算样本')],score_rows) + '<p class="muted">' + bi('Retained ratio is pooled compressed tokens / pooled original tokens. Accuracy uses p_yes ≥ 0.5. Over-budget outputs are flagged and remain in the table; they are not silently truncated.', '保留率＝全部压缩 token 之和／全部原文 token 之和。Accuracy 使用 p_yes ≥ 0.5。超预算输出会标记并保留，不偷偷截断。') + '</p></section>')
    matrix_rows = [[a] + [f"{metrics['infogap'][a][b]['value']:.4f}" for b in methods] for a in methods]
    sections.append('<section><h2>' + bi('Directional InfoGap', '有方向的 InfoGap') + '</h2><p>' + bi('Cell (row A, column B) is the estimated maximum payoff advantage of A over B. Read both directions. A larger cell is not a universal ranking or evidence of statistical significance.', '第 A 行、第 B 列＝A 相对 B 的最大决策收益优势估计。必须双向阅读；单个较大数值不是整体排名，也不代表统计显著。') + '</p>' + table(['A over B'] + methods, matrix_rows) +
        '<div class="note">' + bi('Normalization: G(A,B) = 2 supₜ mean[(t − y)(1[pA ≤ t] − 1[pB ≤ t])]. The factor 2 follows Theorem 4.1 and the proof of Theorem 5.1 in arXiv v2; the displayed estimator in Theorem 5.1 omits it. The JSON also preserves the unscaled value. This is our explicit implementation convention, not an author-confirmed correction.', '归一化：G(A,B) = 2 supₜ mean[(t − y)(1[pA ≤ t] − 1[pB ≤ t])]。系数 2 按 arXiv v2 定理 4.1 及定理 5.1 的证明；定理 5.1 展示的估计式未写该系数。JSON 同时保存未乘 2 的数值。这是本项目明确记录的实现约定，并非作者确认的勘误。') + '</div>' +
        '<p><a href="https://arxiv.org/html/2507.12094v2">Feng, Qian &amp; Tang · InfoGap paper</a></p></section>')
    intervals = []
    for method, b in metrics['bootstrap_vs_full'].items():
        intervals.append([method, f"{b['a_over_b'][0]:.3f}–{b['a_over_b'][1]:.3f}", f"{b['b_over_a'][0]:.3f}–{b['b_over_a'][1]:.3f}"])
    sections.append('<section><h2>' + bi('Pilot uncertainty', '小样本波动') + '</h2>' + table(['Method','G(Full, method)','G(method, Full)'],intervals) + '<p>' + bi(f"Descriptive 95% percentile ranges from {config['bootstrap_repetitions']} paired bootstrap draws, clustered by identical passage. The supremum statistic and small sample make these exploratory ranges; they are not a validated dominance test. This run does not support a general conclusion about which compressor is better.", f"按相同 passage 成组配对重采样 {config['bootstrap_repetitions']} 次，展示描述性 95% 百分位范围。由于 supremum 统计量与小样本的限制，这不是经过验证的支配检验。本次运行不能支持某种压缩方法普遍更好的结论。") + '</p></section>')
    sections.append('<section><h2>' + bi('Probability changes', '概率怎样变化') + '</h2><p>' + bi('Each dot is one item; diagonal = unchanged probability. Teal: gold Yes. Orange: gold No. Hover a dot for its sample ID.', '每个点是一道题；对角线表示概率未变化。青色是真实 Yes，橙色是真实 No。悬停查看样本编号。') + '</p><div class="grid">' + ''.join('<div class="card"><h3>'+m+'</h3>'+scatter(rows,m)+'</div>' for m in methods if m!='full') + '</div></section>')
    details = []
    for row in rows:
        body = '<p><b>Question:</b> ' + html.escape(row['question']) + f"<br><b>Gold:</b> {row['label']}</p>"
        body += table(['Method','p(Yes)','p(No)','Tokens','Budget'], [[m,f"{row['conditions'][m]['p_yes']:.4f}",f"{1-row['conditions'][m]['p_yes']:.4f}",row['conditions'][m]['tokens'],row['conditions'][m]['budget_tokens']] for m in methods])
        for method in methods:
            if method == 'no_context':
                continue
            body += f'<h3>{method}</h3><pre>{html.escape(row["conditions"][method]["text"])}</pre>'
        details.append(f'<details><summary>{html.escape(row["id"])} · {html.escape(row["question"])}</summary>{body}</details>')
    sections.append('<section><h2>' + bi('Inspect every sample', '逐条审计样本') + '</h2>' + ''.join(details) + '</section>')
    html_text = '<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell Compaction · Pilot results</title><style>'+STYLE+'</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Blackwell Compaction · Pilot results','Blackwell Compaction · 基础实验结果')+'</h1><p>'+bi('Query-blind compressors · One reader · Paired probability forecasts','不看 query 的压缩器 · 同一个 reader · 配对概率预测')+'</p><div class="note">'+bi('A small paired pilot; these results do not establish a general ranking or Blackwell dominance.', '这是小规模配对实验，结果不能确立方法的普遍排名或 Blackwell 支配关系。')+'</div></header>'+''.join(sections)+'<script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}</script></body></html>'
    (run_dir / 'report.html').write_text(html_text, encoding='utf-8')
