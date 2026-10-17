"""用真实运行结果生成导读；不调用模型，不读取密钥。"""
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table
from blackwell.storage import read_json, write_json


def p(en, zh):
    return '<p>' + bi(en, zh) + '</p>'


def heading(en, zh):
    return '<h2>' + bi(en, zh) + '</h2>'


def code(text):
    return '<pre>' + html.escape(text) + '</pre>'


def main():
    config = read_json(ROOT / 'configs' / 'pilot.json')
    run = ROOT / 'runs' / config['run_name']
    metrics = read_json(run / 'metrics.json')
    rows = read_json(run / 'predictions.json')
    manifest = read_json(run / 'manifest.json')
    ledger = read_json(run / 'api_attempts.json')
    audit = read_json(run / 'audit.json')
    if audit['status'] != 'passed':
        raise ValueError('Run audit has not passed.')
    if manifest['status'] != 'complete':
        raise ValueError('Only publish a completed run.')
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    shutil.copy2(run / 'report.html', docs / 'Pilot_Results.html')
    published_report = docs / 'Pilot_Results.html'
    attribution = '<section><p>Dataset: <a href="https://github.com/google-research-datasets/boolean-questions">BoolQ · Clark et al.</a>. Reproduced questions, passages and experimental compressed adaptations are shared under <a href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a>. Compression outputs are machine-generated adaptations.</p></section>'
    published_report.write_text(published_report.read_text(encoding='utf-8').replace('</body>', attribution + '</body>'), encoding='utf-8')
    shutil.copy2(run / 'metrics.json', docs / 'pilot_metrics.json')
    shutil.copy2(run / 'audit.json', docs / 'pilot_audit.json')
    shutil.copy2(run / 'predictions.csv', docs / 'pilot_predictions.csv')
    # 上传的是无需密钥的小型审计快照；原始 API 响应与环境不上传。
    write_json(docs / 'pilot_provenance.json', manifest)
    total_tokens = sum(x.get('usage', {}).get('total_tokens', 0) for x in ledger)
    sections = []

    def section(body):
        sections.append('<section>' + body + '</section>')

    section(heading('1 · What the experiment actually compares', '1 · 这个实验到底在比较什么') +
        p('Research question: with the reader fixed, how does query-blind context compaction change the usefulness of its binary probability forecasts?', '研究问题：固定 reader 后，不看后续 query 的上下文压缩，会怎样改变其二元概率预测的决策价值？') +
        '<div class="pipeline">' + bi('Passage X → {Full, No context, Head+tail, Summary} → Same reader + Question Q → p(Yes) → Join gold label y → InfoGap in both directions', '段落 X → {原文、空上下文、首尾截断、摘要} → 同一 reader + 问题 Q → p(Yes) → 接入真实标签 y → 双向 InfoGap') + '</div>' +
        p('A forecast p describes the distribution (P(No), P(Yes)) = (1−p, p). The model is asked to state p directly in JSON. We do not equate that number with a true Bayesian posterior, and we do not claim to recover the model’s internal beliefs.', '一个概率 p 即可表示分布 (P(No), P(Yes)) = (1−p, p)。模型直接用 JSON 报告 p。我们不把这个数预设为真实贝叶斯后验，也不声称恢复了模型内部的真实信念。') +
        p('The comparison concerns compactor + fixed reader + elicitation prompt on BoolQ. The reader may use background knowledge; the no-context control helps reveal this. Higher confidence alone does not imply higher informativeness.', '比较对象是 BoolQ 上的“压缩器 + 固定 reader + 概率提取提示词”。Reader 可以使用背景知识，无上下文对照帮助观察这一点。置信度更高本身不代表更有信息。'))

    section(heading('2 · The four conditions', '2 · 四组输入与两个 baseline') +
        table([bi('Condition','组别'),bi('Construction','构造方法'),bi('Role','作用')], [
            ['full',bi('Passage unchanged.','原始 passage 不变。'),bi('Reference forecast, not ground truth.','参考预测，不是真实答案。')],
            ['no_context',bi('Empty passage; the same question still reaches the reader.','上下文为空；reader 仍看到同一个问题。'),bi('Background-knowledge control.','背景知识对照。')],
            ['head_tail',bi('Keep prefix and suffix within the token budget; count the omission marker.','在 token 预算内保留首尾；省略标记也计入长度。'),bi('Simple deterministic baseline; no compressor API call.','简单确定性 baseline；不调用压缩 API。')],
            ['summary',bi('One query-blind API summary; at most one extra length repair.','一次不看 query 的 API 摘要；最多一次长度修复。'),bi('Generic abstractive baseline, not ACON reproduction.','通用生成式 baseline，不声称复现 ACON。')],
        ]) + p('The requested retained ratio is 0.5. Each item has budget max(1, floor(0.5 × original context tokens)). We use the pinned official Qwen tokenizer and count only the context text; provider usage separately includes prompts and questions.', '目标保留率为 0.5。每条样本预算是 max(1, floor(0.5 × 原文上下文 token 数))。使用固定版本的 Qwen 官方 tokenizer，只对上下文文本计数；服务商 usage 另包含提示词和问题。') +
        p('Summary length is a target rather than a guaranteed exact ratio. After one unsuccessful repair, keep and flag the over-budget summary. Never select a draft using the gold answer or downstream prediction. Head+tail obeys the hard budget.', '摘要长度是目标，不保证恰好达到同一保留率。一次修复后仍超预算，则保留并标记；不能根据真实答案或后续预测挑选摘要。首尾截断严格满足预算。'))

    section(heading('3 · Data and reproducibility', '3 · 数据从哪里来，如何复现') +
        p('The official SuperGLUE v2 BoolQ release is a 4.1 MB ZIP. The original Google download endpoint returned HTTP 403, so this implementation uses the public SuperGLUE distribution and checks its SHA-256. No model weights are downloaded.', '使用 SuperGLUE v2 官方 BoolQ 发行包，ZIP 约 4.1 MB。原始 Google 下载地址返回 HTTP 403，所以本实现改用 SuperGLUE 公开发行包，并核验 SHA-256。不下载模型权重。') +
        p(f"The main pilot uniformly samples {len(rows)} items without replacement from 3,270 dev items, using seed {config['sample_seed']}. Labels are not rebalanced. There are {metrics['positive_labels']} Yes labels in this sample. Two separate train items test the protocol; they do not enter the main metrics.", f"主实验从 3,270 条 dev 数据中均匀无放回抽取 {len(rows)} 条，seed={config['sample_seed']}，不重新平衡标签。本次样本有 {metrics['positive_labels']} 条 Yes。另用 2 条训练集样本检查协议，不纳入主实验指标。") +
        p('In this release, dev means BoolQ/val.jsonl. The source row number is zero-based and saved in examples.json. Passage is the only compressed field; title is excluded. The label is a native boolean, converted once to 0/1.', '这里 dev 对应 BoolQ/val.jsonl。原始行号从 0 开始，保存在 examples.json。只压缩 passage，不加入 title。原生 bool 标签仅转换一次为 0/1。') +
        p('The manifest records config, dataset hash, tokenizer revision/hash, sample fingerprint, Python version and every core Python source hash. The same run name rejects changes to code, config or data. Cached requests are keyed by exact API URL and payload.', 'manifest 记录配置、数据校验值、tokenizer 版本及校验值、样本指纹、Python 版本和每个核心 Python 文件的校验值。同一运行编号拒绝代码、配置或数据变化。API 缓存按完整 URL 与请求正文生成键。'))

    section(heading('4 · Reading the code in order', '4 · 建议按这个顺序读代码') +
        table([bi('File','文件'),bi('Read for','重点看什么')], [
            ['configs/pilot.json',bi('The frozen experimental choices.','固定的实验参数。')],
            ['run_experiment.py',bi('generate_predictions: where X, Q and y split and later rejoin.','generate_predictions：X、Q、y 在哪里分开，在哪里重新汇合。')],
            ['blackwell/prompts.py',bi('Exactly what each LLM call sees.','每次 LLM 调用到底能看到什么。')],
            ['blackwell/compression.py',bi('compress(context, budget, method), with no query argument.','compress(context, budget, method)，没有 query 参数。')],
            ['blackwell/metrics.py',bi('InfoGap direction, factor 2, thresholds and score formulas.','InfoGap 的方向、系数 2、阈值与辅助分数。')],
            ['blackwell/client.py',bi('Strict probability parsing, cache, retry and hard call cap.','严格概率解析、缓存、重试和调用上限。')],
            ['blackwell/data.py / tokenization.py',bi('Sampling and exact token-budget accounting.','固定抽样和 token 预算计算。')],
            ['tests/',bi('Independent payoff checks and failure-mode checks.','独立决策收益核验，以及失败情形测试。')],
        ]) + code('context = example["context"]\ncompressed = compactor.compress(context, budget, "summary")\n# Only after compression does the reader receive the question.\nmessages = prediction_messages(compressed["text"], example["question"])\np_yes = parse_probability(reader_output)\n# example["label"] is used only later, when evaluating predictions.') +
        p('API bodies are saved under api_cache. Open a summary request and inspect messages: there should be the passage and budget, but no appended question, label or full-context forecast. This is also a structural code invariant, not just a prompt instruction.', '实际 API 正文在 api_cache 中。打开摘要请求检查 messages：应只有 passage 和预算，没有额外附上的 question、label 或原文预测。这既是提示词要求，也是代码接口约束。'))

    section(heading('5 · InfoGap, step by step', '5 · InfoGap 怎样算') +
        code('h_t(p, y) = max(t − p, 0) + (p − y) × 1[p ≤ t]\n          = (t − y) × 1[p ≤ t]\n\nD_AB(t) = (1/n) Σ_i [h_t(p_i^A, y_i) − h_t(p_i^B, y_i)]\nG_hat(A, B) = 2 × sup_{t∈[0,1]} D_AB(t)') +
        p('For a candidate threshold t, evaluate the same decision rule for both forecast sets against the same labels. Search over all thresholds for the largest advantage of A. Reversing A and B answers a different question. The formula does not merely measure |pA−pB| or KL divergence.', '对候选阈值 t，两组预测使用相同决策规则、相同真实标签评估。遍历阈值寻找 A 的最大优势。交换 A 和 B 是另一个问题。这不是只计算 |pA−pB| 或 KL 散度。') +
        p('D is linear between consecutive observed probabilities, but can jump at those probabilities when forecasts are miscalibrated. The event-scan algorithm checks every breakpoint before and after adding tied observations. It also includes 0 and 1. The saved side identifies a left limit versus the value at a threshold.', 'D 在相邻预测概率之间是线性的，但失校准时会在概率断点处跳跃。事件扫描在每个断点计入相同概率样本之前和之后各计算一次，也包含 0 和 1。保存的 side 区分左极限和阈值处的值。') +
        '<div class="note">' + bi('Version convention: Theorem 4.1 and the proof of Theorem 5.1 in arXiv:2507.12094v2 use factor 2. The estimator displayed in Theorem 5.1 omits it. This code follows the former and stores both scaled value and unscaled. This discrepancy still requires confirmation against the authors’ final version.', '版本约定：arXiv:2507.12094v2 的定理 4.1 及定理 5.1 证明含系数 2，定理 5.1 展示的 estimator 未写 2。代码采用前者，并同时保存 value 与 unscaled。该差异仍需对照作者最终版本确认。') + '</div>' +
        p('Sanity example: gold labels [0,1], perfect forecasts [0,1], uninformative forecasts [0.5,0.5]. G(perfect, uninformative)=0.5; reverse=0. Tests independently compute actual utility differences for two-action decision problems, rather than only repeating the estimator’s implementation.', '手算检查：标签 [0,1]，完美预测 [0,1]，无信息预测 [0.5,0.5]。G(完美, 无信息)=0.5，反向为 0。测试还独立计算两动作决策的实际收益差，而不是仅重复估计器代码。') +
        p('We do not calibrate forecasts first. We retain probabilities at their original precision. True labels are necessary: prediction-only access in this paper still supplies prediction–label pairs.', '不先对概率做校准；保留原始数值精度。真实标签仍然必要：论文的 prediction-only access 仍提供“预测—标签”样本对。'))

    section(heading('6 · Supporting metrics and uncertainty', '6 · 辅助指标与不确定性') +
        code('Brier = mean((p − y)^2)\nLog loss = −mean[y log(p) + (1−y) log(1−p)]\nAccuracy = mean(1[(p ≥ 0.5) == y])') +
        p('Only log loss clips p to [10⁻¹², 1−10⁻¹²] to avoid log(0). InfoGap and Brier use the original values. Accuracy loses probability information and remains a supporting result, not the main comparison.', '仅 Log loss 为避免 log(0)，把 p 限定到 [10⁻¹², 1−10⁻¹²]。InfoGap 与 Brier 使用原值。Accuracy 会丢失概率信息，所以只作为辅助结果。') +
        p('We draw paired bootstrap samples, grouping duplicate passages together. The 2.5%–97.5% ranges describe instability in this pilot. Because the statistic is a supremum and n is small, these are not a justified test of population Blackwell dominance. More data, multiple readers and a separate inference protocol are needed for paper-level claims.', '对样本做配对 bootstrap，相同 passage 成组抽样。2.5%–97.5% 范围描述 pilot 的波动。由于统计量是 supremum 且 n 很小，这些区间不能作为总体 Blackwell 支配的可靠检验。论文级结论还需要更大样本、多个 reader 和单独设计的统计推断协议。'))

    result_rows = []
    for method, scores in metrics['scores'].items():
        result_rows.append([method, f"{scores['accuracy']:.3f}", f"{scores['brier']:.4f}", f"{metrics['infogap']['full'][method]['value']:.4f}", f"{metrics['infogap'][method]['full']['value']:.4f}", f"{scores['pooled_retained_ratio']:.1%}"])
    section(heading('7 · What actually ran', '7 · 本地实际跑出了什么') +
        p(f"Completed: {len(rows)} dev items × 4 input conditions = {4*len(rows)} probability forecasts. The main run made {len(ledger)} API attempts and the provider reported {total_tokens:,} tokens, including retries where usage was returned. A separate 2-item train smoke test is excluded from this table.", f"已完成：{len(rows)} 条 dev 样本 × 4 组输入 = {4*len(rows)} 个概率预测。主实验共 {len(ledger)} 次 API 尝试，服务商返回的 usage 合计 {total_tokens:,} token，包含能获取 usage 的重试。单独的 2 条训练集检查不进入本表。") +
        table(['Method','Accuracy','Brier','G(Full, method)','G(method, Full)',bi('Actual retained','实际保留率')], result_rows) +
        '<div class="note">' + bi(f"Budget compliance: {metrics['scores']['summary']['over_budget_samples']}/{len(rows)} summaries still exceed their per-item 50% target after one repair. Their pooled retained ratio is {metrics['scores']['summary']['pooled_retained_ratio']:.1%}, versus {metrics['scores']['head_tail']['pooled_retained_ratio']:.1%} for head+tail. The full paired sample is retained; this is not a strictly matched-budget comparison.", f"预算执行情况：{len(rows)} 条摘要中有 {metrics['scores']['summary']['over_budget_samples']} 条在一次修复后仍超过单条 50% 的目标预算。摘要整体保留率为 {metrics['scores']['summary']['pooled_retained_ratio']:.1%}，首尾截断为 {metrics['scores']['head_tail']['pooled_retained_ratio']:.1%}。保留完整配对样本，不丢弃失败项；本次不能视为严格等预算比较。") + '</div>' +
        p(f"The audit checked {audit['forecasts']} saved forecasts and {audit['unique_api_responses_checked']} distinct API responses. It also independently evaluated two-action utilities for all 16 InfoGap cells, without making new API calls.", f"审计检查了 {audit['forecasts']} 个保存的预测和 {audit['unique_api_responses_checked']} 个不同的 API 响应，并独立计算两动作决策收益，核验全部 16 个 InfoGap 单元格；没有新增 API 调用。") +
        p('A lower Brier score or higher accuracy can coexist with a positive InfoGap in the other direction. The objectives are different; do not infer a universal ranking from one scalar. No winner is claimed from these 30 examples.', '较低的 Brier 或较高的 Accuracy 可以与反向正 InfoGap 同时出现；它们评价的目标不同。不能从一个标量推出普遍排序。本次 30 个样本不宣称方法胜负。') +
        p('Open <a href="Pilot_Results.html">the complete results page</a> for both-direction matrices, probability scatterplots and every original/compacted passage. The HTML is standalone; copied alone, the plots and examples still work.', '打开<a href="Pilot_Results.html">完整结果页</a>查看双向矩阵、概率散点图、每条原文和压缩文本。结果 HTML 自包含，单独复制也能查看图和案例。') +
        p('The free Qwen3-8B endpoint repeatedly rate-limited preliminary attempts, which are excluded. The final pilot uses paid Qwen3-30B-A3B-Instruct-2507 for every condition, with at least 1 second between requests and longer waits after HTTP 429. This version does not switch models on failure. Failed requests remain visible in the ledger. An exact currency charge was not obtained from the account endpoint; token counts are recorded instead.', '免费 Qwen3-8B 端点反复限流，初步尝试已排除。最终 pilot 的所有组统一使用付费 Qwen3-30B-A3B-Instruct-2507，请求至少间隔 1 秒，遇到 HTTP 429 再延长等待；失败时不切换模型。失败请求保留在账本中。账户接口未提供可核验的实际扣费，因此只记录 token 用量，不编造人民币金额。'))

    section(heading('8 · Running it yourself', '8 · 你自己怎么运行') +
        code('.\\run.ps1 -Mode Report   # recompute metrics, no LLM API\n.\\run.ps1 -Mode Replay   # replay saved responses, no network\n.\\run.ps1 -Mode Audit    # audit actual requests and payoffs, no LLM API\n.\\run.ps1 -Mode Test     # offline correctness checks\n.\\run.ps1 -Mode Prepare  # download public data/tokenizer only\n.\\run.ps1 -Mode Run      # real API calls when cache is missing') +
        p('On this computer the environment is already installed in .venv on D:. A fresh checkout requires .\\setup.ps1. If script execution is blocked, use powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Mode Report. The wrapper’s default action is Report, not a paid run.', '本机环境已安装在 D 盘仓库的 .venv 中。新克隆需要先执行 .\\setup.ps1。脚本被 Windows 阻止时，可用 powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Mode Report。包装脚本默认动作是 Report，不会默认发起付费运行。') +
        p('For real calls, set SILICONFLOW_API_KEY in the current environment, or edit a local .env copied from .env.example. This file is Git-ignored. The key is read only at the point of a network request and is never written into cache, manifest or reports. OpenAI Pro subscription quota does not pay for OpenAI API usage; this version does not use OpenAI.', '真实调用需要当前环境中的 SILICONFLOW_API_KEY，或复制 .env.example 为本地 .env 后填写。该文件被 Git 忽略。密钥只在发起请求时读取，不写入缓存、manifest 或报告。OpenAI Pro 订阅额度不能抵扣 OpenAI API 用量；本版没有使用 OpenAI。') +
        p('To expand n, copy the config and change both sample_size and run_name. The nominal call count is 5n: four reader calls plus one summary call per item, before length repair and network retries. Adjust max_api_attempts deliberately. It is a hard attempt cap, not an RMB billing guarantee.', '扩大样本量时，复制配置，同时修改 sample_size 和 run_name。基础调用量约为 5n：每例四次 reader、一次摘要，尚未计入长度修复和网络重试。请明确调整 max_api_attempts；它是请求次数硬上限，不是人民币账单保证。'))

    section(heading('9 · Audit questions and limits', '9 · 审计问题与边界') +
        table([bi('Check','检查点'),bi('Where to inspect','去哪里看')], [
            [bi('Does the compressor see Q or y?','压缩器能否看到 Q 或 y？'),'compression.py signature; prompts.py; api_cache request bodies'],
            [bi('Are all methods paired on the same items?','四种方法是否对应同一批样本？'),'examples.json IDs; predictions.json; analyze() completeness check'],
            [bi('Did invalid JSON become a fake probability?','无效 JSON 是否被填成假概率？'),'parse_probability() raises; no default p=0.5'],
            [bi('Are actual lengths reported?','是否报告实际压缩长度？'),'tokens, budget_met, length_repair_used, retained_ratio_actual'],
            [bi('Which direction and scale does InfoGap use?','InfoGap 用哪个方向和尺度？'),'metrics.py; value/unscaled/threshold/side in metrics.json'],
            [bi('Did a resumed run charge again?','恢复运行是否重复计费？'),'api_attempts.json; cache hits; --offline mode'],
            [bi('Did the code silently discard failures?','是否静默丢弃失败样本？'),'manifest status; incomplete runs cannot create a success report'],
        ]) +
        p('Limits: short contexts; small n; a single relatively small reader; verbalized probabilities can be coarse and miscalibrated; model-serving aliases can change; BoolQ may overlap pretraining data; one question per item does not establish query-universal informativeness. The scalar metrics do not recover a true posterior.', '局限：上下文较短、n 很小、只有一个较小的 reader；直接报告的概率可能粗糙且失校准；服务商模型别名可能更新；BoolQ 可能与预训练数据重叠；每条样本一个问题无法证明对所有 query 的 informativeness。标量指标不能恢复真实后验。') +
        p('Next useful additions are HotpotQA native yes/no examples, a stronger fixed reader, and a published query-agnostic compressor. Keep those as new, separately named experiments rather than retroactively changing this pilot.', '下一步适合加入 HotpotQA 原生 yes/no、一个更强的固定 reader，以及已发表的 query-agnostic 压缩方法。都应另建实验编号，不回头改写这次 pilot。'))

    source_blocks = []
    for relative in ['run_experiment.py','blackwell/prompts.py','blackwell/compression.py','blackwell/metrics.py','blackwell/client.py','blackwell/data.py','blackwell/tokenization.py','blackwell/storage.py','tools/audit_run.py','tests/test_metrics.py','tests/test_pipeline.py']:
        source = (ROOT / relative).read_text(encoding='utf-8')
        numbered = '\n'.join(f'{i:03d}  {line}'.rstrip() for i,line in enumerate(source.splitlines(),1))
        source_blocks.append(f'<details><summary>{html.escape(relative)}</summary>{code(numbered)}</details>')
    section(heading('10 · Expandable source snapshot', '10 · 可展开的源码快照') +
        p('These are the actual local source files when this guide was generated. Line numbers are for reading only. Read the first four modules before the transport and file helpers.', '以下是生成文档时的真实本地源码。行号仅用于阅读。建议先读前四个模块，再读网络与文件工具。') + ''.join(source_blocks))

    section(heading('References and data attribution', '参考来源与数据归属') + '<ul>' +
        '<li><a href="https://arxiv.org/html/2507.12094v2">Feng, Qian &amp; Tang · Is This Predictor More Informative than Another?</a></li>' +
        '<li><a href="https://github.com/google-research-datasets/boolean-questions">Clark et al. · BoolQ (NAACL 2019), CC BY-SA 3.0</a></li>' +
        '<li><a href="https://super.gluebenchmark.com/">SuperGLUE</a></li>' +
        '<li><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe">Qwen3-30B-A3B-Instruct-2507 tokenizer revision</a></li>' +
        '<li><a href="https://docs.siliconflow.cn/docs/api/chat-completions-post">SiliconFlow Chat Completions</a></li></ul>' +
        p('BoolQ excerpts and experimental compressed adaptations retain the dataset’s CC BY-SA 3.0 attribution. This does not assign a license to repository code.', 'BoolQ 原文与实验性压缩改写保留数据集的 CC BY-SA 3.0 归属；这不等于为仓库代码指定许可。'))

    extra_css = '.pipeline{padding:20px;background:#edf4fb;border-radius:10px;font-size:17px;font-weight:600}section pre{max-height:640px;overflow:auto}header p{max-width:880px}'
    document = '<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell Compaction · Code Walkthrough & Audit</title><style>'+STYLE+extra_css+'</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Blackwell Compaction<br>Code Walkthrough &amp; Audit','Blackwell Compaction<br>代码导读与审计')+'</h1>'+p('Version 0.1 · Real API pilot · 17 October 2026','基础版本 0.1 · 已运行真实 API · 2026 年 10 月 17 日')+p('A reading guide for a new PhD researcher: follow one sample through the pipeline, then inspect the formulas and the actual source.','为刚开始写实验代码的研究者准备：先跟着一条样本看完整流程，再检查公式与实际源码。')+'</header>'+''.join(sections)+'<script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}</script></body></html>'
    (docs / 'Code_Walkthrough_and_Audit.html').write_text(document, encoding='utf-8')
    print(docs / 'Code_Walkthrough_and_Audit.html')


if __name__ == '__main__':
    main()
