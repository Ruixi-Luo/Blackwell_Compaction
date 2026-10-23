"""从已完成、审计通过的运行生成可分享双语总览；不调用 API。"""
import argparse
import html
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table, decision_curve
from blackwell.storage import read_json, write_json, file_hash
from tools.run_current_suite import DEFAULT_CONFIGS


MATRIX_STYLE = """
.gap-matrix{max-width:850px;border-collapse:separate;border-spacing:5px}
.gap-matrix caption{text-align:left;margin:8px 0;color:#5b6c7e}
.gap-matrix th{white-space:normal;line-height:1.4;font-size:14px}
.gap-matrix td{text-align:center;min-width:95px;padding:20px 12px;border:0;border-radius:7px;
font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.gap-matrix td:focus{outline:3px solid #b66818;outline-offset:1px}
.gap-legend{max-width:420px;margin:12px 0 20px;font-size:13px;color:#5b6c7e}
.gap-gradient{height:13px;border-radius:4px;background:linear-gradient(to right,rgb(243,247,252),rgb(22,75,131))}
.gap-ticks{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
"""


def infogap_matrix(metrics, scale_max):
    """用统一线性色阶展示 G(行, 列)，只读取指标，不重新计算实验结果。"""
    labels = {'full': bi('Full context', '原始上下文'),
              'no_context': bi('No context', '无上下文'),
              'selective_context': 'Selective Context', 'llmlingua2': 'LLMLingua-2',
              'salt_summary': 'SALT-summary'}
    methods = list(metrics['scores'])
    rows = []
    for a in methods:
        cells = []
        for b in methods:
            value = metrics['infogap'][a][b]['value']
            fraction = value / scale_max if scale_max else 0
            rgb = tuple(round(light + fraction * (dark - light))
                        for light, dark in zip((243, 247, 252), (22, 75, 131)))
            ink = '#ffffff' if fraction >= .6 else '#172c42'
            tooltip = html.escape(f'G({a}, {b}) = {value:.8f}')
            cells.append(f'<td data-row="{a}" data-column="{b}" data-gap="{value}" '
                         f'title="{tooltip}" aria-label="{tooltip}" tabindex="0" '
                         f'style="background:rgb{rgb};color:{ink}">{value:.4f}</td>')
        rows.append('<tr><th scope="row">' + labels[a] + '</th>' + ''.join(cells) + '</tr>')
    return ('<p>' + bi(
        'Row A, column B = G(A,B): the largest empirical decision-payoff advantage of A over B. '
        'Read the opposite cell for the reverse direction; both can be positive. Darker means a larger gap, not universal dominance.',
        '第 A 行、第 B 列＝G(A,B)：A 相对 B 的最大经验决策收益优势。反方向看对称位置，两者可同时为正。颜色越深代表 gap 越大，不代表在所有决策中都更优。') +
        '</p><div class="tablewrap"><table class="gap-matrix" data-scale-max="' + str(scale_max) + '"><caption>' +
        bi('Directional InfoGap · same methods on both axes', '双向 InfoGap · 横纵轴为同一组方法') +
        '</caption><thead><tr><th scope="col">' + bi('A (rows) / B (columns)', 'A（行）/ B（列）') +
        '</th>' + ''.join('<th scope="col">' + labels[m] + '</th>' for m in methods) +
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div><div class="gap-legend">' +
        '<div class="gap-gradient"></div><div class="gap-ticks"><span>0.0000</span>' +
        f'<span>{scale_max/2:.4f}</span><span>{scale_max:.4f}</span></div>' + bi(
        'Linear color scale shared across every dataset in this report (0 to observed maximum). Values are payoff gaps, not percentages.',
        '本报告所有数据集共用线性色阶（0 至观测最大值）。数值为收益差，不是百分比。') + '</div>')


def publish_detail_report(source, target, metrics, scale_max, provenance=''):
    """只替换分享副本中的矩阵；保留原运行报告与代码身份记录。"""
    methods = list(metrics['scores'])
    old_rows = [[a] + [f"{metrics['infogap'][a][b]['value']:.4f}" for b in methods] for a in methods]
    old_matrix = table(['A over B'] + methods, old_rows)
    page = source.read_text(encoding='utf-8')
    if page.count(old_matrix) != 1:
        raise ValueError('Expected exactly one directional InfoGap table in the detail report.')
    page = page.replace(old_matrix, infogap_matrix(metrics, scale_max))
    page = page.replace('</style>', MATRIX_STYLE + '</style>', 1)
    page = page.replace('</header>', '</header>' + provenance, 1)
    target.write_text(page, encoding='utf-8', newline='\n')


def publish_text(source, target):
    """报告副本统一为 LF，避免 Git 的行尾规范化改变已记录的校验值。"""
    target.write_text(source.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')


def api_accounting(run):
    """只合计指定运行的账本；不把历史调用计入补跑费用。"""
    ledger = read_json(run / 'api_attempts.json')
    usage = {k: sum(e.get('usage', {}).get(k, 0) for e in ledger)
             for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    return ledger, usage


def extension_note(receipt, destination):
    """合并结果必须交代时间差、旧预测复用与 SALT 的具体模式。"""
    previous = Path(os.path.relpath(ROOT / 'docs/Experiments_20261018/Experiment_Overview.html', destination)).as_posix()
    return '<section><h2>' + bi('SALT-summary extension', 'SALT-summary 补充实验') + '</h2><p>' + bi(
        'The four existing conditions retain their October 18 forecasts and compressed texts exactly. '
        'Only SALT-summary was added on October 23 with the same reader model name, prompts and examples. '
        'The API ledger counts this extension only; old compressor timings are historical measurements. '
        'The dates differ, so unobserved changes in the hosted reader remain a temporal confound.',
        '原有四组条件逐条保留 10 月 18 日的预测和压缩文本，仅在 10 月 23 日补充 SALT-summary。'
        'Reader 模型名称、提示词和题目保持一致。API 账本只统计补跑调用；旧压缩器耗时为历史记录。'
        '两次预测日期不同，托管 reader 的潜在变化仍构成时间混杂因素。') + '</p><p>' + bi(
        'SALT-summary uses the official legacy selector with query=None: topic coverage and sentence selection, '
        'followed by the original sentence order. Official cleaning, duplicate removal and sentence filtering are retained. '
        'The nominal 0.5 word budget uses cleaned, deduplicated sentences before aggressive filtering; it is not half of raw input words or Qwen tokens. '
        'The table measures actual retained Qwen tokens. Overlong sentences may be truncated for embedding scores, but selected original sentences are retained. '
        'We do not truncate the output to force an exact token ratio. '
        'This query-blind setting is not a reproduction of the paper’s query-aware QA tables.',
        'SALT-summary 使用官方 legacy 选择器，显式 query=None：根据主题覆盖选择句子，再按原文顺序输出。'
        '保留官方清洗、去重和句子过滤。原生 0.5 词数预算的分母为清洗、去重后且激进过滤前的句子，'
        '不等于原始词数或 Qwen token 数的一半。表中另报实际 Qwen token 保留率。'
        '超长句的 embedding 打分可能截断，但输出仍保留选中的原句，不靠事后截断强行对齐长度。'
        '这里是 query-blind 条件，不等于复现论文中利用 query 的 QA 主表。') + '</p><p>' + bi(
        'Frozen source run: ', '冻结的来源运行：') + '<code>' + html.escape(receipt['source_run']) + '</code></p>' + \
        f'<a href="{html.escape(previous)}">' + bi('Preserved four-condition report', '保留的四组条件报告') + '</a></section>'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', help='要发布的已完成运行配置；默认使用原三数据集配置。')
    parser.add_argument('--destination', help='报告目录；使用 --configs 时必须显式指定，避免覆盖历史结果。')
    args = parser.parse_args()
    if args.configs and not args.destination:
        parser.error('--configs requires --destination to preserve previous reports.')
    configs = args.configs or DEFAULT_CONFIGS
    destination = ROOT / (args.destination or 'docs/Experiments_20261018')
    if args.configs and destination.resolve() == (ROOT / 'docs/Experiments_20261018').resolve():
        parser.error('Custom runs must be published to a new destination.')
    return configs, destination


def main():
    configs, destination = parse_args()
    destination.mkdir(parents=True, exist_ok=True)
    panels, records, overview = [], {}, []
    questions = {}
    # 所有数据集共用观测最大值，避免相同颜色在不同页面表示不同的 gap。
    all_metrics = [read_json(ROOT / 'runs' / read_json(ROOT / c)['run_name'] / 'metrics.json')
                   for c in configs]
    scale_max = max(gap['value'] for metrics in all_metrics
                    for row in metrics['infogap'].values() for gap in row.values())
    for index, config_path in enumerate(configs):
        config = read_json(ROOT / config_path)
        name = Path(config_path).stem
        run = ROOT / 'runs' / config['run_name']
        manifest = read_json(run / 'manifest.json')
        metrics = read_json(run / 'metrics.json')
        audit = read_json(run / 'audit.json')
        replay = read_json(run / 'replay_verification.json')
        preflight = read_json(run / 'input_preflight.json')
        if manifest['status'] != 'complete' or audit['status'] != 'passed' or replay['status'] != 'passed':
            raise ValueError('Publish only completed, audited, replay-verified experiments.')
        rows = read_json(run / 'predictions.json')
        questions[name] = {' '.join(r['question'].lower().split()) for r in rows}
        ledger, usage = api_accounting(run)
        extension_path = run / 'extension_receipt.json'
        extension = read_json(extension_path) if extension_path.exists() else None
        provenance = extension_note(extension, destination) if extension else ''
        title = manifest['identity']['dataset']['name']
        copied = {}
        for source, suffix in [('report.html','.html'), ('predictions.csv','_predictions.csv'),
                               ('metrics.json','_metrics.json'), ('audit.json','_audit.json'),
                               ('replay_verification.json','_replay.json')]:
            target = destination / (name + suffix)
            if source == 'report.html':
                publish_detail_report(run / source, target, metrics, scale_max, provenance)
            else:
                publish_text(run / source, target)
            copied[target.name] = file_hash(target)
        records[name] = {'run_name':config['run_name'], 'metrics':metrics, 'api_usage':usage,
                         'api_attempts':len(ledger), 'api_statuses':dict(Counter(e['status'] for e in ledger)),
                         'audit':audit, 'replay':replay,
                         'dataset':manifest['identity']['dataset'], 'config':config,
                         'runtime':manifest['identity']['local_compressors'], 'files_sha256':copied}
        if extension:
            target = destination / (name + '_extension.json')
            publish_text(extension_path, target)
            copied[target.name] = file_hash(target)
            source_ledger, source_usage = api_accounting(ROOT / 'runs' / extension['source_run'])
            records[name]['extension'] = extension
            records[name]['historical_api_usage'] = source_usage
            records[name]['historical_api_attempts'] = len(source_ledger)
        receipt = run / 'continuation_receipt.json'
        if receipt.exists():
            target = destination / (name + '_continuation.json')
            publish_text(receipt, target)
            records[name]['continuation'] = read_json(receipt)
            copied[target.name] = file_hash(target)
        overview.append([html.escape(title),metrics['n'],metrics['context_clusters'],
                         f"{metrics['scores']['full']['mean_context_tokens']:,.0f}",
                         f"{metrics['positive_labels']} / {metrics['n']-metrics['positive_labels']}",len(ledger)])
        score_rows = []
        for method,s in metrics['scores'].items():
            score_rows.append([method.replace('salt_summary', 'SALT-summary'), f"{s['accuracy']:.3f}", f"{s['brier']:.4f}",
                               f"{s['brier_delta_vs_full']:+.4f}",f"{s['log_loss']:.4f}",
                               f"{s['pooled_retained_ratio']:.1%}"])
        interval_rows = [[m, f"{b['a_over_b'][0]:.4f}–{b['a_over_b'][1]:.4f}",
                          f"{b['b_over_a'][0]:.4f}–{b['b_over_a'][1]:.4f}"]
                         for m,b in metrics['bootstrap_vs_full'].items()]
        diagnostics = {}
        for method in metrics['scores']:
            probabilities = [r['conditions'][method]['p_yes'] for r in rows]
            diagnostics[method] = {
                'p_zero_or_one':sum(p in (0,1) for p in probabilities),
                'wrong_at_zero_or_one':sum(p == 1-r['label'] for p,r in zip(probabilities,rows)),
                'p_half':probabilities.count(.5), 'distinct_probabilities':len(set(probabilities))}
        records[name]['forecast_diagnostics'] = diagnostics
        diagnostic_html = '<details><summary>'+bi('Inspect probability granularity and extreme forecasts','检查概率粒度与极端预测')+'</summary>'+table(
            ['Method','p = 0 or 1',bi('Wrong at 0 or 1','0/1 概率且答错'),'p = 0.5',bi('Distinct values','不同概率值')],
            [[m,*d.values()] for m,d in diagnostics.items()])+'<p>'+bi(
                'These are verbalized probabilities with a finite observed granularity. Log loss is particularly sensitive to wrong 0/1 forecasts and the stated clipping convention. A value of 0.5 is counted as reported, not automatically interpreted as abstention.',
                '这些是模型直接报告的概率，在样本中呈有限粒度。Log loss 对答错的 0/1 预测及前述截断约定尤其敏感。0.5 按原始数值统计，不自动解释为拒答。')+'</p></details>'
        timing = sum(s['compression_seconds_unique_total'] for s in metrics['scores'].values())
        if extension:
            salt_minutes = metrics['scores']['salt_summary']['compression_seconds_unique_total'] / 60
            accounting = bi(
                f"New SALT reader calls: {len(ledger)} attempts, {usage['prompt_tokens']:,} input + {usage['completion_tokens']:,} output tokens. "
                f"New SALT compression: {salt_minutes:.1f} minutes. Reused old compressor timings: {timing/60-salt_minutes:.1f} minutes. "
                f"Historical four-condition source run: {len(source_ledger)} attempts and {source_usage['total_tokens']:,} tokens (excluded from new usage).",
                f"新增 SALT reader：{len(ledger)} 次尝试，输入 {usage['prompt_tokens']:,}、输出 {usage['completion_tokens']:,} tokens。"
                f"新增 SALT 压缩累计 {salt_minutes:.1f} 分钟；旧压缩器记录为 {timing/60-salt_minutes:.1f} 分钟，本轮未重跑。"
                f"历史四组来源运行有 {len(source_ledger)} 次尝试、{source_usage['total_tokens']:,} tokens，不计入补跑用量。")
        else:
            accounting = bi(
                f"API: {usage['prompt_tokens']:,} input + {usage['completion_tokens']:,} output tokens. Unique local compression: {timing/60:.1f} minutes, excluding initialization. {preflight['contexts_with_length_adapter']} contexts needed the recorded length adapter.",
                f"API：输入 {usage['prompt_tokens']:,}、输出 {usage['completion_tokens']:,} tokens。独立本地压缩累计 {timing/60:.1f} 分钟，不含初始化。{preflight['contexts_with_length_adapter']} 份上下文使用了已记录的长度适配。")
        panels.append(f'<section class="dataset" id="d{index}"'+(' hidden' if index else '')+'><h2>'+html.escape(title)+'</h2>'+table(
            ['Method','Accuracy ↑','Brier ↓','Δ Brier ↓','Log loss ↓',bi('Retained','保留率')],score_rows)+diagnostic_html+
            '<p>'+accounting+'</p>'+provenance+'<h3>'+bi('Directional InfoGap matrix','双向 InfoGap 矩阵')+'</h3>'+infogap_matrix(metrics, scale_max)+'<h3>'+bi('Descriptive uncertainty','描述性波动')+'</h3>'+table(
                         ['Method','G(Full, method): 95%','G(method, Full): 95%'],interval_rows)+
            '<p>'+bi(f"Paired context-cluster bootstrap, {config['bootstrap_repetitions']} repetitions. These percentile intervals describe pilot variability; they are not a validated dominance test.",
                     f"按上下文成组的配对 bootstrap，{config['bootstrap_repetitions']} 次。百分位区间描述探索实验的波动，不是经过验证的支配检验。")+'</p><div class="grid">'+''.join(
                '<div class="card">'+decision_curve(p,m)+'</div>' for m,p in metrics['payoff_curves_vs_full'].items())+'</div><p>'+bi(
                'Positive curve: full context has higher payoff; negative: the compared method has higher payoff. Each panel has its own vertical scale. Hover for threshold and jump-side values.',
                '曲线正值表示原文收益更高，负值表示对比方法更高。各图纵轴刻度独立。悬停查看阈值及跳跃左右侧取值。')+'</p>'+f'<p><a href="{name}.html">'+bi('Open full report and every example','打开完整报告与逐题案例')+f'</a> · <a href="{name}_predictions.csv">CSV</a> · <a href="{name}_metrics.json">Metrics JSON</a></p></section>')
    overlaps = [{'a':a,'b':b,'identical_normalized_questions':len(questions[a]&questions[b])}
                for a in questions for b in questions if a<b]
    write_json(destination / 'summary.json', {'runs':records,'cross_dataset_question_overlap':overlaps,
               'overlap_rule':'Lowercase and whitespace-normalized exact question matching only; not exhaustive semantic deduplication.'})
    publish_text(destination / 'summary.json', destination / 'summary.json')
    options=''.join(f'<option value="d{i}">{html.escape(records[Path(c).stem]["dataset"]["name"])}</option>' for i,c in enumerate(configs))
    first = next(iter(records.values()))
    runtime = first['runtime']
    has_salt = 'salt_summary' in first['metrics']['scores']
    method_count = len(first['metrics']['scores'])
    setup = '<section><h2>'+bi('Fixed experimental setup','固定实验设置')+'</h2>'+table(
        [bi('Component','组件'),bi('Setting','设置')],[
        ['API reader',html.escape(first['config']['model'])+' · temperature 0 · thinking off'],
        [bi('Local compression','本地压缩'),html.escape(runtime['hardware']['gpu_name'])+' · float32'],
        ['PyTorch / CUDA',html.escape(runtime['packages']['torch'])+' / '+html.escape(runtime['hardware']['cuda_build'])],
        ['Selective Context','GPT-2 · phrase-level selection'],
        ['LLMLingua-2','XLM-R-large · MeetingBank checkpoint']] + (
        [['SALT-summary', 'BGE-small-en-v1.5 · official legacy · query=None']] if has_salt else []))+ '<p>'+bi(
            'The text compressors run locally on GPU. The API reader sees the resulting context and question, and returns a JSON p_yes forecast. No model training or probability calibration is performed. In the SALT extension, the old four conditions are reused rather than recomputed.',
            '文本压缩器在本机 GPU 上运行；API reader 接收压缩后的上下文和问题，返回 JSON 格式的 p_yes。没有训练模型，也没有另做概率校准。在 SALT 补充实验中，原四组条件直接复用，不重新计算。')+'</p></section>'
    explanation = '<section><h2>'+bi('How to read the experiment','如何阅读实验')+'</h2><p>'+bi(
        'G(Full, C) is the largest empirical decision-payoff advantage of full context over compressor C; G(C, Full) measures the reverse advantage. Both may be positive. Brier and log loss evaluate probabilistic forecasts; accuracy uses a 0.5 threshold. A nominal 50% retention setting does not imply equal realized token budgets.',
        'G(原文,C) 是原文相对压缩方法 C 的最大经验决策收益优势，G(C,原文) 衡量反向优势；两者可能同时大于零。Brier 和 log loss 评价概率预测，Accuracy 使用 0.5 阈值。名义保留率相同，不代表实际 token 预算相同。')+'</p><p>'+bi(
        'HotpotQA is the larger multi-paragraph pilot, not a long-context benchmark. QASPER uses full papers from train and a conservative text-evidence filter (192 Yes / 91 No); dev/test are reserved. LongBench is a 52-item binary subset of the standard English release; its pooled number describes this task mixture, not the official full benchmark. Dataset results remain separate.',
        'HotpotQA 是样本较多的多段落探索实验，不是长上下文 benchmark。QASPER 使用 train 全文及保守的文字证据筛选（192 Yes / 91 No），dev/test 保留。LongBench 是标准英文版的 52 题二元子集，其汇总值描述该任务混合，不是官方完整 benchmark 分数。各数据集分别报告。')+'</p><p>'+bi(
        'All numbers use one API reader model name and its verbalized p(Yes), not verified Bayesian posteriors. Compressors never receive questions or labels. InfoGap is a normalized payoff gap, not bits or a percentage of information retained. The experiment measures forecast usefulness on these question distributions; it does not prove universal Blackwell informativeness of the compressed text.',
        '所有结果使用相同 API reader 模型名称直接报告的 p(Yes)，不预设为真实贝叶斯后验。压缩器看不到问题或标签。InfoGap 是归一化的收益差，不是比特数或信息保留百分比。本实验测量这些问题分布下的预测价值，不能证明压缩文本对所有未来问题的 Blackwell 信息性。')+'</p></section>'
    explanation += '<section><h2>'+bi('Question overlap across datasets','跨数据集问题重叠')+'</h2>'+table(
        ['A','B',bi('Identical questions','相同问题')],
        [[r['a'],r['b'],r['identical_normalized_questions']] for r in overlaps])+'<p>'+bi(
            'Exact matching after lowercasing and whitespace normalization; paraphrases are not detected. LongBench reuses source datasets, so these experiments are not three fully independent replications.',
            '仅做大小写和空白规范化后的精确匹配，不检测改写题。LongBench 复用来源数据集，因此这三组实验不能视为三次完全独立的重复验证。')+'</p></section>'
    explanation += '<section><h2>'+bi('Metric definitions','指标计算方法')+'</h2><pre>Brier = mean((p_yes − y)²)\nLog loss = −mean(y log(p_yes) + (1−y) log(1−p_yes))\nAccuracy = mean(1[p_yes ≥ 0.5] = y)\nG(A,B) = 2 sup_t mean((t−y)(1[p_A ≤ t] − 1[p_B ≤ t]))\nRetained ratio = sum(compressed Qwen tokens) / sum(original Qwen tokens)</pre><p>'+bi(
        'Only log loss clips probabilities to [1e−12, 1−1e−12] for finite logarithms. InfoGap and Brier use the original probabilities. Thresholds and their one-sided limits are evaluated exactly. The factor 2 follows Theorem 4.1 and the proof of Theorem 5.1 in arXiv v2; the displayed estimator in Theorem 5.1 omits it. We retain the unscaled result too, and do not claim this is an author-confirmed correction.',
        '只有 log loss 把概率截到 [1e−12,1−1e−12] 以避免无穷对数；InfoGap 和 Brier 使用原始概率。精确计算全部阈值断点与单侧极限。系数 2 按 arXiv v2 定理 4.1 和定理 5.1 的证明，定理 5.1 展示式未写该系数；同时保留未缩放结果，这并非作者确认的勘误。')+'</p></section>'
    guide = '<section><h2>'+bi('Code reading guide','代码阅读导引')+'</h2>'+table(
        [bi('Step','步骤'),bi('File','文件'),bi('Responsibility','职责')],[
        ['1','configs/*.json',bi('Choose dataset, sample count, methods, device, rate and API cap.','指定数据集、题数、方法、设备、保留率和调用上限。')],
        ['2','blackwell/data.py → datasets/*.py',bi('Normalize each source into context, question, label and context group.','各来源统一为上下文、问题、标签和上下文分组。')],
        ['3','blackwell/compressors/',bi('Local text-to-text compressors; explicit cpu/cuda device, no query input.','本地文本输入/输出压缩器；显式选择 cpu/cuda，不接收 query。')],
        ['4','run_experiment.py',bi('Prepare → compress → predict → analyze → report.','准备数据 → 压缩 → 预测 → 分析 → 报告。')],
        ['5','blackwell/metrics.py',bi('Exact threshold scan, both directional gaps and probability scores.','精确扫描阈值，计算双向差距和概率评分。')],
        ['6','tools/run_current_suite.py',bi('Run datasets sequentially through compression, prediction and audits; avoid shared API rate limits.','按数据集串行执行压缩、预测和审计，避免共享 API 配额限流。')],
        ['7','tools/audit_run.py / verify_offline_replay.py',bi('Inspect actual requests and replay cached results with network/model loading blocked.','核验实际请求，并在禁止网络和加载模型的情况下重放缓存。')]])+'<p>'+bi(
        'Full local records live in runs/&lt;run_name&gt;/. Shareable reports and probability tables live beside this overview; the previous four-condition snapshot remains in docs/Experiments_20261018/. Old runs are preserved. CPU startup runs were interrupted for the CUDA migration and are not part of these result tables. Initial overlapping API readers triggered HTTP 429; execution was serialized while preserving caches. The API ledger retains failed and interrupted attempts; all reported forecasts have validated responses.',
        '完整本地记录位于 runs/&lt;run_name&gt;/，可分享报告和概率表位于当前总览目录，原四组快照仍保留在 docs/Experiments_20261018/。旧实验保留。切换 CUDA 前中止的 CPU 试运行不计入这些结果表。最初并行 reader 触发 HTTP 429，随后保留缓存改为串行。API 账本保留失败和中断尝试，所有用于指标的预测均有核验通过的响应。')+'</p></section>'
    for name, record in records.items():
        if 'continuation' in record:
            c = record['continuation']
            guide += '<section><details><summary>'+bi('Continuation audit: ','续跑审计：')+html.escape(name)+'</summary><p>'+bi(
                f"Rate-limit retries exhausted the original attempt cap. A separate run preserved {c['source_complete_questions']} completed questions, {c['copied_api_responses']} API responses and the entire ledger. Only the run name and attempt cap changed ({c['source_config']['max_api_attempts']} → {c['continuation_config']['max_api_attempts']}); data, code, prompts, model and compression outputs stayed fixed.",
                f"限流重试耗尽原尝试上限后，单独建立续跑记录，保留 {c['source_complete_questions']} 道已完成题目、{c['copied_api_responses']} 份 API 响应和全部账本。仅运行名和尝试上限改变（{c['source_config']['max_api_attempts']} → {c['continuation_config']['max_api_attempts']}）；数据、代码、提示词、模型和压缩输出保持一致。")+f'</p><a href="{name}_continuation.json">'+bi('Verified cache transfer receipt','已核验的缓存转移记录')+'</a></details></section>'
    if has_salt:
        guide += '<section><h2>' + bi('Reading the SALT extension code', '如何阅读 SALT 补充代码') + '</h2><p>' + bi(
            'blackwell/compressors/salt_summary.py wraps the frozen official selector. tools/extend_with_salt.py copies each source row, appends only the SALT condition, and checks that the old conditions stay identical. The extension receipt records the source run and file hashes; the API ledger contains only new attempts.',
            'blackwell/compressors/salt_summary.py 封装固定版本的官方选择器。tools/extend_with_salt.py 逐条复制来源样本，只新增 SALT 条件，并核验旧条件完全一致。extension_receipt 记录来源运行及文件哈希，API 账本只包含新增尝试。') + '</p></section>'
    page = '<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell Compaction · Paired experiment</title><style>'+STYLE+MATRIX_STYLE+'[hidden]{display:none!important}</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Compaction and decision informativeness','上下文压缩与决策信息性')+'</h1><p>'+bi(f'{len(configs)} datasets · {method_count} paired conditions · Nominal 50% retention · CUDA compression',f'{len(configs)} 个数据集 · {method_count} 组配对输入 · 名义保留率 50% · CUDA 压缩')+'</p></header><section>'+table(
        ['Dataset','Questions','Contexts',bi('Mean context tokens','平均上下文 token'),'Yes / No',bi('New API attempts' if has_salt else 'API attempts','新增 API 尝试' if has_salt else 'API 尝试')],overview)+'<p><label>'+bi('Inspect dataset: ','查看数据集：')+'<select onchange="selectDataset(this.value)">'+options+'</select></label></p></section>'+''.join(panels)+setup+explanation+guide+'<section><h2>'+bi('Sources','来源')+'</h2><p><a href="https://hotpotqa.github.io/">HotpotQA · CC BY-SA 4.0</a> · <a href="https://aclanthology.org/2021.naacl-main.365/">QASPER · CC BY 4.0</a> · <a href="https://github.com/THUDM/LongBench">LongBench</a> · <a href="https://arxiv.org/html/2507.12094v2">InfoGap</a> · <a href="https://arxiv.org/abs/2607.17486">SALT paper</a> · <a href="https://github.com/oteomamo/SALT">SALT official code</a></p></section><script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}function selectDataset(x){document.querySelectorAll(".dataset").forEach(el=>el.hidden=el.id!==x)}</script></body></html>'
    (destination / 'Experiment_Overview.html').write_text(page, encoding='utf-8', newline='\n')
    print(destination / 'Experiment_Overview.html')


if __name__ == '__main__':
    main()
