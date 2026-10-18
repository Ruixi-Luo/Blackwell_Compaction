"""从已完成、审计通过的运行生成可分享双语总览；不调用 API。"""
import html
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table, decision_curve
from blackwell.storage import read_json, write_json, file_hash
from tools.run_current_suite import DEFAULT_CONFIGS


def publish_text(source, target):
    """报告副本统一为 LF，避免 Git 的行尾规范化改变已记录的校验值。"""
    target.write_text(source.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')


def main():
    destination = ROOT / 'docs/Experiments_20261018'
    destination.mkdir(parents=True, exist_ok=True)
    panels, records, overview = [], {}, []
    questions = {}
    for index, config_path in enumerate(DEFAULT_CONFIGS):
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
        ledger = read_json(run / 'api_attempts.json')
        usage = {k: sum(e.get('usage', {}).get(k, 0) for e in ledger)
                 for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        title = manifest['identity']['dataset']['name']
        copied = {}
        for source, suffix in [('report.html','.html'), ('predictions.csv','_predictions.csv'),
                               ('metrics.json','_metrics.json'), ('audit.json','_audit.json'),
                               ('replay_verification.json','_replay.json')]:
            target = destination / (name + suffix)
            publish_text(run / source, target)
            copied[target.name] = file_hash(target)
        records[name] = {'run_name':config['run_name'], 'metrics':metrics, 'api_usage':usage,
                         'api_attempts':len(ledger), 'api_statuses':dict(Counter(e['status'] for e in ledger)),
                         'audit':audit, 'replay':replay,
                         'dataset':manifest['identity']['dataset'], 'config':config,
                         'runtime':manifest['identity']['local_compressors'], 'files_sha256':copied}
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
            score_rows.append([method, f"{s['accuracy']:.3f}", f"{s['brier']:.4f}",
                               f"{s['brier_delta_vs_full']:+.4f}",f"{s['log_loss']:.4f}",
                               f"{metrics['infogap']['full'][method]['value']:.4f}",
                               f"{metrics['infogap'][method]['full']['value']:.4f}",
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
        panels.append(f'<section class="dataset" id="d{index}"'+(' hidden' if index else '')+'><h2>'+html.escape(title)+'</h2>'+table(
            ['Method','Accuracy ↑','Brier ↓','Δ Brier ↓','Log loss ↓','G(Full, method)','G(method, Full)',bi('Retained','保留率')],score_rows)+diagnostic_html+
            '<p>'+bi(f"API: {usage['prompt_tokens']:,} input + {usage['completion_tokens']:,} output tokens. Unique local compression: {timing/60:.1f} minutes, excluding initialization. {preflight['contexts_with_length_adapter']} contexts needed the recorded length adapter.",
                     f"API：输入 {usage['prompt_tokens']:,}、输出 {usage['completion_tokens']:,} tokens。独立本地压缩累计 {timing/60:.1f} 分钟，不含初始化。{preflight['contexts_with_length_adapter']} 份上下文使用了已记录的长度适配。")+'</p><h3>'+bi('Directional InfoGap and descriptive uncertainty','双向 InfoGap 与描述性波动')+'</h3>'+table(
                         ['Method','G(Full, method): 95%','G(method, Full): 95%'],interval_rows)+
            '<p>'+bi('Paired context-cluster bootstrap, 500 repetitions. These percentile intervals describe pilot variability; they are not a validated dominance test.',
                     '按上下文成组的配对 bootstrap，500 次。百分位区间描述探索实验的波动，不是经过验证的支配检验。')+'</p><div class="grid">'+''.join(
                '<div class="card">'+decision_curve(p,m)+'</div>' for m,p in metrics['payoff_curves_vs_full'].items())+'</div><p>'+bi(
                'Positive curve: full context has higher payoff; negative: the compared method has higher payoff. Each panel has its own vertical scale. Hover for threshold and jump-side values.',
                '曲线正值表示原文收益更高，负值表示对比方法更高。各图纵轴刻度独立。悬停查看阈值及跳跃左右侧取值。')+'</p>'+f'<p><a href="{name}.html">'+bi('Open full report and every example','打开完整报告与逐题案例')+f'</a> · <a href="{name}_predictions.csv">CSV</a> · <a href="{name}_metrics.json">Metrics JSON</a></p></section>')
    overlaps = [{'a':a,'b':b,'identical_normalized_questions':len(questions[a]&questions[b])}
                for a in questions for b in questions if a<b]
    write_json(destination / 'summary.json', {'runs':records,'cross_dataset_question_overlap':overlaps,
               'overlap_rule':'Lowercase and whitespace-normalized exact question matching only; not exhaustive semantic deduplication.'})
    publish_text(destination / 'summary.json', destination / 'summary.json')
    options=''.join(f'<option value="d{i}">{html.escape(records[Path(c).stem]["dataset"]["name"])}</option>' for i,c in enumerate(DEFAULT_CONFIGS))
    first = next(iter(records.values()))
    runtime = first['runtime']
    setup = '<section><h2>'+bi('Fixed experimental setup','固定实验设置')+'</h2>'+table(
        [bi('Component','组件'),bi('Setting','设置')],[
        ['API reader',html.escape(first['config']['model'])+' · temperature 0 · thinking off'],
        [bi('Local compression','本地压缩'),html.escape(runtime['hardware']['gpu_name'])+' · float32'],
        ['PyTorch / CUDA',html.escape(runtime['packages']['torch'])+' / '+html.escape(runtime['hardware']['cuda_build'])],
        ['Selective Context','GPT-2 · phrase-level selection'],
        ['LLMLingua-2','XLM-R-large · MeetingBank checkpoint']])+ '<p>'+bi(
            'The GPU runs the two text compressors locally. The API reader sees the resulting context and question, and returns a JSON p_yes forecast. No model training or probability calibration is performed in this run.',
            'GPU 在本机运行两个文本压缩器；API reader 接收压缩后的上下文和问题，返回 JSON 格式的 p_yes。本轮没有训练模型，也没有另做概率校准。')+'</p></section>'
    explanation = '<section><h2>'+bi('How to read the experiment','如何阅读实验')+'</h2><p>'+bi(
        'G(Full, C) is the largest empirical decision-payoff advantage of full context over compressor C; G(C, Full) measures the reverse advantage. Both may be positive. Brier and log loss evaluate probabilistic forecasts; accuracy uses a 0.5 threshold. A nominal 50% retention setting does not imply equal realized token budgets.',
        'G(原文,C) 是原文相对压缩方法 C 的最大经验决策收益优势，G(C,原文) 衡量反向优势；两者可能同时大于零。Brier 和 log loss 评价概率预测，Accuracy 使用 0.5 阈值。名义保留率相同，不代表实际 token 预算相同。')+'</p><p>'+bi(
        'HotpotQA is the larger multi-paragraph pilot, not a long-context benchmark. QASPER uses full papers from train and a conservative text-evidence filter (192 Yes / 91 No); dev/test are reserved. LongBench is a 52-item binary subset of the standard English release; its pooled number describes this task mixture, not the official full benchmark. Dataset results remain separate.',
        'HotpotQA 是样本较多的多段落探索实验，不是长上下文 benchmark。QASPER 使用 train 全文及保守的文字证据筛选（192 Yes / 91 No），dev/test 保留。LongBench 是标准英文版的 52 题二元子集，其汇总值描述该任务混合，不是官方完整 benchmark 分数。各数据集分别报告。')+'</p><p>'+bi(
        'All numbers use one fixed API reader and its verbalized p(Yes), not verified Bayesian posteriors. Compressors never receive questions or labels. InfoGap is a normalized payoff gap, not bits or a percentage of information retained. The experiment measures forecast usefulness on these question distributions; it does not prove universal Blackwell informativeness of the compressed text.',
        '所有结果使用同一个 API reader 直接报告的 p(Yes)，不预设为真实贝叶斯后验。压缩器看不到问题或标签。InfoGap 是归一化的收益差，不是比特数或信息保留百分比。本实验测量这些问题分布下的预测价值，不能证明压缩文本对所有未来问题的 Blackwell 信息性。')+'</p></section>'
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
        'Full local records live in runs/&lt;run_name&gt;/. Shareable reports and probability tables live in docs/Experiments_20261018/. Old runs are preserved. CPU startup runs were interrupted for the CUDA migration and are not part of these result tables. Initial overlapping API readers triggered HTTP 429; execution was serialized while preserving caches. The API ledger retains failed and interrupted attempts; all reported forecasts have validated responses.',
        '完整本地记录位于 runs/&lt;run_name&gt;/，可分享报告和概率表位于 docs/Experiments_20261018/。旧实验保留。切换 CUDA 前中止的 CPU 试运行不计入这些结果表。最初并行 reader 触发 HTTP 429，随后保留缓存改为串行。API 账本保留失败和中断尝试，所有用于指标的预测均有核验通过的响应。')+'</p></section>'
    for name, record in records.items():
        if 'continuation' in record:
            c = record['continuation']
            guide += '<section><details><summary>'+bi('Continuation audit: ','续跑审计：')+html.escape(name)+'</summary><p>'+bi(
                f"Rate-limit retries exhausted the original attempt cap. A separate run preserved {c['source_complete_questions']} completed questions, {c['copied_api_responses']} API responses and the entire ledger. Only the run name and attempt cap changed ({c['source_config']['max_api_attempts']} → {c['continuation_config']['max_api_attempts']}); data, code, prompts, model and compression outputs stayed fixed.",
                f"限流重试耗尽原尝试上限后，单独建立续跑记录，保留 {c['source_complete_questions']} 道已完成题目、{c['copied_api_responses']} 份 API 响应和全部账本。仅运行名和尝试上限改变（{c['source_config']['max_api_attempts']} → {c['continuation_config']['max_api_attempts']}）；数据、代码、提示词、模型和压缩输出保持一致。")+f'</p><a href="{name}_continuation.json">'+bi('Verified cache transfer receipt','已核验的缓存转移记录')+'</a></details></section>'
    page = '<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell Compaction · Three-dataset experiment</title><style>'+STYLE+'[hidden]{display:none!important}</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Compaction and decision informativeness','上下文压缩与决策信息性')+'</h1><p>'+bi('Three datasets · Four paired conditions · Nominal 50% retention · CUDA compression','三个数据集 · 四组配对输入 · 名义保留率 50% · CUDA 压缩')+'</p></header><section>'+table(
        ['Dataset','Questions','Contexts',bi('Mean context tokens','平均上下文 token'),'Yes / No','API attempts'],overview)+'<p><label>'+bi('Inspect dataset: ','查看数据集：')+'<select onchange="selectDataset(this.value)">'+options+'</select></label></p></section>'+''.join(panels)+setup+explanation+guide+'<section><h2>'+bi('Sources','来源')+'</h2><p><a href="https://hotpotqa.github.io/">HotpotQA · CC BY-SA 4.0</a> · <a href="https://aclanthology.org/2021.naacl-main.365/">QASPER · CC BY 4.0</a> · <a href="https://github.com/THUDM/LongBench">LongBench</a> · <a href="https://arxiv.org/html/2507.12094v2">InfoGap</a></p></section><script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}function selectDataset(x){document.querySelectorAll(".dataset").forEach(el=>el.hidden=el.id!==x)}</script></body></html>'
    (destination / 'Experiment_Overview.html').write_text(page, encoding='utf-8', newline='\n')
    print(destination / 'Experiment_Overview.html')


if __name__ == '__main__':
    main()
