"""从已完成、审计通过的运行生成可分享双语总览；不调用 API。"""
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table, decision_curve
from blackwell.storage import read_json, write_json, file_hash
from tools.run_current_suite import DEFAULT_CONFIGS


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
            shutil.copyfile(run / source, target)
            copied[target.name] = file_hash(target)
        records[name] = {'run_name':config['run_name'], 'metrics':metrics, 'api_usage':usage,
                         'api_attempts':len(ledger), 'audit':audit, 'replay':replay,
                         'dataset':manifest['identity']['dataset'], 'config':config,
                         'runtime':manifest['identity']['local_compressors'], 'files_sha256':copied}
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
        timing = sum(s['compression_seconds_unique_total'] for s in metrics['scores'].values())
        panels.append(f'<section class="dataset" id="d{index}"'+(' hidden' if index else '')+'><h2>'+html.escape(title)+'</h2>'+table(
            ['Method','Accuracy ↑','Brier ↓','Δ Brier ↓','Log loss ↓','G(Full, method)','G(method, Full)',bi('Retained','保留率')],score_rows)+
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
    options=''.join(f'<option value="d{i}">{html.escape(records[Path(c).stem]["dataset"]["name"])}</option>' for i,c in enumerate(DEFAULT_CONFIGS))
    explanation = '<section><h2>'+bi('How to read the experiment','如何阅读实验')+'</h2><p>'+bi(
        'G(Full, C) is the largest empirical decision-payoff advantage of full context over compressor C; G(C, Full) measures the reverse advantage. Both may be positive. Brier and log loss evaluate probabilistic forecasts; accuracy uses a 0.5 threshold. A nominal 50% retention setting does not imply equal realized token budgets.',
        'G(原文,C) 是原文相对压缩方法 C 的最大经验决策收益优势，G(C,原文) 衡量反向优势；两者可能同时大于零。Brier 和 log loss 评价概率预测，Accuracy 使用 0.5 阈值。名义保留率相同，不代表实际 token 预算相同。')+'</p><p>'+bi(
        'HotpotQA is the larger multi-paragraph pilot, not a long-context benchmark. QASPER uses full papers from train and a conservative text-evidence filter (192 Yes / 91 No); dev/test are reserved. LongBench is a 52-item binary subset of the standard English release; its pooled number describes this task mixture, not the official full benchmark. Dataset results remain separate.',
        'HotpotQA 是样本较多的多段落探索实验，不是长上下文 benchmark。QASPER 使用 train 全文及保守的文字证据筛选（192 Yes / 91 No），dev/test 保留。LongBench 是标准英文版的 52 题二元子集，其汇总值描述该任务混合，不是官方完整 benchmark 分数。各数据集分别报告。')+'</p><p>'+bi(
        'All numbers use one fixed API reader and its verbalized p(Yes), not verified Bayesian posteriors. Compressors never receive questions or labels. The experiment measures forecast usefulness on these question distributions; it does not prove universal Blackwell informativeness of the compressed text.',
        '所有结果使用同一个 API reader 直接报告的 p(Yes)，不预设为真实贝叶斯后验。压缩器看不到问题或标签。本实验测量这些问题分布下的预测价值，不能证明压缩文本对所有未来问题的 Blackwell 信息性。')+'</p></section>'
    guide = '<section><h2>'+bi('Code reading guide','代码阅读导引')+'</h2>'+table(
        [bi('Step','步骤'),bi('File','文件'),bi('Responsibility','职责')],[
        ['1','configs/*.json',bi('Choose dataset, sample count, methods, device, rate and API cap.','指定数据集、题数、方法、设备、保留率和调用上限。')],
        ['2','blackwell/data.py → datasets/*.py',bi('Normalize each source into context, question, label and context group.','各来源统一为上下文、问题、标签和上下文分组。')],
        ['3','blackwell/compressors/',bi('Local text-to-text compressors; explicit cpu/cuda device, no query input.','本地文本输入/输出压缩器；显式选择 cpu/cuda，不接收 query。')],
        ['4','run_experiment.py',bi('Prepare → compress → predict → analyze → report.','准备数据 → 压缩 → 预测 → 分析 → 报告。')],
        ['5','blackwell/metrics.py',bi('Exact threshold scan, both directional gaps and probability scores.','精确扫描阈值，计算双向差距和概率评分。')],
        ['6','tools/run_current_suite.py',bi('Orchestrate existing steps: one GPU compression job, overlapping API readers.','串联已有步骤：一次一组 GPU 压缩，API 预测可重叠运行。')],
        ['7','tools/audit_run.py / verify_offline_replay.py',bi('Inspect actual requests and replay cached results with network/model loading blocked.','核验实际请求，并在禁止网络和加载模型的情况下重放缓存。')]])+'<p>'+bi(
        'Full local records live in runs/&lt;run_name&gt;/. Shareable reports and probability tables live in docs/Experiments_20261018/. Old runs are preserved. CPU startup runs were interrupted for the CUDA migration and are not part of these result tables.',
        '完整本地记录位于 runs/&lt;run_name&gt;/，可分享报告和概率表位于 docs/Experiments_20261018/。旧实验保留。切换 CUDA 前中止的 CPU 试运行不计入这些结果表。')+'</p></section>'
    page = '<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell Compaction · Three-dataset experiment</title><style>'+STYLE+'[hidden]{display:none!important}</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Compaction and decision informativeness','上下文压缩与决策信息性')+'</h1><p>'+bi('Three datasets · Four paired conditions · Nominal 50% retention · CUDA compression','三个数据集 · 四组配对输入 · 名义保留率 50% · CUDA 压缩')+'</p></header><section>'+table(
        ['Dataset','Questions','Contexts',bi('Mean context tokens','平均上下文 token'),'Yes / No','API attempts'],overview)+'<p><label>'+bi('Inspect dataset: ','查看数据集：')+'<select onchange="selectDataset(this.value)">'+options+'</select></label></p></section>'+''.join(panels)+explanation+guide+'<section><h2>'+bi('Sources','来源')+'</h2><p><a href="https://hotpotqa.github.io/">HotpotQA · CC BY-SA 4.0</a> · <a href="https://aclanthology.org/2021.naacl-main.365/">QASPER · CC BY 4.0</a> · <a href="https://github.com/THUDM/LongBench">LongBench</a> · <a href="https://arxiv.org/html/2507.12094v2">InfoGap</a></p></section><script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}function selectDataset(x){document.querySelectorAll(".dataset").forEach(el=>el.hidden=el.id!==x)}</script></body></html>'
    (destination / 'Experiment_Overview.html').write_text(page, encoding='utf-8')
    print(destination / 'Experiment_Overview.html')


if __name__ == '__main__':
    main()
