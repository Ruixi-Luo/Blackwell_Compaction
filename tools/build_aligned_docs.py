"""从通过审计的新运行生成双语 HTML；不调用 API、不修改预测概率。"""
import html
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table, decision_curve, scatter
from blackwell.storage import read_json, write_json

DESTINATION = ROOT / 'docs/Experiments_20261024_Aligned'
NAMES = {'full':'Full','no_context':'No context','head_tail':'Head–tail',
         'selective_context':'Selective Context','llmlingua2':'LLMLingua-2',
         'salt_summary':'SALT legacy / no query','llm_summary':'LLM summary / budget-adapted',
         'telegrapher':'Telegrapher v5 / adapted'}


def page(title, body):
    return ('<!doctype html><html lang="en" data-language="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+STYLE+
            '.heat td{text-align:center}.heat th{white-space:normal}.heat td{min-width:65px} .heat td:hover{outline:2px solid #214b78}.legend{height:12px;background:linear-gradient(to right,#f7fbff,#19558b);max-width:280px} '
            '</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+html.escape(title)+'</h1></header>'+body+
            '<script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}</script></body></html>')


def matrix(metrics, scale):
    methods = list(metrics['scores'])
    rows = []
    for a in methods:
        cells = ['<th>'+NAMES[a]+'</th>']
        for b in methods:
            gap = metrics['infogap'][a][b]['value']
            alpha = gap/scale if scale else 0
            rgb = tuple(round(c+(d-c)*alpha) for c,d in zip((247,251,255),(25,85,139)))
            color = 'white' if alpha > .65 else '#243445'
            cells.append(f'<td style="background:rgb{rgb};color:{color}" title="G({a},{b}) = {gap:.8f}">{gap:.4f}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    return ('<div class="tablewrap"><table class="heat"><tr><th>A over B</th>'+''.join('<th>'+NAMES[m]+'</th>' for m in methods)+
            '</tr>'+''.join(rows)+'</table></div><p>'+bi('Row A versus column B: larger means a larger empirical maximum payoff advantage of A. Read both directions; this is not a ranking or significance test.',
            '第 A 行相对第 B 列：数值越大表示 A 的最大经验决策收益优势越大。需双向阅读，不是整体排名或显著性检验。')+
            f'</p><div class="legend"></div><small>0 → {scale:.4f} · '+bi('shared scale across datasets','各数据集共用色阶')+'</small>')


def reliability_plot(data, name):
    dots = []
    for row in data['reliability']:
        if row['n']:
            x,y = 40+230*row['mean_p_yes'],265-230*row['observed_yes_rate']
            tooltip = html.escape(f"n={row['n']}, mean p={row['mean_p_yes']:.4f}, observed Yes={row['observed_yes_rate']:.4f}")
            dots.append(f'<circle cx="{x}" cy="{y}" r="{min(10,3+row["n"]**.5/3)}" fill="#267c92"><title>{tooltip}</title></circle>')
    return '<svg viewBox="0 0 310 305" role="img" aria-label="Reliability diagram"><path d="M40 35V265H270 M40 265L270 35" fill="none" stroke="#9aa"/>'+''.join(dots)+f'<text x="40" y="20" font-size="13">{html.escape(name)}</text><text x="100" y="297" font-size="12">Mean p(Yes)</text><text x="3" y="35" font-size="11">1</text><text x="3" y="265" font-size="11">0</text><text x="270" y="282" font-size="11">1</text></svg>'


def protocol_section():
    return '<section><h2>'+bi('What changed','本轮改进')+'</h2><p>'+bi(
        'Eight paired conditions; all forecasts were collected anew. The target cap is floor(0.5 × original Qwen tokens). The preferred band is 95–100% of this cap (about 47.5–50% of the original). Local methods use at most eight length-only native-rate trials. The formal baselines use one API rewrite per unique context and method; optional length-repair drafts are supported in code but disabled here. Final equality is enforced by the separate common-length step. A draft within the target band is preferred; otherwise select the closest measured length, favoring feasible drafts at ties. A selected overshoot is clamped with a common head–tail adapter and explicitly counted. Underfilled outputs are kept and reported; no padding, gold-based selection or silent sample removal.',
        '共八组配对输入，全部重新获取预测。共同上限为 floor(0.5×原文 Qwen tokens)，目标区间是上限的 95%–100%，约等于原文的 47.5%–50%。本地方法最多尝试八个原生保留率，只看长度；正式 baseline 每种方法对每份独立上下文只改写一次；代码保留可选的长度修订能力，本轮关闭。最终等长由单独的共同长度步骤保证。优先选目标区间内候选，否则选实测长度距预算最近者，同距时优先不超长；选中超长稿则使用共同首尾裁剪并明确计数。填不满的结果保留且报告，不补无意义内容，不用答案选候选，不悄悄删题。')+'</p><p>'+bi(
        'For the primary comparison, each context is then matched to the shortest feasible output among the six compressed conditions. Longer outputs are deterministically trimmed until their actual Qwen token counts are exactly equal. Full and no-context controls are unchanged. Local native algorithms are not rerun at this lower common budget: this experiment compares compressor-plus-trimming compositions, not native algorithms optimized directly for that budget. The common budget can be below 50% and varies by context; no query or label is used to choose it. Read the actual-length distribution, underfill and clamp counts alongside InfoGap. Summary and Telegrapher both use DeepSeek-V3.2 as the rewrite model; the reader remains Qwen3-30B-A3B-Instruct-2507. Rewrite requests contain no query. Telegrapher uses the pinned official v5 grammar with an explicit budget instruction; full-document input, DeepSeek deployment and length repairs are adaptations, not the original paper configuration.',
        '主比较还会按每份上下文在六种压缩条件中取最短可行长度，将更长输出确定性裁剪到实际 Qwen token 数完全相同。原文和空上下文对照不变。本轮不会在更低的共同预算下重跑原生算法，因此比较的是“压缩器＋裁剪”的组合，不是原生算法直接针对该低预算优化的表现。共同预算可能低于 50%，因上下文而异，确定它不使用 query 或标签。解读 InfoGap 时应同时查看实际长度分布、欠填和裁剪次数。摘要与 Telegrapher 均用 DeepSeek-V3.2 改写，reader 保持 Qwen3-30B-A3B-Instruct-2507；改写请求不含 query。Telegrapher 使用固定官方 v5 语法及额外预算指令；全文输入、DeepSeek 部署和长度修订属于适配，并非论文原配置。')+'</p><p>'+bi(
        'Compression receives only context and budget. The reader alone receives the evaluation question; labels are used only during evaluation. Shared paper contexts reuse the same compression. All six compressed conditions have the same token count per question, including the head–tail control. The 458 HotpotQA / 283 QASPER-train / 52 LongBench binary examples and order are frozen from the old runs. LongBench is a native binary subset, not its official full benchmark; 14 questions overlap HotpotQA and one overlaps QASPER train.',
        '压缩只接收 context 与预算；只有 reader 接收评测问题，标签只用于评估。同篇论文复用同一压缩文本。包括首尾截断在内的六种压缩条件在每题上 token 数完全相同。458 道 HotpotQA、283 道 QASPER train、52 道 LongBench 二元题沿用旧样本及顺序。LongBench 为原生二元子集，并非官方完整 benchmark；其中 14 题与 HotpotQA 重叠，1 题与 QASPER train 重叠。')+'</p></section>'


def method_panels(config, metrics, diagnostics, scale, usage, rows=None):
    methods = config['methods']
    score_rows, length_rows, diag_rows = [], [], []
    for m in methods:
        s = metrics['scores'][m]
        d = diagnostics[m]['probability']
        r = diagnostics[m]['retention']
        score_rows.append([NAMES[m],f"{s['accuracy']:.3f}",f"{s['brier']:.4f}",f"{s['log_loss']:.4f}",f"{s['pooled_retained_ratio']:.2%}"])
        before = r.get('selected_before_clamp_pooled_ratio')
        length_rows.append([NAMES[m],f'{before:.1%}' if before is not None else '—',f"{r.get('before_match_pooled_ratio',0):.1%}",f"{r['min']:.1%}",f"{r['p10']:.1%}",f"{r['median']:.1%}",f"{r['p90']:.1%}",f"{r['max']:.1%}",r['within_target_band'],r.get('before_match_underfilled',r['underfilled']),r['post_clip_used'],r.get('matching_trims',0)])
        diag_rows.append([NAMES[m],f"{d['mean_p_yes']:.3f}",f"{d['ece_equal_width_10']:.4f}",d['p_half'],d['p_zero_or_one'],d['wrong_zero_or_one'],d['confident_wrong_95'],d['distinct_probabilities']])
    body = '<section><h2>'+bi('Forecast performance','预测表现')+'</h2>'+table(['Method','Accuracy ↑','Brier ↓','Log loss ↓',bi('Actual retained','实际保留率')],score_rows)+'</section>'
    body += '<section><h2>'+bi('Directional InfoGap matrix','双向 InfoGap 矩阵')+'</h2>'+matrix(metrics,scale)+'</section>'
    body += '<section><details><summary>'+bi('Decision payoff across thresholds','展开：不同阈值下的决策收益差')+'</summary><p>'+bi('Positive values favor Full; negative values favor the compared method. The positive maximum and negative minimum recover the two InfoGap directions. Hover points to distinguish threshold values from left limits. Each panel has its own vertical scale.','正值表示原文收益更高，负值表示对比方法收益更高；正向最大值与负向最小值对应双向 InfoGap。悬停可区分断点值与左极限。各图纵轴刻度不同。')+'</p><div class="grid">'+''.join('<div class="card">'+decision_curve(points,NAMES[m])+'</div>' for m,points in metrics['payoff_curves_vs_full'].items())+'</div></details></section>'
    body += '<section><h2>'+bi('Actual budget diagnostics','实际预算诊断')+'</h2>'+table(['Method',bi('Native draft','原生草稿'),bi('Before matching','共同对齐前'),'Min','P10','Median','P90','Max',bi('Final in band','最终落在目标区间'),bi('Pre-match below band','等长对齐前欠填'),bi('Initial cap trims','初次上限裁剪'),bi('Matching trims','共同长度裁剪')],length_rows)+'<p>'+bi('Quantiles are per-question realized retained ratios. Before-clamp ratio measures the selected native draft, prior to the last-resort cap. If many outputs require clamping, interpret the condition as compressor plus budget adapter, not the unmodified native method. Target-band counts do not apply to Full or No context. Timing includes all search trials.','分位数按每题最终实际保留率统计。裁剪前保留率测量选定原生草稿，不含末次裁剪。如果许多样本需要裁剪，应将其解读为“压缩器＋预算适配器”，不能当作未修改的原生方法。原文和空上下文不适用目标区间。耗时包含全部长度试探。')+'</p></section>'
    body += '<section><h2>'+bi('Probability diagnostics','概率诊断')+'</h2>'+table(['Method','Mean p(Yes)','ECE ↓','p = 0.5','p = 0 or 1',bi('Wrong 0/1','0/1 且错'),bi('Wrong ≥95% confidence','≥95% 自信且错'),bi('Distinct p','不同概率数')],diag_rows)
    base_rate = diagnostics[methods[0]]['probability']['base_rate']
    body += '<p>'+bi(f'Observed Yes rate: {base_rate:.1%}.',f'真实 Yes 比例：{base_rate:.1%}。')+'</p>'
    body += '<p>'+bi('ECE uses ten fixed equal-width bins: Σ(nᵦ/n)|mean(p)ᵦ − mean(y)ᵦ|. Empty bins do not contribute. Reliability diagrams plot reported p(Yes) against the observed Yes fraction; points are descriptive and noisy in small bins. No probabilities are recalibrated or changed. Perfect calibration is not a prerequisite for the InfoGap formulation used here.',
                        'ECE 使用十个固定等宽箱：Σ(nᵦ/n)|mean(p)ᵦ−mean(y)ᵦ|，空箱不贡献。可靠性图对比报告的 p(Yes) 与实际 Yes 比例；小箱波动大，仅作描述。没有重新校准或改动概率；这里使用的 InfoGap 公式不要求预测器完美校准。')+'</p><div class="grid">'+''.join('<div class="card">'+reliability_plot(diagnostics[m]['probability'],NAMES[m])+'</div>' for m in methods)+'</div>'
    body += '<details><summary>'+bi('Log-loss sensitivity to endpoint clipping','Log loss 对端点截断的敏感性')+'</summary>'+table(['Method','ε = 1e−12','ε = 1e−6','ε = 1e−3'],[[NAMES[m],*[f'{v:.4f}' for v in diagnostics[m]['probability']['log_loss_sensitivity'].values()]] for m in methods])+'<p>'+bi('Only logarithms use clipping. InfoGap, Brier, ECE and all saved probabilities use the original values.','只有对数计算使用截断；InfoGap、Brier、ECE 和所有保存的概率均使用原值。')+'</p></details></section>'
    if rows is not None:
        body += '<section><details><summary>'+bi('Probability changes versus Full','展开：相对原文的概率变化')+'</summary><p>'+bi('Each point is one question. The diagonal means unchanged p(Yes). Teal indicates gold Yes, orange gold No; hover for the sample ID.','每个点是一道题；对角线表示 p(Yes) 不变。青色为真实 Yes，橙色为真实 No；悬停查看样本编号。')+'</p><div class="grid">'+''.join('<div class="card"><h3>'+NAMES[m]+'</h3>'+scatter(rows,m)+'</div>' for m in methods if m != 'full')+'</div></details></section>'
    body += '<section><h2>'+bi('Descriptive uncertainty','描述性波动')+'</h2>'+table(['Method','G(Full, method): 95%','G(method, Full): 95%'],[[NAMES[m],f"{b['a_over_b'][0]:.4f}–{b['a_over_b'][1]:.4f}",f"{b['b_over_a'][0]:.4f}–{b['b_over_a'][1]:.4f}"] for m,b in metrics['bootstrap_vs_full'].items()])+'<p>'+bi('Paired context-cluster bootstrap; percentile ranges are descriptive, not a validated dominance test.','按上下文成组的配对 bootstrap；百分位区间描述波动，不是经过验证的支配检验。')+'</p></section>'
    if metrics['source_tasks']:
        task_rows = []
        for task, values in metrics['source_tasks'].items():
            for method, gaps in values['infogap_vs_full'].items():
                task_rows.append([task, values['n'], NAMES[method], f"{values['scores'][method]['brier']:.4f}", f"{gaps['full_over_method']['value']:.4f}", f"{gaps['method_over_full']['value']:.4f}"])
        body += '<section><details><summary>'+bi('LongBench source-task breakdown','展开：LongBench 来源任务分解')+'</summary>'+table(['Task','n','Method','Brier','G(Full, method)','G(method, Full)'],task_rows)+'<p>'+bi('These small subsets are descriptive; they are not independent benchmark replications.','这些小子集仅作描述，并非独立的 benchmark 重复实验。')+'</p></details></section>'
    body += '<section><h2>'+bi('API usage','API 用量')+'</h2>'+table(['Stage','Attempts','Input tokens','Output tokens'],[[key,v['attempts'],f"{v['prompt_tokens']:,}",f"{v['completion_tokens']:,}"] for key,v in usage.items()])+'<p>'+bi('Compression and reader attempts have separate hard caps and ledgers. Usage excludes prior experiments and smoke runs. Time/cost of search is part of this adapted protocol, not an intrinsic speed claim for the original methods.','压缩和 reader 各有独立硬性尝试上限与账本。用量不含旧实验和小样本；长度搜索的时间及费用属于此适配协议，不能据此宣称原方法本身的速度。')+'</p></section>'
    return body


def sample_section(rows, methods):
    samples = []
    for row in rows:
        body = '<p>Gold: '+str(row['label'])+'</p>'+table(['Method','p(Yes)','Qwen tokens'],[[NAMES[m],row['conditions'][m]['p_yes'],row['conditions'][m]['tokens']] for m in methods])
        for m in methods:
            if m != 'no_context':
                body += '<details><summary>'+NAMES[m]+'</summary><pre>'+html.escape(row['conditions'][m]['text'])+'</pre></details>'
        samples.append('<details><summary>'+html.escape(row['id']+' · '+row['question'])+'</summary>'+body+'</details>')
    return '<section><h2>'+bi('Inspect every example','逐题查看原文与压缩结果')+'</h2>'+''.join(samples)+'</section>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['hotpotqa','qasper','longbench'],default=['hotpotqa','qasper','longbench'])
    parser.add_argument('--longbench-config', default='configs/aligned_longbench.json',
                        help='Explicit config for an audited LongBench recovery run; original config remains unchanged.')
    args = parser.parse_args()
    datasets = args.datasets
    DESTINATION.mkdir(exist_ok=True,parents=True)
    records = []
    for name in datasets:
        config_path = args.longbench_config if name == 'longbench' else f'configs/aligned_{name}.json'
        config = read_json(ROOT / config_path)
        run = ROOT / 'runs' / config['run_name']
        require_audit = read_json(run/'audit.json')
        if require_audit['status'] != 'passed' or read_json(run/'manifest.json')['status'] != 'complete':
            raise ValueError('Only audited complete runs may be published.')
        usage = {}
        for stage, folder in [('Reader',run),('Compression',run/'compression_api')]:
            ledger_paths = [folder/'api_attempts.json'] if stage == 'Reader' else sorted(folder.rglob('api_attempts.json'))
            ledger = [entry for path in ledger_paths for entry in read_json(path)]
            usage[stage] = {'attempts':len(ledger), **{k:sum(e.get('usage',{}).get(k,0) for e in ledger) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
        records.append({'name':name,'config':config,'run':run,'metrics':read_json(run/'metrics.json'),
                        'diagnostics':read_json(run/'diagnostics.json'),'rows':read_json(run/'predictions.json'),'usage':usage,'audit':require_audit})
        record = records[-1]
        for method in config['methods']:
            native_tokens = 0
            before_match_tokens = 0
            before_match_underfilled = 0
            for row in record['rows']:
                condition = row['conditions'][method]
                before_match_tokens += condition.get('before_match_tokens',condition['tokens'])
                if condition.get('budget_tokens') is not None:
                    before_match_underfilled += condition.get('before_match_tokens',condition['tokens']) < .95*condition['budget_tokens']
                if 'local_compression_hash' in condition:
                    entry = read_json(run/'aligned_compression_cache'/f"{condition['local_compression_hash']}.json")['result']
                    native_tokens += entry['candidates'][entry['selected_candidate']]['tokens']
                else:
                    native_tokens += condition.get('before_match_tokens',condition['tokens'])
            record['diagnostics'][method]['retention']['before_match_underfilled'] = before_match_underfilled
            record['diagnostics'][method]['retention']['before_match_pooled_ratio'] = before_match_tokens / sum(row['original_tokens'] for row in record['rows'])
            record['diagnostics'][method]['retention']['selected_before_clamp_pooled_ratio'] = native_tokens / sum(row['original_tokens'] for row in record['rows'])
    scale = max(c['value'] for r in records for row in r['metrics']['infogap'].values() for c in row.values())
    overview = '<section><p>'+bi(f'Completed and audited datasets shown: {len(records)} / 3. The page is updated as each full dataset finishes.',f'当前展示已完成且通过审计的数据集：{len(records)} / 3；每个完整数据集完成后更新。')+'</p></section>'+protocol_section()
    overview += '<section><p>'+bi('Only complete API rewrites are accepted. An output-limit response triggers at most one identical-prompt retry with a larger output ceiling (maximum 32768); truncated text is excluded. Raw responses and attempts remain recorded. Reader forecasts are strictly parsed without this retry. Validated cache reuse and original compute times are recorded in each v2 run’s cache_reuse_provenance.json; previous runs are preserved.', '仅接受完整 API 改写。输出上限截断最多触发一次相同提示词、提高输出上限的重试（最多 32768），截断文本不参与实验。原响应及尝试次数均留存。reader 概率严格解析且不使用此重试。v2 的 cache_reuse_provenance.json 记录经核验的缓存复用和原计算时间；旧运行全部保留。')+'</p></section>'
    overview += '<section><h2>'+bi('Choose a dataset','选择数据集')+'</h2><p>'+ ' · '.join(f'<a href="#{r["name"]}">{r["name"]} (n={r["metrics"]["n"]})</a>' for r in records)+'</p><p><a href="../Code_Walkthrough_20261024_Aligned.html">'+bi('Read the new pipeline and its code','查看新流程与逐块代码导读')+'</a></p></section>'
    for r in records:
        name, config, run = r['name'],r['config'],r['run']
        title = f"{name} · n={r['metrics']['n']} · {config['dataset_split']}"
        panels = method_panels(config,r['metrics'],r['diagnostics'],scale,r['usage'],r['rows'])
        if (run/'timeout_recovery_provenance.json').exists():
            panels += '<section><p>'+bi('Recovery provenance: this dataset uses a separately preserved run with a 30-minute network timeout instead of 10 minutes. Model requests, prompts, output ceilings, samples and cumulative API attempt limits are unchanged. Verified cached results and original attempt ledgers are reused once; failed runs remain preserved.', '恢复记录：该数据集使用独立保存的运行，将网络等待从 10 分钟延长到 30 分钟。模型请求、提示词、输出上限、样本及累计 API 尝试上限不变；核验后的缓存与原账本只统计一次，失败运行完整保留。')+'</p></section>'
        overview += f'<section id="{name}"><h2>{title}</h2><a href="{name}.html">'+bi('Every example and compressed text','逐题预测与压缩文本')+f'</a> · <a href="{name}_predictions.csv">CSV</a> · <a href="{name}_metrics.json">InfoGap JSON</a></section>'+panels
        full = page(title,protocol_section()+panels+sample_section(r['rows'],config['methods']))
        (DESTINATION/f'{name}.html').write_text(full,encoding='utf-8')
        (run/'report.html').write_text(full,encoding='utf-8')
        for file in ('metrics.json','predictions.csv','diagnostics.json','audit.json'):
            shutil.copyfile(run/file,DESTINATION/f'{name}_{file}')
        write_json(DESTINATION/f'{name}_diagnostics.json',r['diagnostics'])
    overview += '<section><h2>'+bi('Interpretation and sources','解释与来源')+'</h2><pre>G(A,B) = 2 sup_t mean[(t−y)(1[p_A≤t]−1[p_B≤t])]</pre><p>'+bi('All breakpoints and left limits are evaluated. Both scaled and unscaled values are saved. These are verbalized forecasts, not verified Bayesian posteriors. The experiment evaluates usefulness for these question distributions; it does not establish query-distribution-independent Blackwell dominance.','计算所有概率断点及左侧极限，同时保存乘 2 与未乘 2 的结果。这些是模型口头报告的概率，并非核实过的贝叶斯后验。实验评价当前问题分布下的预测价值，不证明与问题分布无关的 Blackwell 支配。')+'</p><p><a href="https://hotpotqa.github.io/">HotpotQA</a> · <a href="https://aclanthology.org/2021.naacl-main.365/">QASPER</a> · <a href="https://github.com/THUDM/LongBench">LongBench</a> · <a href="https://arxiv.org/html/2507.12094v2">InfoGap</a> · <a href="https://github.com/sibmike/telegrapher/tree/7aadc093eeebe63953722b79df1a57c57f569288">Telegrapher pinned source</a> · <a href="https://github.com/oteomamo/SALT">SALT</a></p><p><a href="../Experiments_20261023_SALT/Experiment_Overview.html">'+bi('Previous five-condition results (unchanged)','旧五组结果（保留不变）')+'</a></p></section>'
    (DESTINATION/'Experiment_Overview.html').write_text(page('Compaction · matched context lengths & probability diagnostics',overview),encoding='utf-8')
    write_json(DESTINATION/'summary.json',[{k:v for k,v in r.items() if k not in ('rows','run')} for r in records])
    print(DESTINATION/'Experiment_Overview.html')


if __name__ == '__main__':
    main()
