"""Build shareable bilingual HTML, a detailed offline audit viewer, and an English PDF."""
import json, html, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
SUITE=ROOT/'runs/mini100_cap50_20261025_v4'
OUT=ROOT/'docs/Mini100_Cap50_20261025'
METHODS=['full','no_context','selective_context','llmlingua2','llm_summary','dac']
LABEL={'full':'Full','no_context':'No context','selective_context':'Selective Context','llmlingua2':'LLMLingua-2','llm_summary':'LLM summary','dac':'DAC'}
SHORT=['Full','No ctx','SC','LLMLingua-2','Summary','DAC']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def esc(s): return html.escape(str(s))
def bi(en,zh): return f'<span class="en">{en}</span><span class="zh">{zh}</span>'
def cell_color(v,maximum):
    t=v/maximum if maximum else 0
    return '#'+''.join(f'{round(a+(b-a)*t):02x}' for a,b in zip((242,247,250),(35,89,130)))
CSS='''*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#172b3b;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:36px auto;padding:0 28px 50px}h1{font-size:36px;line-height:1.2;letter-spacing:-.5px}h2{font-size:24px;margin-top:36px}h3{font-size:18px}p{max-width:1000px}.eyebrow{color:#526d7d;letter-spacing:2px;font-size:12px}.card{background:white;border:1px solid #dce5eb;border-radius:14px;padding:24px;margin:20px 0}.lead{font-size:19px;color:#365468}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.stat{font-size:28px;font-weight:650}small,.muted{color:#526574;font-size:13px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 9px;text-align:right;border-bottom:1px solid #dfe7eb}th:first-child,td:first-child{text-align:left}thead{background:#edf3f6}a{color:#176190}button,select,input{font:inherit;padding:8px 12px;border:1px solid #b9cbd7;border-radius:8px;background:white;color:#24445d}button{cursor:pointer}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.zh{display:none}html[lang=zh-CN] .zh{display:inline}html[lang=zh-CN] .en{display:none}.warn{border-left:4px solid #b77318;padding:12px 18px;background:#fff7e7}.matrix td{text-align:center;border:3px solid white;font-variant-numeric:tabular-nums}.matrix th{font-size:12px;text-align:center}.matrix th:first-child{text-align:left}.matrix-wrap{overflow-x:auto}.text{white-space:pre-wrap;overflow-wrap:anywhere;max-height:650px;overflow:auto;font:14px/1.6 ui-monospace,Consolas,monospace;background:#f8fafb;border:1px solid #dae4eb;padding:16px}.chip{display:inline-block;font-size:12px;background:#e5eef4;padding:3px 9px;border-radius:20px;margin-right:6px}details{margin:12px 0}summary{cursor:pointer;color:#176190}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.foot{margin-top:32px;border-top:1px solid #dce5eb;padding-top:16px}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:0 15px}h1{font-size:28px}}@media print{button,.toolbar{display:none}main{max-width:none;margin:0}.card{break-inside:avoid}}'''
LANG="function setLang(l){document.documentElement.lang=l}document.querySelectorAll('[data-lang]').forEach(b=>b.onclick=()=>setLang(b.dataset.lang));"
def shell(title,body,script=''):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+CSS+'</style><main><div class="toolbar"><button data-lang="en">English</button><button data-lang="zh-CN">中文</button></div>'+body+'</main><script>'+LANG+script+'</script></html>'

def matrix_html(metrics,maximum):
    head='<tr><th>'+bi('Row → column','行 → 列')+'</th>'+''.join('<th>'+x+'</th>' for x in SHORT)+'</tr>'
    rows=[]
    for a,label in zip(METHODS,SHORT):
        cells=[]
        for b in METHODS:
            g=metrics['infogap'][a][b]; v=g['value']; fg='white' if maximum and v/maximum>.57 else '#172b3b'
            cells.append(f'<td style="background:{cell_color(v,maximum)};color:{fg}" title="t={g["threshold"]}; {g["side"]}">{v:.3f}</td>')
        rows.append('<tr><th>'+label+'</th>'+''.join(cells)+'</tr>')
    return '<div class="matrix-wrap"><table class="matrix">'+head+''.join(rows)+'</table></div>'

def retention_range(rows, method):
    values = [r['conditions'][method]['retained_ratio_actual'] for r in rows]
    return f'{min(values):.0%}-{max(values):.0%}'

def score_table(data):
    metrics, rows = data['metrics'], data['rows']
    header = ['Method', 'Accuracy ↑', 'Brier ↓', 'Log loss ↓', 'Retained', 'Min–max']
    values = []
    for method in METHODS:
        score = metrics['scores'][method]
        values.append([LABEL[method], f'{score["accuracy"]:.1%}', f'{score["brier"]:.3f}',
                       f'{score["log_loss"]:.3f}', f'{score["pooled_retained_ratio"]:.1%}', retention_range(rows, method)])
    return '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in header)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in values)+'</tbody></table>'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(SUITE/'manifest.json')
    if manifest['status']!='complete': raise ValueError('Both audits must finish before publication')
    data={name:{'rows':read(SUITE/name/'predictions.json'),'metrics':read(SUITE/name/'metrics.json'),
               'audit':read(SUITE/name/'audit.json'),'diagnostics':read(SUITE/name/'diagnostics.json')} for name in ['hotpotqa','qasper']}
    if any(d['audit']['status']!='passed' for d in data.values()): raise ValueError('Audit failure')
    maximum=max(d['metrics']['infogap'][a][b]['value'] for d in data.values() for a in METHODS for b in METHODS)
    spending=read(SUITE/'spending_reservations.json')
    accounted=sum(x['accounted_cny'] for x in spending)
    unknown=sum(x.get('unknown_usage_attempts',0) + int(x['status']=='interrupted_usage_unknown') for x in spending)
    total_band=sum(sum(v for v in d['audit']['below_47_5_percent_reference_counts'].values()) for d in data.values())
    findings=[]
    for name,d in data.items():
        winner=min(METHODS[2:],key=lambda m:d['metrics']['scores'][m]['brier'])
        full=d['metrics']['scores']['full']['brier']; value=d['metrics']['scores'][winner]['brier']
        findings.append((f'{name.title()}: {LABEL[winner]} has the lowest compressed-condition Brier score ({value:.3f}; Full {full:.3f}).',
                         f'{name.title()}：压缩条件中 {LABEL[winner]} 的 Brier 最低（{value:.3f}；Full 为 {full:.3f}）。'))
    improving=[f'{n}: {LABEL[m]}' for n,d in data.items() for m in METHODS[2:] if d['metrics']['scores'][m]['brier']<d['metrics']['scores']['full']['brier']]
    wins=[f'{n}: {LABEL[m]}' for n,d in data.items() for m in METHODS[2:] if d['metrics']['infogap'][m]['full']['value']>1e-12]
    dom=[f'{n}: {LABEL[m]}' for n,d in data.items() for m in METHODS[2:] if d['metrics']['infogap']['full'][m]['value']<1e-12 and d['metrics']['infogap'][m]['full']['value']>1e-12]
    body='<p class="eyebrow">BLACKWELL COMPACTION · EXPLORATORY STUDY · 25 OCT 2026</p><h1>'+bi('What survives a fixed compression budget?','固定压缩预算下，哪些信息被保留？')+'</h1><p class="lead">'+bi('200 binary questions · two datasets · six conditions · query-blind compression','200 道二元问题 · 两个数据集 · 六个条件 · 压缩不看 query')+'</p>'
    body+='<div class="card"><h2 style="margin-top:0">'+bi('Findings at a glance','主要观察')+'</h2>'+''.join('<p>'+bi(en,zh)+'</p>' for en,zh in findings)+'<p>'+bi('Positive reverse InfoGap identifies decision-specific gains, not universal information improvement. '+('No compressed condition empirically dominates Full across the tested threshold family.' if not dom else 'Empirical zero-loss directions require held-out validation; they do not establish population dominance.'),
        '反向 InfoGap 为正意味着某些决策下有收益，并非整体信息量提升。'+('没有压缩条件在本次阈值族比较中单向支配 Full。' if not dom else '观察到的零损失方向仍需独立数据验证，不能据此证明总体支配。'))+'</p></div>'
    body+='<div class="card">'+bi('All compressed inputs satisfy the same preassigned 50% cap. No minimum length is required. Actual lengths differ, so performance differences combine method and length effects.', '所有压缩输入都满足预先固定的 50% 上限，不要求最低长度。实际长度不同，因此性能差异同时包含方法与长度的影响。')+'</div>'
    body+='<h2>'+bi('Protocol','实验设置')+'</h2><div class="card"><p>'+bi('Reader: Qwen3-30B-A3B-Instruct-2507. Summary writer: DeepSeek-V3.2. Temperature 0; thinking disabled; one verbalized p(Yes) per question and condition. The API returns a probability JSON, not token-logprob-derived or calibrated posteriors.',
        'Reader：Qwen3-30B-A3B-Instruct-2507；摘要模型：DeepSeek-V3.2。temperature=0，关闭 thinking；每题每条件一次口头概率 p(Yes)。这些数值并非由 token logprob 推导或已经校准的后验。')+'</p><p>'+bi('Each source receives B=floor(0.5 × original Qwen tokens) before compression. There is no minimum fill requirement. Summary selection takes the first verified complete candidate within B in saved generation order, without using questions, labels or prediction scores. No shortest-output matching, post-truncation, label feedback or query input is allowed.',
        '压缩前按原文确定 B=floor(0.5 × 原文 Qwen tokens)，不要求用满预算。摘要按保存的生成顺序，选择第一份完整且不超过 B 的候选，不参考问题、标签或预测效果；不向最短输出对齐，不事后截断，不使用标签反馈，也不向压缩器传 query。')+'</p><p>'+bi('HotpotQA: 100 native yes/no distractor-dev questions (45 Yes / 55 No). QASPER: 100 consistent, text-evidence yes/no train questions from 97 papers (71 Yes / 29 No). Samples are uniformly drawn from the previously defined pools with seed 20261025, without outcome filtering. This is an exploratory subset; QASPER train is not a held-out test.',
        'HotpotQA：distractor dev 原生 yes/no 随机 100 题，45 Yes／55 No。QASPER：train 中标签一致、证据可在正文找到的 yes/no 随机 100 题，来自 97 篇论文，71 Yes／29 No。从先前定义的样本池以种子 20261025 均匀抽样，不按模型结果筛选。这是探索性子集，QASPER train 不是独立测试集。')+'</p></div>'
    body+='<div class="card"><h3>'+bi('Prompt controls and summary provenance','Prompt 控制与摘要来源')+'</h3><p>'+bi('The reader system prompt, question wrapper, model and generation parameters are unchanged across conditions; only context text changes. Summary drafts are heterogeneous historical outputs: HotpotQA uses 49 original-prompt drafts and 51 length-feedback drafts (49 proportional, 2 earlier feedback); QASPER uses 97 original-prompt drafts covering 100 questions. All compressor requests are query-blind. This summary condition is not a uniform one-shot prompt baseline.', '各条件共享未改动的 reader system prompt、问题包装、模型与生成参数，仅改变上下文文本。摘要来自不同历史阶段：HotpotQA 49 份为原始提示词稿，51 份为长度反馈稿（49 份比例反馈、2 份早期反馈）；QASPER 97 份原始提示词稿覆盖 100 题。所有压缩请求均不含 query。摘要条件不是统一提示词的一次调用 baseline。')+'</p></div>'
    for name,d in data.items():
        body+='<h2>'+name.title()+'</h2><div class="card">'+score_table(d)+'<small>'+bi('Retained = pooled compressed/source tokens. Log loss clips probabilities only for its logarithm (ε=10⁻¹²). InfoGap and Brier use the unchanged saved probabilities.',
            'Retained 为所有题目的压缩／原文 token 总量之比。Log loss 仅在取对数时用 ε=10⁻¹² 截断；InfoGap、Brier 使用原始保存概率。')+'</small></div>'
    body+='<h2>'+bi('Bidirectional InfoGap','双向 InfoGap')+'</h2><p>'+bi('Read a cell as G(row, column): the row predictor’s largest normalized decision-payoff advantage over the column predictor. Darker means a larger gap. Both datasets share the same color scale. Two positive directions mean different decisions favor different predictors; the two cells are not a net gain to subtract.',
        '每格为 G(行,列)：行预测器相对列预测器的最大归一化决策收益优势。越深表示 gap 越大，两组共用色阶。双向均为正表示不同决策各有优劣，不能相减解释为净信息增益。')+'</p>'
    for name,d in data.items(): body+='<div class="card"><h3>'+name.title()+'</h3>'+matrix_html(d['metrics'],maximum)+'</div>'
    body+='<div class="card"><h3>'+bi('Against Full: both directions and uncertainty','相对 Full 的双向值与不确定性')+'</h3><table><tr><th>Dataset / method</th><th>G(Full, method)</th><th>G(method, Full)</th></tr>'
    for n,d in data.items():
        for m in METHODS[2:]:
            ci=d['metrics']['bootstrap_vs_full'][m]
            def gap_cell(a,b,interval): return f'{d["metrics"]["infogap"][a][b]["value"]:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]'
            body+=f'<tr><td>{n.title()} / {LABEL[m]}</td><td>{gap_cell("full",m,ci["a_over_b"])}</td><td>{gap_cell(m,"full",ci["b_over_a"])}</td></tr>'
    body+='</table><small>'+bi('500 paired cluster-bootstrap resamples by context; descriptive 95% percentile intervals. They are not dominance tests. The supremum is selected on these same 100 examples, so held-out validation remains necessary.',
        '按共享上下文成组配对 bootstrap 500 次，展示描述性 95% 百分位区间，并非支配检验。最大收益阈值仍在同一批 100 题上选择，因此需要后续独立验证。')+'</small></div>'
    body+='<h2>'+bi('Interpretation and limits','解释与边界')+'</h2><div class="card"><p>'+bi('The tested object is a compressor-plus-fixed-reader pipeline on a particular question distribution. Raw p(Yes) may be miscalibrated; confidence shifts can change InfoGap even without better factual retention. These results do not establish a query-agnostic Blackwell ordering of compressed text.',
        '评测对象是特定问题分布下的“压缩器＋固定 reader”。原始 p(Yes) 可能失准；置信度变化也会改变 InfoGap，并不必然代表事实保留更好。本实验不能建立压缩文本本身 query-agnostic 的 Blackwell 排序。')+'</p><p>'+bi('DAC uses the official dynamic attention + information-score selection (10 iterations, α=0.8), with a full-context blocked attention reduction and blocked vocabulary logits for 8GB GPU memory. Two short examples matched the official compressed text exactly; the complete longest selected source (16,755 tokens) passed at a 3.25 GiB peak. No source tokens were truncated. Floating-point reductions can differ slightly from the reference implementation.',
        'DAC 使用官方动态注意力＋信息量筛选（10 次迭代，α=0.8）；为适配 8GB 显存，完整上下文注意力与词表 logits 分块计算。两份短样例与官方压缩文本完全一致；最长 16,755 tokens 完整输入通过，峰值约 3.25 GiB。没有截掉原文；浮点累加顺序仍可能产生微小数值差异。')+'</p><p>'+bi('Historical outputs are reused only for identical complete reader requests. Summary drafts are reused from historical source-only requests; no new summary generation is needed under the cap-only protocol. Local compressor outputs are reused unchanged. This is not a fresh one-call generation benchmark, and prior length-adaptation effort differs across methods. All 1,200 probabilities, compression lengths and 72 InfoGap cells passed the audit; old result files were hash-checked unchanged.',
        '历史预测仅在完整 reader 请求一致时复用。摘要复用历史上仅含原文和旧草稿的请求，本次上限方案不再生成摘要；本地压缩输出原样复用。这不是从零开始的单次调用比较，各方法此前经历的长度适配计算量也不同。1,200 条概率、压缩长度和 72 个 InfoGap 单元格通过审计，旧结果文件哈希保持不变。')+'</p></div>'
    body+=f'<p class="muted">'+bi(f'Conservative accounted new API spend: ¥{accounted:.2f}; unknown-usage records: {unknown}. This is not the provider invoice. Local computation and cached responses incur no new model API charge.',f'新增 API 保守记账约 ¥{accounted:.2f}；用量未知的记录 {unknown} 次。此数并非平台账单。本地计算与直接复用缓存不新增模型 API 费用。')+'</p>'
    body+='<div class="foot"><a href="https://arxiv.org/abs/2507.12094">InfoGap theory</a> · <a href="https://aclanthology.org/2025.acl-long.952/">DAC</a> · <a href="https://aclanthology.org/2024.findings-acl.57/">LLMLingua-2</a> · <a href="https://aclanthology.org/2023.emnlp-main.391/">Selective Context</a><p>'+bi('Companion files:','配套文件：')+' <a href="Mini100_Audit.html">'+bi('Per-example audit','逐题审计')+'</a> · <a href="Mini100_Report.pdf">PDF</a></p></div>'
    (OUT/'Mini100_Report.html').write_text(shell('Fixed-budget compaction: mini study',body),encoding='utf-8')
    build_audit(data,manifest)
    build_pdf(data,manifest,findings,maximum,total_band,accounted)
    write_json_simple(OUT/'report_data.json',{'manifest':manifest,'metrics':{n:d['metrics'] for n,d in data.items()},'spending_accounted_cny':accounted})
    print('Report HTML, audit HTML and PDF created in',OUT)

def write_json_simple(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def build_audit(data,manifest):
    documents={}; records=[]
    for dataset,d in data.items():
        folder=SUITE/dataset
        compression_index={}
        for source_file in (folder/'compressions').glob('*.json'):
            if not source_file.name.endswith('.drafts.json'):
                entry=read(source_file); compression_index[(entry['request']['context'],entry['request']['method'])]=entry
        for row in d['rows']:
            key=row['context_group']
            if key not in documents:
                documents[key]={'source':row['context'],'original_tokens':row['original_tokens'],'compressions':{}}
                for method in METHODS[2:]:
                    # Resolve by exact source/method request, independent of question.
                    documents[key]['compressions'][method]=compression_index[(row['context'],method)]
                    if method == 'llm_summary':
                        entry = documents[key]['compressions'][method]
                        chosen = entry['candidates'][entry['result']['selected_candidate']]
                        request_file = ROOT/chosen['api_folder']/'api_cache'/f"{chosen['request_hash']}.json"
                        entry['selected_request_for_audit'] = read(request_file)['request']

            conditions={}
            for method,c in row['conditions'].items():
                response=read(folder/'api_cache'/f'{c["prediction_request_hash"]}.json')
                conditions[method]={**{k:v for k,v in c.items() if k!='text'},
                    'raw_output':response['response']['choices'][0]['message']['content'],
                    'usage':response['response'].get('usage'),
                    'received_utc':response.get('received_utc'),
                    'request_settings':{k:v for k,v in response['request']['payload'].items() if k!='messages'}}
            records.append({'dataset':dataset,'id':row['id'],'context_group':key,'question':row['question'],
                            'label':row['label'],'conditions':conditions})
    data_json=json.dumps({'records':records,'documents':documents,'methods':METHODS,'labels':LABEL},ensure_ascii=False).replace('<','\\u003c')
    body='<p class="eyebrow">BLACKWELL COMPACTION · OFFLINE AUDIT</p><h1>'+bi('Inspect every experimental output','逐题审计实验输出')+'</h1><p>'+bi('Filter examples, compare exact texts, inspect raw probability responses and trace the selected compression candidate. The summary uses the first complete draft within its fixed cap; later drafts are retained only as history. Shorter outputs are valid. Nothing on this page calls an API.',
        '筛选题目，对比原始文本与压缩文本，查看原始概率响应和压缩候选来源。摘要取第一份完整合规稿，后续草稿仅作历史记录；偏短也是合规结果。此页面不会调用任何 API。')+'</p><div class="toolbar"><select id="dataset"><option value="all">Both datasets / 全部</option><option>hotpotqa</option><option>qasper</option></select><input id="search" placeholder="Question or ID / 问题或编号"><select id="method"></select><select id="filter"><option value="all">All / 全部</option><option value="wrong">Wrong at 0.5 / 错误预测</option><option value="changed">Label differs from Full / 与 Full 判断不同</option><option value="under">Retention &lt;47.5% / 保留比例低于47.5%</option></select></div><div class="card"><select id="example" style="width:100%"></select><div id="meta"></div></div><div class="grid"><div class="card"><h3>'+bi('Original source','原始上下文')+'</h3><div id="original" class="text"></div></div><div class="card"><h3>'+bi('Selected method input','所选方法输入')+'</h3><div id="compressed" class="text"></div></div></div><div class="card"><h2>'+bi('Probability and provenance','概率与来源')+'</h2><div id="probability"></div><details><summary>'+bi('Raw probability response / usage / request settings','原始概率响应／用量／请求参数')+'</summary><pre id="raw"></pre></details><details><summary>'+bi('Saved candidates and selection (first complete feasible summary)','历史候选和选择（摘要取第一份完整合规稿）')+'</summary><pre id="candidate"></pre></details></div><p class="foot"><a href="Mini100_Report.html">'+bi('Back to research report','返回研究报告')+'</a></p>'
    script='const DATA='+data_json+''';
const $=s=>document.querySelector(s);let visible=[];
function escapeText(s){const e=document.createElement('span');e.textContent=s;return e.innerHTML}
$('#method').innerHTML=DATA.methods.map(m=>`<option value="${m}">${DATA.labels[m]}</option>`).join('');$('#method').value='llm_summary';
function filterRows(){let m=$('#method').value,ds=$('#dataset').value,q=$('#search').value.toLowerCase(),f=$('#filter').value;
visible=DATA.records.filter(r=>{let c=r.conditions[m];return (ds==='all'||r.dataset===ds)&&(!q||(r.question+' '+r.id).toLowerCase().includes(q))&&(f==='all'||f==='wrong'&&c.predicted_label!==r.label||f==='changed'&&c.predicted_label!==r.conditions.full.predicted_label||f==='under'&&c.budget_tokens&&c.budget_fill<.95)});
$('#example').innerHTML=visible.map((r,i)=>`<option value="${i}">${i+1}. [${r.dataset}] ${escapeText(r.question)}</option>`).join('');render();}
function render(){let r=visible[Number($('#example').value)||0];if(!r){$('#meta').textContent='No matching examples / 没有匹配题目';['original','compressed','probability','raw','candidate'].forEach(id=>$('#'+id).textContent='');return}
let m=$('#method').value,c=r.conditions[m],doc=DATA.documents[r.context_group],entry=doc.compressions[m];
$('#meta').innerHTML=`<h3>${escapeText(r.question)}</h3><p>${escapeText(r.id)}</p><span class="chip">Gold: ${r.label?'Yes':'No'}</span><span class="chip">Original: ${doc.original_tokens} tokens</span><span class="chip">Selected: ${c.tokens} tokens</span><span class="chip">Budget: ${c.budget_tokens??'N/A'}</span><span class="chip">${c.reader_reused?'Reused exact request / 复用':'New request / 新调用'}</span>`;
$('#original').textContent=doc.source;$('#compressed').textContent=m==='full'?doc.source:m==='no_context'?'(No context / 无上下文)':entry.result.text;
$('#probability').innerHTML=`<p><b>p(Yes) = ${c.p_yes}</b> · Full p(Yes) = ${r.conditions.full.p_yes} · Brier contribution = ${(c.p_yes-r.label)**2}</p><p>Retention: ${(100*c.retained_ratio_actual).toFixed(2)}% · Within 50% cap: ${c.budget_met??'N/A'}</p><small>Request hash: ${c.prediction_request_hash}</small>`;
$('#raw').textContent=JSON.stringify({raw_output:c.raw_output,usage:c.usage,received_utc:c.received_utc,request_settings:c.request_settings,question:r.question},null,2);
$('#candidate').textContent=entry?JSON.stringify(entry,null,2):'Control condition / 对照条件';}
['dataset','method','filter'].forEach(id=>$('#'+id).onchange=filterRows);$('#search').oninput=filterRows;$('#example').onchange=render;filterRows();'''
    (OUT/'Mini100_Audit.html').write_text(shell('Compaction: per-example audit',body,script),encoding='utf-8')

def build_pdf(data,manifest,findings,maximum,underfill,accounted):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='BodyMini',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='NoteMini',fontName='Helvetica',fontSize=8,leading=11,textColor=colors.HexColor('#526574'),spaceAfter=7))
    story=[]
    def p(text,style='BodyMini'): return Paragraph(text,styles[style])
    def table(rows,widths=None,small=False):
        t=Table([[p(str(v),'NoteMini' if small else 'BodyMini') for v in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d7e2e9'))]))
        return t
    story += [p('BLACKWELL COMPACTION | 25 OCTOBER 2026','NoteMini'),p('Fixed-budget context compaction','Title'),p('An exploratory comparison on 100 HotpotQA and 100 QASPER binary questions','Heading2'),Spacer(1,10)]
    for en,zh in findings: story.append(p(en))
    story.append(p('Reader: Qwen3-30B-A3B-Instruct-2507. Summary writer: DeepSeek-V3.2. Temperature 0; thinking off. Probabilities are verbalized p(Yes), not calibrated or token-logprob-derived posteriors.'))
    story.append(p('Before compression, each source receives a budget of half its Qwen token count. There is no minimum retention requirement. No query, label or prediction is passed to a compressor; no cross-method minimum matching or post-truncation is used.'))
    for name,d in data.items():
        story.append(p(name.title(), 'Heading2'))
        rows=[['Method','Accuracy','Brier','Log loss','Retained','Min-max']]
        for m in METHODS:
            s=d['metrics']['scores'][m]; band=sum(r['conditions'][m].get('within_target_band',False) for r in d['rows'])
            rows.append([LABEL[m],f'{s["accuracy"]:.1%}',f'{s["brier"]:.3f}',f'{s["log_loss"]:.3f}',f'{s["pooled_retained_ratio"]:.1%}',retention_range(d['rows'],m)])
        story.append(table(rows,[135,65,58,65,65,60],True));story.append(Spacer(1,8))
    story.append(p('Accuracy: higher is better. Brier and log loss: lower is better. Retention uses pooled token counts. Log loss clips only for the logarithm at epsilon=1e-12.','NoteMini'))
    story.append(p('All outputs respect the 50% cap. Actual lengths differ; this comparison does not isolate length from method effects. Summary uses the first complete feasible saved draft, not the longest or best-performing draft.','NoteMini'))
    story.append(PageBreak())
    story += [p('Bidirectional InfoGap','Title'),p('Each cell is G(row, column): the largest normalized decision-payoff advantage of the row predictor over the column predictor. Darker cells show larger gaps. Both panels use the same scale.')]
    for name,d in data.items():
        story.append(p(name.title(),'Heading2'))
        rows=[['Row / column',*SHORT]]
        for a,label in zip(METHODS,SHORT): rows.append([label,*[f'{d["metrics"]["infogap"][a][b]["value"]:.3f}' for b in METHODS]])
        t=Table(rows,colWidths=[92]+[59]*6,rowHeights=[31]+[31]*6,hAlign='LEFT')
        commands=[('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),8),('ALIGN',(1,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f5')),('GRID',(1,1),(-1,-1),2,colors.white)]
        for i,a in enumerate(METHODS,1):
            for j,b in enumerate(METHODS,1):
                v=d['metrics']['infogap'][a][b]['value'];commands.append(('BACKGROUND',(j,i),(j,i),colors.HexColor(cell_color(v,maximum))))
                if maximum and v/maximum>.57: commands.append(('TEXTCOLOR',(j,i),(j,i),colors.white))
        t.setStyle(TableStyle(commands));story += [t,Spacer(1,14)]
    story.append(p('G(A,B) = 2 sup_t mean[(t-y){1[p_A &lt;= t] - 1[p_B &lt;= t]}]. All observed probability breakpoints and one-sided limits are scanned. The unscaled value is also saved.','NoteMini'))
    story.append(p('Two positive directions identify decision tradeoffs, not a net information gain. A positive G(compressed, Full) alone does not establish Blackwell dominance.'))
    story.append(PageBreak())
    story += [p('Uncertainty, validity and reproducibility','Title'),p('Uncertainty relative to Full','Heading2')]
    rows=[['Dataset / method','G(Full, method), 95% interval','G(method, Full), 95% interval']]
    for name,d in data.items():
        for m in METHODS[2:]:
            ci=d['metrics']['bootstrap_vs_full'][m]
            def f(a,b,interval):return f'{d["metrics"]["infogap"][a][b]["value"]:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]'
            rows.append([name.title()+' / '+LABEL[m],f('full',m,ci['a_over_b']),f(m,'full',ci['b_over_a'])])
    story.append(table(rows,[174,137,137],True))
    story.append(p('Intervals: 500 paired cluster-bootstrap draws by source context; descriptive percentile intervals, not dominance tests. Thresholds are selected on the same evaluation examples.','NoteMini'))
    for title,text in [
        ('Sampling','Seed 20261025; uniform draws from previously defined pools, without model-outcome selection. HotpotQA: 45 Yes / 55 No, distractor dev. QASPER: 71 Yes / 29 No, 97 papers, consistent binary annotations with text-verifiable evidence, train split. Evidence filters define the subset but do not select compressor input. This is not official full-benchmark scoring.'),
        ('Prompt controls','Reader prompt and generation settings are unchanged across conditions. Summary provenance: HotpotQA uses 49 original-prompt and 51 length-feedback drafts; QASPER uses 97 original-prompt drafts for 100 questions. All compression is query-blind; this is not a uniform one-shot summary baseline.'),
        ('DAC implementation','Pinned official code; Qwen2-0.5B-Instruct, float32, dynamic_attn_ppl, additive fusion, alpha=0.8, 10 iterations. Query-row blocks retain attention to the entire causal prefix; vocabulary logits are also blocked. Two short examples produced identical reference outputs. The full longest source (16,755 tokens) passed at a 3.25 GiB peak. Floating-point reduction differences remain possible.'),
        ('Audit and interpretation','Summaries reuse the first complete feasible historical draft; no new summary call is made. This is not a fresh one-shot generation benchmark. All 1,200 probabilities replay from exact request caches; all 72 matrix cells were independently checked. Historical results remain unchanged. Cached predictions span earlier collection dates. This evaluates reader-dependent usefulness on a fixed query distribution, not intrinsic query-agnostic informativeness. Calibration and small-sample uncertainty limit dominance claims.'),
        ('Cost',f'Conservative accounted new API cost: CNY {accounted:.2f}, with CNY 20 reserved. This accounting is not the provider invoice; local GPU compute and reused caches create no new model API charge.')]:
        story.append(p(title,'Heading3'));story.append(p(text,'NoteMini'))
    story.append(p('Sources: <link href="https://arxiv.org/abs/2507.12094">InfoGap</link>; <link href="https://aclanthology.org/2025.acl-long.952/">DAC (ACL 2025)</link>; <link href="https://aclanthology.org/2024.findings-acl.57/">LLMLingua-2</link>; <link href="https://aclanthology.org/2023.emnlp-main.391/">Selective Context</link>. The companion HTML includes an offline per-example audit viewer.','NoteMini'))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526574'));canvas.drawString(48,28,'Blackwell Compaction | Exploratory mini study');canvas.drawRightString(A4[0]-48,28,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'Mini100_Report.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=45)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':main()
