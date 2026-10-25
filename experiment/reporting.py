"""从审计通过的保存结果生成独立双语 HTML；没有模型调用。"""
import html
import json
from blackwell.storage import read_json, file_hash
from blackwell.reporting import STYLE, bi, table

LABELS = {'full': 'Full', 'no_context': 'No context', 'selective_context': 'Selective Context',
          'llmlingua2': 'LLMLingua-2', 'llm_summary': 'LLM summary', 'dac': 'DAC'}


def page(title, body, script=''):
    """title：标题；body：已转义 HTML 内容；script：本地交互代码。"""
    return ('<!doctype html><html lang="en" data-language="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+
        '</title><style>'+STYLE+'</style><div class="toolbar"><button onclick="switchLanguage(\'en\')">English</button>'
        '<button onclick="switchLanguage(\'zh\')">中文</button></div>'+body+
        '<script>function switchLanguage(l){document.documentElement.dataset.language=l;document.documentElement.lang=l;}'
        +script+'</script></html>')


def heatmap(metrics, methods, maximum):
    """metrics：某数据集指标；methods：矩阵顺序；maximum：跨数据集统一色阶。"""
    rows = []
    for a in methods:
        cells = '<th>'+LABELS[a]+'</th>'
        for b in methods:
            value = metrics['infogap'][a][b]['value']
            strength = value/maximum if maximum else 0
            color = '#'+''.join(f'{round(x+(y-x)*strength):02x}' for x,y in zip((241,246,250),(32,86,133)))
            cells += f'<td style="background:{color};color:{"white" if strength>.55 else "#172b3b"}" title="G({LABELS[a]}, {LABELS[b]})={value}">{value:.4f}</td>'
        rows.append('<tr>'+cells+'</tr>')
    return '<div class="tablewrap"><table class="matrix"><tr><th>Row → column</th>'+''.join('<th>'+LABELS[m]+'</th>' for m in methods)+'</tr>'+''.join(rows)+'</table></div>'


def build_report(config, run):
    """config：本轮配置；run：结果目录。生成 Report.html 与 Audit.html，不改旧报告。"""
    methods = config['methods']
    results, rows_by_dataset = {}, {}
    for name in config['datasets']:
        folder = run/name
        audit = read_json(folder/'audit.json')
        if audit['status'] != 'passed' or any(audit[k+'_sha256'] != file_hash(folder/(k+'.json')) for k in ('predictions','metrics')):
            raise ValueError('Re-run the audit before publishing changed results.')
        results[name] = read_json(folder/'metrics.json')
        rows_by_dataset[name] = read_json(folder/'predictions.json')
    maximum = max(d['value'] for m in results.values() for row in m['infogap'].values() for d in row.values())
    body = '<header><h1>'+bi('Context compaction: paired probability study','上下文压缩：配对概率实验')+'</h1><p>'+bi(
        'Original-prompt first-draft summaries; fixed local compression caps. Actual lengths are reported.',
        '摘要统一使用原始提示词第一稿；本地方法使用固定上限。报告实际长度。')+'</p></header>'
    body += '<section><h2>'+bi('Protocol','实验协议')+'</h2><p>'+bi(
        f"{config['sample_size']} questions per dataset; seed {config['sample_seed']}; requested retention {config['retained_ratio']:.0%}. Reader: {config['reader']['model']}. Summary: {config['summary']['model']}.",
        f"每个数据集 {config['sample_size']} 题；种子 {config['sample_seed']}；目标保留率 {config['retained_ratio']:.0%}。Reader：{config['reader']['model']}；摘要模型：{config['summary']['model']}。")+'</p><p>'+bi(
        'Compression never sees questions or labels. Summaries are not revised or clipped, even if over target. Local methods use only measured length to tune their native rate. This is not an equal-length ranking. Probabilities are verbalized, not calibrated posteriors.',
        '压缩器不接收问题或标签。摘要即使超目标也不修订或裁剪；本地方法仅按长度调整原生保留率。这不是等长排名。概率为模型口头报告值，不是已校准后验。')+'</p></section>'
    for name, metrics in results.items():
        rows = []
        for m in methods:
            score = metrics['scores'][m]
            rows.append([LABELS[m], f"{score['accuracy']:.1%}", f"{score['brier']:.4f}", f"{score['log_loss']:.4f}",
                         f"{score['pooled_retained_ratio']:.1%}", str(score['over_budget_samples'])])
        body += '<section><h2>'+('HotpotQA' if name=='hotpotqa' else 'QASPER')+'</h2>'+table(
            ['Method','Accuracy ↑','Brier ↓','Log loss ↓','Retained','Over target'],rows)
        body += '<h3>'+bi('Bidirectional InfoGap','双向 InfoGap')+'</h3><p>'+bi(
            'G(row, column): largest normalized decision-payoff advantage of the row predictor. Both directions can be positive; this is not a net information gain or a dominance test.',
            'G(行,列)：行预测器相对列预测器的最大归一化决策收益优势。双向可同时为正，不能解读为净信息增益或支配检验。')+'</p>'+heatmap(metrics,methods,maximum)+'</section>'
    body += '<section><h2>'+bi('Scope and uncertainty','范围与不确定性')+'</h2><p>'+bi(
        'Exploratory binary subsets: HotpotQA distractor dev and text-evidence-consistent QASPER train. Full supplied contexts are retained before compression. InfoGap scans every probability breakpoint and its one-sided limits; factor 2 and unscaled results are both saved. Paired cluster-bootstrap intervals and probability diagnostics are in the per-dataset JSON files.',
        '探索性二元子集：HotpotQA distractor dev、QASPER train 中标签一致且正文证据可核验的题目。压缩前保留完整提供上下文。InfoGap 扫描所有概率断点及单侧极限，保存系数 2 和未归一化值。按上下文成组的配对 bootstrap 区间及概率诊断存于各数据集 JSON。')+'</p>'
    spending = read_json(run/'spending_reservations.json') if (run/'spending_reservations.json').exists() else []
    amount = sum(x['accounted_cny'] for x in spending)
    body += '<p>'+bi(f'Cumulative accounting including inherited history and unknown-usage reserves: CNY {amount:.2f}. Provider invoice is authoritative.',
                    f'包含继承历史和未知用量预留的累计记账：{amount:.2f} 元；实际扣费以平台账单为准。')+'</p>'
    if (run/'import_provenance.json').exists():
        body += '<p>'+bi('This report replays verified historical compression outputs and API responses; it is not a new independent model run.',
                        '本报告重放经核验的历史压缩输出与 API 响应，不是新的一轮独立模型生成。')+'</p>'
    body += '<p><a href="Audit.html">'+bi('Inspect individual examples','逐题审计')+'</a></p></section>'
    (run/'Report.html').write_text(page('Compaction study',body),encoding='utf-8')
    build_audit(config, run, rows_by_dataset)
    print('Reports:',run/'Report.html',run/'Audit.html')


def build_audit(config, run, rows_by_dataset):
    """嵌入原文、实际输入及原始 API 请求/响应，离线页面不含任何 API 密钥。"""
    records = []
    for name, rows in rows_by_dataset.items():
        for row in rows:
            conditions = {}
            for method,c in row['conditions'].items():
                raw = read_json(run/'api/reader/api_cache'/f"{c['prediction_request_hash']}.json")
                entry = {'text':c['text'],'tokens':c['tokens'],'p_yes':c['p_yes'],
                    'budget_met':c['budget_met'],'reader_request':raw['request'],'reader_response':raw['response']}
                if method=='llm_summary':
                    key=c.get('request_hash',c.get('original_request_hash'))
                    summary=read_json(run/'api/summary/api_cache'/f'{key}.json')
                    entry['summary_request']=summary['request']
                conditions[method]=entry
            records.append({'dataset':name,'id':row['id'],'question':row['question'],'label':row['label'],
                            'context':row['context'],'original_tokens':row['original_tokens'],'conditions':conditions})
    body='<header><h1>'+bi('Audit every example','逐题审计')+'</h1><p>'+bi('Select a dataset, question and method. This page works offline.','选择数据集、问题与方法。此页面完全离线。')+'</p></header>'
    body+='<section><select id="dataset"><option value="all">Both / 全部</option>'+''.join('<option>'+x+'</option>' for x in config['datasets'])+'</select> <select id="method">'+''.join(f'<option value="{m}">{LABELS[m]}</option>' for m in config['methods'])+'</select> <select id="filter"><option value="all">All / 全部</option><option value="over">Over target / 超预算</option></select><p><select id="example" style="max-width:100%"></select></p><div id="meta"></div></section><section class="grid"><div><h3>'+bi('Original source','原始上下文')+'</h3><pre id="original"></pre></div><div><h3>'+bi('Method input','方法输出')+'</h3><pre id="compressed"></pre></div></section><section><details><summary>'+bi('Actual reader and summary requests','实际 reader 和摘要请求')+'</summary><pre id="request"></pre></details><a href="Report.html">Back / 返回</a></section>'
    script='const DATA='+json.dumps(records,ensure_ascii=False).replace('<','\\u003c')+r''';
const $=s=>document.querySelector(s);let visible=[];
function refresh(){visible=DATA.filter(r=>($('#dataset').value==='all'||r.dataset===$('#dataset').value)&&($('#filter').value==='all'||r.conditions[$('#method').value].budget_met===false));$('#example').replaceChildren(...visible.map((r,i)=>new Option(`${i+1}. ${r.question}`,i)));show();}
function show(){const r=visible[Number($('#example').value)];if(!r){['meta','original','compressed','request'].forEach(x=>$('#'+x).textContent='');return;}const c=r.conditions[$('#method').value];$('#meta').textContent=`${r.id} | y=${r.label} | p(Yes)=${c.p_yes} | ${c.tokens}/${r.original_tokens} tokens`;$('#original').textContent=r.context;$('#compressed').textContent=c.text;$('#request').textContent=JSON.stringify({reader_request:c.reader_request,reader_response:c.reader_response,summary_request:c.summary_request},null,2);}
['dataset','method','filter'].forEach(x=>$('#'+x).onchange=refresh);$('#example').onchange=show;refresh();'''
    (run/'Audit.html').write_text(page('Compaction audit',body,script),encoding='utf-8')
