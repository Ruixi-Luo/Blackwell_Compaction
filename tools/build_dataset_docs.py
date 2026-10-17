"""把已核验的 census 和实验结果编成中英文导读；不会调用模型或改实验。"""
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table
from blackwell.storage import read_json, write_json


def paragraph(en, zh):
    return '<p>' + bi(en, zh) + '</p>'


def section(en, zh, content):
    return '<section><h2>' + bi(en, zh) + '</h2>' + content + '</section>'


def main():
    inventory = read_json(ROOT/'docs/dataset_inventory.json')
    config = read_json(ROOT/'configs/hotpotqa.json')
    run = ROOT/'runs'/config['run_name']
    manifest = read_json(run/'manifest.json')
    if manifest['status'] != 'complete':
        raise RuntimeError('Finish the HotpotQA run before publishing its results.')
    metrics = read_json(run/'metrics.json')
    rows = read_json(run/'predictions.json')
    lengths = [r['original_tokens'] for r in rows]
    hotpot = inventory['hotpotqa']
    sections = []
    sections.append(section('What changed', '这次增加了什么', paragraph(
        'BoolQ remains the two-item quick check. HotpotQA native yes/no is now a runnable dataset using the same reader and metrics. QASPER has been counted from the complete v0.3 release, but no QASPER LLM predictions have been run.',
        'BoolQ 保留为两条样本的快速检查；HotpotQA 原生 yes/no 已接入同一套 reader 和指标。QASPER 已完整核验 v0.3 的规模，但尚未运行 LLM 预测。') +
        paragraph('<a href="HotpotQA_Results.html">Open the 30-item HotpotQA results →</a>', '<a href="HotpotQA_Results.html">打开 30 条 HotpotQA 的逐条实验结果 →</a>')))
    sections.append(section('HotpotQA: a useful intermediate step', 'HotpotQA：更长上下文的过渡实验',
        table([bi('Population','统计范围'),bi('Questions','题数'), 'Yes / No', bi('Mean context tokens','平均上下文 token'),bi('Range','范围')], [
            [bi('All dev native yes/no','dev 全部原生 yes/no'), hotpot['questions'], f"{hotpot['yes']} / {hotpot['no']}", f"{hotpot['context_tokens']['mean']:.1f}", f"{hotpot['context_tokens']['min']}–{hotpot['context_tokens']['max']}"],
            [bi('Fixed-seed pilot','固定种子 pilot'),len(rows),f"{metrics['positive_labels']} / {len(rows)-metrics['positive_labels']}",f"{sum(lengths)/len(lengths):.1f}",f"{min(lengths)}–{max(lengths)}"]]) +
        paragraph('There are 7,405 dev items in the pinned distractor file. We keep only native answers yes/no, without balancing labels. All 458 eligible items are comparison questions. Of these, 447 provide 10 paragraphs and 11 provide 2–9. All supplied paragraphs and titles are retained in their original order; no gold evidence selection or question-based reordering is performed.',
                  '固定版本的 distractor dev 共 7,405 题，只筛原生 yes/no 答案，不平衡标签。458 道入选题全部为 comparison 类：447 道有 10 段，11 道有 2–9 段。保留全部所提供段落及标题和原始顺序，不按 gold evidence 筛选，不用问题重排。') +
        paragraph('This is roughly 1.2k-token context, longer than the previous BoolQ pilot, but not a 4k–16k long-context benchmark. The context retrieval was question-conditioned during dataset construction; only our compression step is query-blind. Results apply to this binary comparison subset, not all HotpotQA questions.',
                  '这里约为 1.2k token 上下文，比此前的 BoolQ 长，但尚不属于 4k–16k 的长上下文评测。数据集构造时的检索依赖问题，只有我们的压缩环节是 query-blind。结果适用于该二元 comparison 子集，不能代表全部 HotpotQA。')))
    summary_rows=[]
    for method, s in metrics['scores'].items():
        summary_rows.append([method, f"{s['accuracy']:.3f}",f"{s['brier']:.4f}",f"{s['pooled_retained_ratio']:.1%}",
                             f"{metrics['infogap']['full'][method]['value']:.4f}",f"{metrics['infogap'][method]['full']['value']:.4f}"])
    sections.append(section('The pilot, with both InfoGap directions', 'Pilot 结果与双向 InfoGap',
        table(['Method','Accuracy ↑','Brier ↓',bi('Actual retained','实际保留率'),'G(Full, M)','G(M, Full)'],summary_rows) +
        paragraph('Reader: SiliconFlow Qwen3-30B-A3B-Instruct-2507, thinking off, temperature 0. Compressors: official Selective Context (GPT-2, phrase) and LLMLingua-2 (XLM-R-large MeetingBank), local CPU. Each uses a nominal 50% retained ratio. Actual output lengths differ and are not post-truncated. The reader reports p(Yes); these forecasts are not assumed to be true Bayesian posteriors.',
                  'Reader 固定为硅基流动 Qwen3-30B-A3B-Instruct-2507，关闭思考、temperature=0。压缩使用官方 Selective Context（GPT-2、phrase）和 LLMLingua-2（XLM-R-large MeetingBank），本地 CPU 运行，名义保留率均为 50%。实际长度有差异，不追加截断。Reader 报告 p(Yes)，不预设这些概率等于真实贝叶斯后验。') +
        paragraph('A larger G(A,B) means a larger estimated decision-payoff advantage of A over B. Read both directions; n=30 is exploratory. These numbers neither prove population Blackwell dominance nor identify a generally superior compressor.',
                  'G(A,B) 越大，表示 A 相对 B 的最大决策收益优势估计越大，必须双向阅读。30 条仅用于探索，不能证明总体 Blackwell 支配或某个压缩器普遍更好。')))
    qrows=[]
    for split, value in inventory['qasper'].items():
        strict=value['strict_boolean']; text=value['text_verified']
        qrows.append([split,value['all_questions'],strict['questions'],strict['papers'],f"{strict['yes']} / {strict['no']}",
                      text['questions'],text['papers'],f"{text['yes']} / {text['no']}"])
    sections.append(section('QASPER: what is actually usable?', 'QASPER：究竟有多少可用样本？',
        paragraph('The counting unit is a question, not an answer annotation. Strict eligibility requires a nonempty body, at least one annotation, every annotation classified as boolean by the official evaluator precedence, and unanimous yes/no labels. Unanswerable takes precedence; false means No, whereas null is missing. No majority vote or first-annotation shortcut is used.',
                  '计数单位是问题，不是答案标注。严格入选要求正文非空、至少一份标注、全部标注按官方 evaluator 优先级都属于 boolean，且 yes/no 一致。unanswerable 优先，false 表示 No，null 才是缺失。不用多数投票或第一份答案代替一致性检查。') +
        table(['Split',bi('All Q','全部题'),bi('Strict Q','严格一致题'),bi('Papers','论文数'),'Yes / No',bi('Text-verified Q','文字证据可核验题'),bi('Papers','论文数'),'Yes / No'],qrows) +
        paragraph('The more conservative text-verified subset requires nonempty evidence in every annotation, no FLOAT SELECTED figure/table marker, and every evidence passage present in the proposed full-text input after whitespace normalization. This is an additional experiment-design filter, not an official QASPER split. It does not certify annotation correctness or complete PDF extraction.',
                  '更保守的文字证据子集要求每份标注 evidence 非空、无 FLOAT SELECTED 图表标记，且所有证据段落在规范空白后的全文输入中都能找到。这是额外实验设计筛选，不是官方 QASPER 划分，也不能保证标注正确或 PDF 抽取完整。') +
        paragraph('The conservative filter removes many No questions whose evidence is empty, shifting the label distribution. Therefore both counts are retained: do not describe the conservative subset as an unbiased sample, and do not equate No with unanswerable.',
                  '保守筛选会排除许多 evidence 为空的 No 问题，从而改变标签比例。因此同时保留两套统计：不能把保守子集称为无偏样本，也不能把 No 当作 unanswerable。')))
    erows=[]; lrows=[]
    for split, value in inventory['qasper'].items():
        exc=value['exclusions']; ev=value['evidence_status']
        erows.append([split,exc.get('no_boolean',0),exc.get('mixed_answer_types',0),exc.get('conflicting_boolean_labels',0),exc.get('missing_full_text',0),ev.get('figure_or_table',0),ev.get('missing_evidence',0)])
        for name in ['strict_boolean','text_verified']:
            s=value[name]; t=s['tokens_per_question']; b=t['bins']
            lrows.append([split,name,f"{t['mean']:.0f}",t['p50'],t['p90'],f"{t['min']}–{t['max']}",b['4000-7999']+b['8000-15999'],s['papers_with_multiple_questions']])
    sections.append(section('Where the samples go', '样本在哪一步被排除',
        table(['Split',bi('No boolean annotation','没有 boolean 标注'),bi('Mixed types','混合类型'),bi('Yes/No conflict','Yes/No 冲突'),bi('Missing body','缺正文'),bi('Figure/table¹','含图表¹'),bi('Missing evidence¹','缺证据¹')],erows) +
        paragraph('First four exclusion columns partition questions excluded before strict eligibility (no-annotation count is zero here). ¹The last two are within the strict subset and are removed only by the conservative text filter. None of the remaining textual evidence passages failed the text-presence check.',
                  '前四列是严格一致筛选前被排除的问题，本版本无空标注问题。¹后两列是在严格一致子集内部，仅被保守文字筛选排除的题，不能和前四列混用分母。其余文字证据均通过全文可见性检查。')))
    sections.append(section('Full-paper length and the next experiment', '全文长度与下一步实验',
        table(['Split','Subset',bi('Mean tokens','平均 token'),'P50','P90',bi('Range','范围'),bi('4k–16k Q¹','4k–16k 题¹'),bi('Multi-Q papers','多题论文')],lrows) +
        paragraph('¹4,000 ≤ tokens < 16,000. Tokens use the same frozen Qwen tokenizer, excluding prompts and questions. The proposed QASPER input is title + abstract + section headings + all body paragraphs, without selecting gold evidence or adding figure images/table bodies. These lengths are question-weighted; the JSON separately records unique-paper-weighted lengths.',
                  '¹4,000 ≤ token &lt; 16,000。沿用固定 Qwen tokenizer，只计上下文，不含提示词和问题。QASPER 候选输入为标题＋摘要＋章节标题＋全部正文段落，不筛 gold evidence，不额外加入图片或表格内容。此表按题目加权，JSON 另存按独立论文加权的长度。') +
        paragraph('Use dev for protocol decisions and keep test predictions held out. Under the conservative rule, dev has 51 questions (24 in 4k–16k), test has 88 (59 in 4k–16k). That is enough for a pilot, not a large benchmark. Questions sharing a paper must reuse the same compressed context and be grouped by paper in splits and resampling; multiple queries per paper are limited in this subset.',
                  '建议用 dev 定协议，暂不看 test 的预测结果。保守规则下 dev 51 题（4k–16k 为 24 题），test 88 题（4k–16k 为 59 题），适合 pilot，但不是大规模 benchmark。同论文的题应复用同一压缩文本，划分和重采样按论文分组；本子集多题论文的数量也有限。')))
    pipeline = 'config → load_dataset → context → compressor → compressed context\nreader(compressed context, question) → p_yes\nevaluate(p_yes, label) → InfoGap / Brier / log loss'
    files=[('configs/hotpotqa.json','Choose dataset, seed, sample count, methods and API limit.','选数据、seed、样本数、方法和 API 上限。'),
           ('blackwell/data.py','Dispatch to a loader; keep the shared example format.','选择 loader，保持统一样本格式。'),
           ('blackwell/datasets/hotpotqa.py','Filter native binary labels and assemble all supplied paragraphs.','筛原生二元标签，拼接全部提供段落。'),
           ('blackwell/datasets/qasper.py','Interpret annotations and define census eligibility.','解释标注并定义统计规则。'),
           ('run_experiment.py','Prepare → compress → predict → evaluate; inspect this main flow next.','准备→压缩→预测→评估，接着阅读主流程。'),
           ('blackwell/compressors/','Official-method adapters accept context and ratio only.','官方方法适配器仅接收上下文和保留率。'),
           ('blackwell/metrics.py','Pure mathematical evaluation, separate from API calls.','纯数学评估，与 API 调用分离。'),
           ('tests/test_datasets.py','Examples of label, leakage and regression checks.','学习标签、信息隔离及回归检查。')]
    links=[]
    for path,en,zh in files:
        url='https://github.com/Ruixi-Luo/Blackwell_Compaction/'+('tree/main/' if path.endswith('/') else 'blob/main/')+path
        links.append([f'<a href="{url}">{path}</a>',bi(en,zh)])
    sections.append(section('Read the project in this order', '按这个顺序读科研代码', '<pre>'+html.escape(pipeline)+'</pre>'+table(['File',bi('Responsibility','职责')],links) +
        paragraph('A loader returns a list of plain dictionaries with context, question, label, id and context_group, plus provenance. Compression is deliberately called with context only, never the whole dictionary. A new dataset needs a small loader and one dispatch entry; do not rewrite the reader or InfoGap. The reader remains deliberately restricted to the verified SiliconFlow/Qwen configuration.',
                  'Loader 返回普通字典列表（context、question、label、id、context_group 等）和来源记录。压缩只接收 context，不接收整个字典。新增数据只需一个小 loader 和入口注册，不重写 reader 或 InfoGap。Reader 目前仍明确限制为已验证的硅基流动/Qwen 组合。')))
    commands = r'''# Run from the repository root. All caches stay on D:.
$env:PIP_CACHE_DIR = "$PWD/.cache/pip"
$env:TEMP = "$PWD/.cache/tmp"; $env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\run.ps1 -Mode Test
.\run.ps1 -Mode Prepare -Config configs/hotpotqa.json
# Calls the paid reader on cache misses; 30 × 4 forecasts, maximum 150 attempts.
.\run.ps1 -Mode Run -Config configs/hotpotqa.json
.\run.ps1 -Mode Audit -Config configs/hotpotqa.json
.\.venv\Scripts\python.exe tools/check_local_chunking.py --config configs/hotpotqa.json
.\.venv\Scripts\python.exe tools/verify_offline_replay.py --config configs/hotpotqa.json
# QASPER inventory only; no LLM calls. Omit --offline for the first download.
.\.venv\Scripts\python.exe tools/inspect_datasets.py --offline
# Two-item BoolQ quick check, no local compressor weights required.
.\run.ps1 -Mode Run -Config configs/quickcheck.json'''
    sections.append(section('Run and audit it yourself', '自己运行与审计', '<pre>'+html.escape(commands)+'</pre>' +
        paragraph('Run setup.ps1 first on a new clone, and setup-local.ps1 once for local compressor weights. Old runs are preserved. Changing code, configuration or data requires a new run_name; reports can still be recomputed from old predictions. Strict replay requires the corresponding source version. Raw datasets, environments and API caches remain ignored by Git.',
                  '新克隆先运行 setup.ps1，本地压缩器额外运行一次 setup-local.ps1。旧实验全部保留。修改代码、配置或数据要换 run_name；旧预测仍可重算报告，严格重放需匹配源码版本。原始数据、环境及 API 缓存均不进入 Git。')))
    detail = '<details><summary>'+bi('Inspect the exact census and source hashes','展开精确统计与来源校验值')+'</summary><pre>'+html.escape(json.dumps({**inventory,'qasper':{s:{k:v for k,v in d.items() if k!='eligible_questions'} for s,d in inventory['qasper'].items()}},ensure_ascii=False,indent=2))+'</pre></details>'
    audit = read_json(run/'audit.json')
    replay = read_json(run/'replay_verification.json')
    chunks = read_json(run/'local_chunk_check.json')
    if any(result['status'] != 'passed' for result in [audit, replay, chunks]):
        raise RuntimeError('All audits must pass before publishing the report.')
    ledger = read_json(run/'api_attempts.json')
    usage = sum(e.get('usage',{}).get('total_tokens',0) for e in ledger)
    verification = paragraph(
        f"Verified {audit['forecasts']} paired forecasts and {audit['local_compressions_checked']} local compression records against their actual inputs. Offline replay reproduced predictions, metrics and the API ledger byte for byte, with {replay['new_api_attempts']} new API calls. The HotpotQA run used {len(ledger)} API attempts and {usage:,} reported input+output tokens; QASPER used no LLM calls.",
        f"已核验 {audit['forecasts']} 条配对概率预测及 {audit['local_compressions_checked']} 条本地压缩记录的真实输入。离线重放逐字节复现预测、指标和 API 账本，新增 API 调用为 {replay['new_api_attempts']}。HotpotQA 使用 {len(ledger)} 次 API 尝试、服务端报告 {usage:,} 个输入＋输出 token；QASPER 未调用 LLM。")
    detail += '<details><summary>'+bi('LLMLingua-2 chunk boundary check','LLMLingua-2 分块边界核验')+'</summary>' + paragraph(
        f"Across these 30 inputs, no content token was removed by the classifier input-length cap. The official implementation produced {chunks['chunks_with_only_final_separator_trimmed']} chunks with 511 content tokens; adding start/end tokens gives 513, so its native 512-position slicing removes only the final separator. We retain and disclose this upstream behavior rather than change the baseline.",
        f"这 30 条输入没有正文 token 被分类器的长度上限截去。官方实现有 {chunks['chunks_with_only_final_separator_trimmed']} 个块包含 511 个正文 token，加首尾标记后为 513，原生 512 位裁剪仅去掉末尾分隔符。本项目保留并记录这一官方行为。")+'</details>'
    sections.append(section('Sources and reproducibility', '来源与可复核性',
        '<p><a href="https://hotpotqa.github.io/">HotpotQA · EMNLP 2018 · CC BY-SA 4.0</a><br><a href="https://aclanthology.org/2021.naacl-main.365/">QASPER · NAACL 2021 · CC BY 4.0</a><br><a href="https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py">QASPER official evaluator</a><br><a href="https://huggingface.co/datasets/allenai/qasper/blob/main/qasper.py">QASPER v0.3 release URLs</a><br><a href="https://arxiv.org/html/2507.12094v2">InfoGap theory and normalization source</a></p>' +
        paragraph('The original HotpotQA server failed during this setup; the loader uses a pinned, SHA-256-verified Hugging Face mirror. QASPER uses the author-provided train/dev and test archives. The full inventory JSON includes eligible question IDs and per-paper token counts, enabling independent selection checks without committing raw documents.',
                  '本次 HotpotQA 原服务器不可用，loader 使用固定 revision 和 SHA-256 的 Hugging Face 镜像。QASPER 使用作者提供的 train/dev 和 test 归档。完整 inventory JSON 保存入选 question ID 及全文 token 数，便于独立复核筛选，不提交原始文档。') + verification + detail))
    output='<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blackwell Compaction · Datasets &amp; code guide</title><style>'+STYLE+'</style><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('From BoolQ to longer contexts','从 BoolQ 走向更长上下文')+'</h1><p>'+bi('HotpotQA pilot · QASPER feasibility · A readable research project','HotpotQA pilot · QASPER 可用规模 · 可读的科研项目')+'</p></header>'+''.join(sections)+'<script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}</script></body></html>'
    (ROOT/'docs/Datasets_and_Experiment_Guide.html').write_text(output,encoding='utf-8')
    attribution = '<footer><p>HotpotQA: Yang, Qi, Zhang et al. (EMNLP 2018). Questions, passages and experimental compressed adaptations are shared under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>. <a href="https://hotpotqa.github.io/">Original dataset and authors</a>. Compression outputs are machine-generated transformations.</p></footer>'
    report = (run/'report.html').read_text(encoding='utf-8').replace('</body>',attribution+'</body>')
    (ROOT/'docs/HotpotQA_Results.html').write_text(report,encoding='utf-8')
    for name in ['metrics.json','audit.json','replay_verification.json','predictions.csv','local_compression_check.json','local_chunk_check.json']:
        shutil.copyfile(run/name,ROOT/'docs'/('hotpotqa_'+name))
    write_json(ROOT/'docs/hotpotqa_provenance.json',manifest)
    print('Wrote bilingual dataset guide and verified HotpotQA result snapshots.')


if __name__ == '__main__':
    main()
