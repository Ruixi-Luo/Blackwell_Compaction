"""生成独立的本地压缩器导读与六条件结果快照，保留旧版报告。"""
import html
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.reporting import STYLE, bi, table
from blackwell.storage import read_json, write_json


def section(title_en, title_zh, content):
    return '<section><h2>' + bi(title_en, title_zh) + '</h2>' + content + '</section>'


def paragraph(en, zh):
    return '<p>' + bi(en, zh) + '</p>'


def main():
    config = read_json(ROOT / 'configs/local_comparison.json')
    run = ROOT / 'runs' / config['run_name']
    metrics = read_json(run / 'metrics.json')
    audit = read_json(run / 'audit.json')
    if audit['status'] != 'passed':
        raise ValueError('Audit must pass before publishing a result snapshot.')
    manifest = read_json(run / 'manifest.json')
    rows = read_json(run / 'predictions.json')
    runtime = manifest['identity']['local_compressors']
    docs = ROOT / 'docs'
    for source, name in [('report.html','Local_Compressors_Results.html'),
                         ('metrics.json','local_comparison_metrics.json'),
                         ('predictions.csv','local_comparison_predictions.csv'),
                         ('manifest.json','local_comparison_provenance.json'),
                         ('audit.json','local_comparison_audit.json'),
                         ('local_compression_check.json','local_compression_check.json')]:
        shutil.copyfile(run / source, docs / name)
    attribution = '<footer><p>BoolQ / SuperGLUE: passages and questions from the official dataset, CC BY-SA 3.0. <a href="https://github.com/google-research-datasets/boolean-questions">Dataset source</a>. Compression outputs are transformations of these passages.</p></footer>'
    results_file = docs / 'Local_Compressors_Results.html'
    results_file.write_text(results_file.read_text(encoding='utf-8').replace('</body>',attribution+'</body>'), encoding='utf-8')
    sections = []
    sections.append(section('What has been integrated', '接入了什么',
        paragraph('Two official text compressors now share one small interface. The compressor receives only context and a retained ratio. The evaluation question is added later by the unchanged API reader. No training is performed.',
                  '两个官方文本压缩器现在共用一个小接口。压缩器只接收上下文和保留率，评测问题在之后才由原有 API reader 加入，不进行训练。') +
        table(['Method',bi('Official implementation / model','官方实现／模型'),bi('Pipeline','流程')], [
            ['Selective Context','selective-context 0.1.4 · GPT-2 (124M)',bi('Text → sentence-level GPT-2 self-information → spaCy phrases → discard low-score phrases → text','文本 → GPT-2 按句计算自信息 → spaCy 短语聚合 → 删除低分短语 → 文本')],
            ['LLMLingua-2','llmlingua 0.2.2 · XLM-R-large MeetingBank',bi('Text → encoder token scores → mean score per word → percentile selection → text','文本 → 编码器 token 分类分数 → 词内平均 → 百分位筛选 → 文本')],
        ]) + '<p><a href="https://github.com/liyucheng09/Selective_Context">Selective Context · official repository</a> · <a href="https://aclanthology.org/2023.emnlp-main.391/">EMNLP 2023 paper</a><br><a href="https://github.com/microsoft/LLMLingua">LLMLingua-2 · official repository</a> · <a href="https://aclanthology.org/2024.findings-acl.57/">ACL Findings 2024 paper</a></p>'))
    example = '''from blackwell.compressors import create_compressor

# Create once; reuse the loaded model across many contexts.
compressor = create_compressor("llmlingua2")
# Or: create_compressor("selective_context")

compressed_text = compressor.compress(
    "Alice booked a train to Paris on Friday. The booking was later cancelled.",
    retained_ratio=0.5,
)
print(compressed_text)  # str, not a probability or model object
'''
    commands = r'''.\setup-local.ps1
.\run.ps1 -Mode Test
.\run.ps1 -Mode Report -Config configs/local_comparison.json
.\run.ps1 -Mode Audit -Config configs/local_comparison.json
.\run.ps1 -Mode Replay -Config configs/local_comparison.json
# Makes paid reader calls only when a complete request is not cached:
.\run.ps1 -Mode Run -Config configs/local_comparison.json'''
    sections.append(section('Use it in your experiment', '在实验中怎么用',
        paragraph('Run from the repository root using its .venv Python. The input and output are ordinary strings. Create a separate instance for each method; the first creation loads its local model. Inference needs no API key.',
                  '在仓库根目录使用 .venv 中的 Python 运行。输入和输出都是普通字符串；每个方法创建一个实例，首次创建时加载本地模型。压缩推理本身不需要 API key。') +
        '<pre>' + html.escape(example) + '</pre>' + paragraph('One-time installation and experiment commands:', '一次性安装与实验命令：') + '<pre>' + html.escape(commands) + '</pre>' +
        paragraph('The 30-item comparison config imports matching responses from the previous local pilot. A fresh clone has no private run caches: configs/local_fresh.json provides a four-condition comparison (full, no_context, Selective Context, LLMLingua-2) without reuse. Run it with -Config configs/local_fresh.json after checking the paid API budget. Do not change a completed run in place.',
                  '30 条对比配置会导入上一次本机 pilot 中请求完全一致的响应。新克隆不含 runs 缓存：configs/local_fresh.json 提供不依赖缓存的四组对比（原文、无上下文、SC、LLMLingua-2）；核对付费预算后使用 -Config configs/local_fresh.json 运行。不要原地修改已完成的运行。') +
        paragraph('Report and Audit can still read the original pilot. Strict Replay also verifies source hashes, so replaying that original code requires commit fb641ef in a separate checkout. The old HTML snapshots remain unchanged.',
                  'Report 和 Audit 仍可读取原 pilot。严格 Replay 还核验源码哈希，因此重放原代码需在独立 checkout 使用 fb641ef 提交。旧 HTML 快照保留原样。')))
    sections.append(section('Length conventions and fidelity', '长度约定与复现边界',
        paragraph('retained_ratio always means the fraction to keep. Selective Context receives reduce_ratio = 1 − retained_ratio; LLMLingua-2 receives rate = retained_ratio. We keep the native output unchanged, count it using the same Qwen tokenizer as the reader, and mark outputs above the Qwen-token budget.',
                  'retained_ratio 始终表示保留比例。Selective Context 收到 reduce_ratio = 1 − retained_ratio；LLMLingua-2 收到 rate = retained_ratio。原生输出不做二次截断，统一用 reader 的 Qwen tokenizer 计数，并标记超预算输出。') +
        paragraph('Thus equal nominal ratios do not imply equal lengths. Phrase filtering, word filtering and summarization allocate length differently. These pilot results cannot establish a fair matched-token winner. A later experiment should compare curves at matched measured budgets.',
                  '因此名义比例相同不代表输出等长。短语筛选、词筛选、摘要的长度分配不同，本次结果不能确立严格等 token 预算下的优胜者。后续应按实际预算比较曲线。') +
        paragraph('Selective Context uses its published package code; only its model-loading hook is overridden for a pinned local safetensors model on CPU. Its old spaCy 3.2.0 / click 8.0.4 dependency pins are bypassed: the tested versions are recorded below. This is an official-implementation integration with modern dependency adaptation, not an exact reproduction of the original paper environment. Sentences exceeding GPT-2’s 1024-token limit fail explicitly.',
                  'Selective Context 使用官方包源码，只覆盖模型加载钩子，使其加载固定的本地 safetensors 权重并在 CPU 运行。旧版 spaCy 3.2.0 / click 8.0.4 依赖锁定被绕过，实测版本见下表。这是适配现代依赖的官方实现接入，不是原论文环境的逐位复现。单句超过 GPT-2 的 1024-token 上限会明确报错。') +
        paragraph('LLMLingua-2 uses the dedicated second-generation method, with context-level filtering off and no forced question tokens or digit preservation. CPU float32, four PyTorch threads and a batch size of one keep this pilot portable. The reader remains Qwen3-30B-A3B-Instruct-2507 on SiliconFlow.',
                  'LLMLingua-2 调用专属二代方法，关闭 context-level filtering，不强制保留问题词或数字。此 pilot 使用 CPU float32、4 个 PyTorch 线程、batch size 1。Reader 仍为硅基流动 Qwen3-30B-A3B-Instruct-2507。')))
    ledger = read_json(run / 'api_attempts.json')
    usage = sum(x.get('usage',{}).get('total_tokens',0) for x in ledger)
    matrix_rows = []
    for method in config['methods']:
        s = metrics['scores'][method]
        matrix_rows.append([method, f"{s['pooled_retained_ratio']:.1%}", f"{s['accuracy']:.3f}",
                            f"{s['brier']:.4f}", f"{metrics['infogap']['full'][method]['value']:.4f}",
                            f"{metrics['infogap'][method]['full']['value']:.4f}"])
    sections.append(section('What actually ran', '实际跑了什么',
        paragraph(f"Exactly the same 30 BoolQ dev items, seed 20261017, with six paired conditions. {audit['unchanged_forecasts_from_previous_run']} old forecasts are unchanged. Both local methods compressed all 30 contexts while socket connections were blocked. The new run made {len(ledger)} reader API attempts ({usage:,} tokens reported by the provider).",
                  f"完全相同的 30 条 BoolQ dev 样本，seed 20261017，六组配对条件。原有 {audit['unchanged_forecasts_from_previous_run']} 条预测保持一致。两个本地方法在禁止 socket 连接的情况下，各完成 30 条上下文压缩。本轮 reader 新增 {len(ledger)} 次 API 尝试，服务端记录 {usage:,} tokens。") +
        table(['Method',bi('Actual kept','实际保留'),'Accuracy','Brier','G(Full, method)','G(method, Full)'], matrix_rows) +
        paragraph('InfoGap is directional: G(A,B) is the largest estimated payoff advantage of A over B. Both directions can be positive. The factor-2 convention, exact breakpoint scan and auxiliary probability scores are unchanged. A 30-item, short-context pilot using verbalized forecasts is not a general method ranking or a proof of Blackwell dominance.',
                  'InfoGap 有方向：G(A,B) 为 A 相对 B 的最大决策收益优势估计，两边都可能为正。系数 2 的约定、精确断点扫描和辅助概率指标保持不变。30 条短上下文样本、模型自报概率的 pilot 不能用于给方法普遍排名或证明 Blackwell 支配。') +
        '<p><a href="Local_Compressors_Results.html">' + bi('Open all six conditions, full matrix and 30 inspectable cases','查看六组条件、完整矩阵及 30 条可展开案例') + '</a></p>'))
    guide_rows = [
        ['blackwell/compressors/__init__.py',bi('Factory; creates only the selected compressor','工厂入口，仅创建所选压缩器')],
        ['blackwell/compressors/selective_context.py',bi('GPT-2 phrase-filter adapter; flips keep to remove ratio','GPT-2 短语筛选封装；保留率转删除率')],
        ['blackwell/compressors/llmlingua2.py',bi('Dedicated LLMLingua-2 adapter; context only','LLMLingua-2 专属封装；只传上下文')],
        ['blackwell/local_cache.py',bi('Caches context-only input and native output with fingerprints','缓存上下文输入及原生输出，带指纹')],
        ['blackwell/cache_reuse.py',bi('Validates and imports previous API responses','核验并导入旧 API 响应')],
        ['run_experiment.py',bi('Configured methods → compression → reader → paired metrics','按配置选择方法 → 压缩 → reader → 配对指标')],
        ['tools/check_local_compressors.py',bi('All 30 local compressions with network blocked','禁止网络后验证全部本地压缩')],
        ['tools/audit_run.py',bi('Checks inputs, raw responses, unchanged forecasts and all 36 InfoGap cells','核验输入、原始响应、旧预测一致性和 36 个 InfoGap 单元格')],
    ]
    code_details = ''
    for relative in ['blackwell/compressors/__init__.py','blackwell/compressors/selective_context.py','blackwell/compressors/llmlingua2.py','blackwell/local_cache.py','blackwell/compression.py']:
        code_details += '<details><summary>' + relative + '</summary><pre>' + html.escape((ROOT/relative).read_text(encoding='utf-8')) + '</pre></details>'
    sections.append(section('Audit the code', '怎么审计代码', table([bi('File','文件'),bi('Responsibility','职责')],guide_rows) +
        paragraph('Start with the two compress() signatures: no question or label parameter exists. Then inspect generate_predictions(): compression is complete before prediction_messages() adds the question. Audit checks the actual cached inputs, rather than relying only on interface names.',
                  '先看两个 compress() 的签名：没有 question 或 label 参数。再看 generate_predictions()：压缩完成后，prediction_messages() 才加入问题。审计还会核对实际缓存输入，而不是仅凭接口名称认定隔离。') + code_details))
    replay = read_json(run / 'replay_verification.json')
    if replay['status'] != 'passed':
        raise ValueError('Offline replay verification must pass.')
    sections.append(section('Validation evidence', '验证记录',
        paragraph('22 automated tests passed. Both compressors processed all 30 items without network access. The audit checked all 180 forecasts, 60 local compression records, 120 unchanged prior forecasts and 36 InfoGap cells using an independent two-action payoff calculation. A replay with API keys removed, sockets blocked and local model loading blocked made zero API attempts and reproduced predictions, metrics and the API ledger byte for byte.',
                  '22 项自动化测试通过。两种压缩器均在断网下处理了全部 30 条样本。审计核验了 180 条预测、60 条本地压缩记录、120 条旧预测的一致性，并以独立两动作收益计算核验 36 个 InfoGap 单元格。移除 API key、禁止 socket 和本地模型加载后，完整重放新增 API 次数为 0，预测、指标、API 账本逐字节一致。') +
        paragraph('The longest item has 599 XLM-R tokens before chunking. Its native chunks contain 493 and 106 tokens; with special tokens, both fit the 512-token encoder input. All 30 items were checked for this limit. Model downloads, CPU inference time and API-token usage are different costs and should not be conflated.',
                  '最长样本在分块前有 599 个 XLM-R token，原生分块分别为 493 和 106 个；加特殊 token 后均满足编码器 512-token 输入上限。全部 30 条样本都做了此项检查。模型下载、本地 CPU 推理时间与 API token 用量属于不同成本，不应混为一谈。')))
    sections.append(section('Frozen assets and local storage', '固定版本与本地存储',
        table(['Asset','Revision'], [[m, '<code>'+s['revision']+'</code>'] for m,s in runtime['assets'].items()]) +
        table(['Package','Version'],list(runtime['packages'].items())) +
        paragraph('Model files are pinned by repository revision and per-file SHA256 in configs/local_models.lock.json. Weight downloads total about 2.8 GB (decimal); Python packages and download caches take additional space. All are stored under data/, .venv/ and .cache/ on D:, and are excluded from Git. The published docs contain no API keys.',
                  'configs/local_models.lock.json 固定模型仓库 revision 与每个文件的 SHA256。两个模型权重合计约 2.8 GB（十进制），Python 依赖和下载缓存另占空间。全部存于 D 盘的 data/、.venv/、.cache/，不提交到 Git。公开文档不包含 API key。')))
    document = '<!doctype html><html lang="en" data-language="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local compressors · Code and comparison</title><style>'+STYLE+'</style></head><body><header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Selective Context &amp; LLMLingua-2','Selective Context 与 LLMLingua-2')+'</h1><p>'+bi('Module guide · Paired BoolQ pilot · 2026-10-17','模块导读 · BoolQ 配对 pilot · 2026-10-17')+'</p></header>'+''.join(sections)+'<script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}</script></body></html>'
    (docs/'Local_Compressors.html').write_text(document, encoding='utf-8')
    print(docs/'Local_Compressors.html')


if __name__ == '__main__':
    main()
