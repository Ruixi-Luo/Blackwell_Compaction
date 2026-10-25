"""Build a self-contained bilingual walkthrough from real current source files."""
import ast
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'docs/Clean_Code_Walkthrough.html'
FILES = [
 ('run.py','Start here','从这里开始','A command selects one stage; main orchestrates, while helpers do the work.','命令行选择阶段；main 只组织流程，具体操作交给小函数。'),
 ('experiment/config.py','Experiment settings','实验设置','JSON describes what to run. Validation catches invalid choices before any model call.','JSON 描述要跑什么；先检查配置，避免模型调用后才发现参数错误。'),
 ('experiment/pipeline.py','The execution order','运行主线','Read prepare → compress → predict → evaluate → audit. Data flows forward; labels never flow into compression.','按 prepare → compress → predict → evaluate → audit 阅读。数据向前流动，标签不进入压缩。'),
 ('experiment/data.py','Sampling from datasets','从数据集抽题','Load the eligible pool, sort by ID, then sample without replacement using a fixed seed. No historical prediction file is needed.','加载合格样本池，按 ID 排序，再用固定种子无放回抽样，不依赖历史预测文件。'),
 ('experiment/compression.py','Context to compressed context','上下文变成压缩上下文','Full and No context are controls. SC, LLMLingua-2 and DAC use bounded native-rate search. Summary is handled separately as one draft.','Full 与 No context 是对照；SC、LLMLingua-2、DAC 有限次试探原生比例；摘要单独采用第一稿规则。'),
 ('experiment/summary.py','One original summary draft','只生成原始摘要第一稿','The prompt is byte-identical to the selected original drafts. A complete over-target output stays; a truncated response fails.','提示词与本轮原始稿逐字一致。完整但超目标的输出保留；被截断的响应报错。'),
 ('experiment/api.py','Reader and API boundary','Reader 与 API 边界','Only predict_one receives a question. Requests are cached by the complete payload. New paid requests require --allow-api.','只有 predict_one 接收问题。用完整请求确定缓存身份；新增付费请求要求显式 --allow-api。'),
 ('experiment/evaluation.py','Probabilities to metrics','概率变成指标','Require all paired examples, compute scores, scan both InfoGap directions and bootstrap by source context.','要求全部配对题目齐全，再算评分、双向 InfoGap，并按原文上下文成组 bootstrap。'),
 ('experiment/reporting.py','Readable results','可读结果页','Report and audit HTML read validated saved results. They never call a model.','结果和逐题审计 HTML 读取已验证的保存结果，不调用模型。'),
 ('experiment/import_results.py','Optional history importer','可选历史导入器','This is a migration utility, not a dependency of a fresh run. It validates samples, original summary prompts, raw responses and inherited costs.','这是迁移工具，不是新实验的前置依赖。它核验选题、摘要原始提示词、原始响应和继承费用。'),
 ('blackwell/metrics.py','Inside InfoGap','深入 InfoGap','Pure mathematics, with no API or GPU dependency. The sweep checks breakpoints and left limits exactly.','纯数学函数，不依赖 API 或 GPU。扫描全部断点和左极限，避免粗网格漏掉最大值。'),
 ('blackwell/prompts.py','The unchanged reader prompt','保持不变的 reader 提示词','The clean path uses prediction_messages; the older compression_messages helper is retained for historical experiments only.','clean 路径使用 prediction_messages；旧 compression_messages 仅为历史实验保留。'),
 ('blackwell/datasets/hotpotqa.py','HotpotQA details','HotpotQA 加载细节','Keep all supplied paragraphs. The original retrieval is question-conditioned, while our compressor remains query-blind.','保留提供的全部段落。原数据检索与问题有关；我们的压缩器仍然不看问题。'),
 ('blackwell/datasets/qasper.py','QASPER details','QASPER 加载细节','Keep the whole paper. Evidence filtering defines the eligible subset; it does not crop or annotate the compressor input.','保留全文。证据筛选只定义合格子集，不用于裁剪正文或标注压缩输入。'),
 ('blackwell/compressors/selective_context.py','Selective Context adapter','Selective Context 适配器','Official GPT-2 phrase scoring, with a lossless adapter for sentences beyond the model limit.','官方 GPT-2 短语打分，对超过模型上限的长句做不丢字符的适配。'),
 ('blackwell/compressors/llmlingua2.py','LLMLingua-2 adapter','LLMLingua-2 适配器','Official second-generation token classification; no question is supplied.','官方二代 token 分类路径，不提供问题。'),
 ('blackwell/compressors/dac.py','DAC adapter','DAC 适配器','Blocked attention computation preserves every causal predecessor. Query blocks here mean attention rows, not evaluation questions.','分块计算注意力，保留全部因果前文。这里的 query block 指注意力查询行，不是评测问题。'),
 ('blackwell/compressors/chunking.py','Lossless length adaptation','不丢文本的长度适配','Split only when a local model limit requires it; source text is preserved.','仅在本地模型长度上限要求时分段，保留原文内容。'),
 ('blackwell/compressors/runtime.py','Models and device','模型与设备','Pin assets, verify hashes, configure D-drive caches and explicitly choose CPU/CUDA.','固定模型资产、校验哈希，把缓存放在 D 盘，并显式选择 CPU/CUDA。'),
 ('blackwell/client.py','Exact API caching','精确 API 缓存','Cache keys include model, prompt and generation settings. Invalid JSON or incomplete output never becomes a fabricated probability.','缓存键含模型、提示词和生成设置；无效 JSON 或不完整输出绝不填成伪造概率。'),
 ('blackwell/streaming.py','Complete streamed responses','完整流式响应','A stream must end normally and include its terminator. Partial text is not a valid summary.','流必须正常结束并有终止标记；部分文本不是有效摘要。'),
 ('blackwell/cost_guard.py','Cost reservation','费用预留','Reserve before sending; settle with returned usage, retaining conservative allowances for unknown usage. Local rates are estimates, not the platform invoice.','先预留再请求，按返回 usage 结算，未知用量继续保守预留；本地价格是估算，不是平台账单。'),
 ('blackwell/tokenization.py','Token counting','Token 计数','Qwen tokens define measurement. Clean code uses count, never the historical head_tail clipping helper.','以 Qwen token 统一计量。clean 代码使用 count，不调用旧的 head_tail 裁剪函数。'),
 ('blackwell/storage.py','Saved evidence','文件与证据','Stable hashes identify exact contents; atomic writes avoid partially written JSON.','稳定哈希标识准确内容；先写临时文件再替换，避免半个 JSON。')]

# Every parameter encountered in the embedded source has an explanation.
PARAM_TEXT = '''self|The current object; Python supplies it automatically.|当前对象；Python 自动传入，不需要手动填写。
config|Experiment settings, or the selected reader/summary settings in an API helper.|实验配置；API 辅助函数中则是选出的 reader/summary 模型配置。
run|The independent output directory for this run_name.|本次 run_name 对应的独立输出目录。
run_dir|Directory holding this client's cache and attempt ledger.|此客户端保存缓存和尝试账本的目录。
folder|The working data/result/model directory used by this function.|此函数处理的数据、结果或模型目录。
path|A filesystem path to read, validate or write.|要读取、验证或写入的文件路径。
source|The explicitly selected historical source directory; read only during import.|显式选定的历史来源目录；导入时只读。
context|The full source text or reader evidence; never an example dict containing labels.|完整原文或 reader 使用的证据文本；不能是含标签的整条题目字典。
question|The downstream question; only the reader receives it. Dataset helpers use it to read annotations.|下游问题；只有 reader 接收。数据集辅助函数用它读取标注。
budget|Target retained Qwen token count, derived from the original source before compression.|压缩前按原文计算的目标 Qwen token 保留数量。
retained_ratio|Fraction of source tokens to KEEP, e.g. 0.5; not the fraction deleted.|要保留的比例，如 0.5；不是删除比例。
method|One named compression/control condition.|一个压缩方法或对照条件的名称。
methods|The ordered list of conditions, also the row/column order of the matrix.|条件的有序列表，也决定矩阵行列顺序。
tokenizer|The shared Qwen tokenizer object used to measure lengths.|用于统一测量长度的 Qwen tokenizer 对象。
engine|A loaded local compressor, reused for multiple contexts.|已经加载的本地压缩模型，供多个上下文复用。
client|The model client that verifies caches and, when authorized, sends requests.|检查缓存并在允许时发送请求的模型客户端。
allow_api|False uses cached responses only; True permits paid cache misses within limits.|False 仅允许缓存响应；True 才允许在限额内为缺失缓存付费调用。
offline|True forbids downloads and new model inference; missing evidence raises an error.|True 禁止下载和新增模型推理；缺少缓存就报错。
role|Either reader or summary; separates their model settings and cache ledgers.|reader 或 summary；区分模型设置与缓存账本。
messages|Ordered role/content messages actually sent to the model.|实际发送给模型、按顺序排列的 role/content 消息。
maximum|API output-token ceiling; in heatmap, the shared largest gap for color scaling.|API 输出 token 上限；在 heatmap 中是统一色阶的最大 gap。
max_tokens|Maximum output tokens requested from the API, not the source-retention budget.|API 输出 token 上限，不是原文保留预算。
json_mode|Whether to request a JSON object response, used for reader probabilities.|是否要求 JSON 对象回答；reader 概率请求使用 True。
key|Exact request hash; in a helper it identifies the stored response.|准确请求哈希，用于定位保存的响应。
request_hash|SHA-256 identity of a model request; used to trace response provenance.|模型请求的 SHA-256 身份，用于追溯响应来源。
expected|Reconstructed request identity that saved data must equal.|重新构造的请求身份，保存数据必须与它一致。
result|A complete compression result, including text, measured length and metadata.|完整压缩结果，包含文本、实测长度和元数据。
provenance|Source URLs/hashes or optional historical-import evidence.|来源网址/哈希，或可选的历史导入证据。
candidates|Generated outputs in order; local selection sees lengths, never labels.|按生成顺序排列的候选；本地选择只看长度，不看标签。
manifest|Frozen configuration, code hashes, sample identities and run status.|冻结的配置、代码哈希、样本身份与运行状态。
rows|Dataset records or saved per-question predictions, according to the function.|原始数据记录，或保存的逐题预测；具体由所在函数决定。
rows_by_dataset|Saved prediction rows grouped by dataset name.|按数据集名称分组的预测记录。
metrics|Computed scores, InfoGap matrices and uncertainty summaries.|计算好的评分、InfoGap 矩阵和不确定性结果。
probabilities|An ordered list of numerical p(Yes), between zero and one.|按题目顺序排列、介于 0 与 1 的 p(Yes) 数值列表。
probabilities_a|Probabilities of predictor A, paired with B and labels by question.|预测器 A 的概率，与 B 及标签逐题配对。
probabilities_b|Probabilities of predictor B, in the same question order as A.|预测器 B 的概率，题目顺序与 A 相同。
a|Predictor A's probabilities in the metric functions.|指标函数中预测器 A 的概率列表。
b|Predictor B's probabilities in the metric functions.|指标函数中预测器 B 的概率列表。
labels|Ground-truth binary answers, used in evaluation, never in compression.|真实二元答案，只用于评估，绝不进入压缩。
groups|Context IDs for cluster bootstrap; questions sharing a paper stay together.|成组 bootstrap 的上下文 ID；同一论文的题目一起抽样。
repetitions|Number of bootstrap resamples; more draws reduce Monte Carlo noise.|bootstrap 重采样次数；更多次数可减少模拟噪声。
seed|Random seed controlling reproducible sampling/bootstrap.|控制可复现抽样或 bootstrap 的随机种子。
size|Number of questions requested from the eligible dataset pool.|从合格样本池抽取的题目数量。
sample_size|Number of selected questions, validated against pool size.|抽取题数，必须不超过样本池规模。
name|Dataset identifier, or an asset/field name in a helper.|数据集标识，或辅助函数中的模型资产/字段名称。
split|Dataset partition, e.g. dev or train; QASPER selection is explicit.|数据划分，如 dev/train；QASPER 的划分显式指定。
paper|One raw QASPER paper record, including its full text.|一条原始 QASPER 论文记录，含全文。
papers|The mapping of QASPER paper IDs to raw paper records.|QASPER 论文 ID 到原始论文记录的映射。
answer|One QASPER annotation to classify as boolean/extractive/etc.|一条 QASPER 答案标注，用于判断 boolean、抽取式等类型。
require_text_evidence|Whether all annotations need evidence found in full text; this filters eligibility only.|是否要求全部标注的证据可在全文找到；只用于筛选合格题目。
device|cpu or cuda; unavailable CUDA raises an error instead of silently switching.|cpu 或 cuda；CUDA 不可用就报错，不暗中切换。
query_block|Number of attention query rows computed together; all causal keys remain visible.|一次计算的注意力查询行数；仍能看见全部因果前文 key。
logit_block|Number of token positions whose vocabulary logits are computed together.|一次计算词表 logits 的 token 位置数。
input_ids|Token IDs consumed by the local transformer model.|送入本地 Transformer 的 token ID。
attention_mask|Mask describing valid tokens; DAC's adapter requires one unpadded sequence.|有效 token 掩码；DAC 适配要求单条无 padding 序列。
hidden_states|Token embeddings entering an attention layer.|进入注意力层的 token 隐状态向量。
attention|The actual model attention-layer object being adapted.|正在适配的模型注意力层对象。
position_embeddings|Rotary positional embeddings supplied by the transformer.|Transformer 提供的旋转位置编码。
past_key_values|Transformer KV cache argument; this full-source scorer does not use cached decoding.|Transformer KV 缓存参数；全文打分不使用缓存解码。
cache_position|Transformer cache-position argument retained for interface compatibility.|为接口兼容保留的 Transformer 缓存位置参数。
return_attn|Whether scoring also returns aggregated attention importance.|是否除打分外还返回聚合注意力重要性。
sents|Source sentences awaiting length checks.|待检查模型长度限制的原文句子。
limit|Maximum length in chunking, or the currency reservation ceiling in CostGuard.|分段时的最大长度；CostGuard 中则是费用预留上限。
count_tokens|Callable used to measure a string in a particular model's tokenizer.|按特定模型 tokenizer 计数的函数。
origin_text|Original text before LLMLingua's native chunking.|LLMLingua 原生分块之前的原文。
chunk_end_tokens|Upstream chunk-ending markers; kept from the official implementation.|上游实现指定的分块结束标记，沿用官方设置。
chunks|Received stream events to verify or assemble.|要核验或组装的流式响应事件。
response|An HTTP response stream or assembled model response.|HTTP 响应流，或已经组装好的模型响应。
body|Raw API response object; in page(), the report's HTML body.|原始 API 响应对象；page() 中则为报告 HTML 正文。
text|The string being parsed, counted, compressed or rendered.|当前要解析、计数、压缩或展示的字符串。
previous|Previous draft, only in historical helper code; unused by clean first-draft summary.|旧辅助函数中的上一稿；clean 第一稿摘要不使用它。
previous_tokens|Measured token count of that historical previous draft.|旧辅助函数中上一稿的实测 token 数。
guard|Shared cost-reservation object that approves each new paid request.|共享费用预留对象，在新增付费请求前检查限额。
model|Provider model identifier used to select pricing assumptions.|提供商的模型标识，用于选择费用估算参数。
index|Index of the reservation entry to settle in the cost ledger.|待结算费用账本记录的下标。
attempts|Transport attempts and usage records made for this completion.|本次生成发生的网络尝试及 usage 记录。
url|Pinned public dataset/tokenizer URL to download when permitted.|允许下载时使用的固定公开数据/tokenizer 网址。
expected_sha256|Expected file digest; a mismatch prevents using altered assets.|预期文件哈希；不一致时拒绝使用被改变的资产。
value|The JSON-compatible object to save or hash.|要保存或计算哈希的 JSON 兼容对象。
values|Numbers sorted to extract a bootstrap percentile interval.|排序后用于提取 bootstrap 百分位区间的数值。
title|Human-readable page title.|供人阅读的页面标题。
script|Local JavaScript enabling language switching and example selection.|启用语言切换和题目选择的本地 JavaScript。
kwargs|Additional upstream keyword arguments retained for compatibility.|为上游接口兼容接收的其他关键字参数。'''
PARAMS={parts[0]:parts[1:] for parts in (line.split('|') for line in PARAM_TEXT.splitlines())}
ACTIONS={
 'main':('Choose a stage, then call its function in order.','选择阶段，再按顺序调用对应函数。'),
 'run_lock':('Prevent two processes from writing the same run.','防止两个进程同时写同一轮实验。'),
 'load_config':('Validate the requested experiment before doing any work.','在执行前验证实验配置。'),
 'run_directory':('Map run_name to a dedicated result directory.','把 run_name 映射到独立结果目录。'),
 'code_hashes':('Record exact computational source and model-lock fingerprints.','记录计算源码与模型锁文件的准确哈希。'),
 'load_keys':('Load local credentials without printing them.','读取本机凭据，不打印它们。'),
 'package_versions':('Record installed dependency versions without loading GPU models.','记录依赖版本，不加载 GPU 模型。'),
 'prepare':('Load raw data and freeze the selected samples and settings.','加载原始数据并冻结选题与设置。'),
 'budget_for':('Compute an ex-ante budget from the original source.','从原文计算事先固定的预算。'),
 'compress':('Process source-only inputs, one local model at a time.','处理纯原文输入，每次仅加载一个本地模型。'),
 'condition_for':('Read one condition in a format shared by the reader.','把某个条件读成 reader 使用的统一格式。'),
 'predict':('Get one probability for each question and condition, reusing exact caches.','逐题逐条件得到一个概率，并复用准确请求缓存。'),
 'evaluate':('Recompute metrics from saved probabilities.','从保存概率重新计算指标。'),
 'audit':('Reconstruct real requests and validate results before publication.','重建真实请求，发布前验证结果。'),
 'check_api_record':('Match a saved response to the reconstructed complete request.','把保存响应与重建的完整请求匹配。'),
 'load_examples':('Select reproducible questions from raw eligible pools.','从原始合格样本池可复现地选题。'),
 'compression_path':('Build a cache path from source text and method.','由原文和方法构建缓存路径。'),
 'load_compression':('Read a compression only after checking its input and output hashes.','检查输入与输出哈希后读取压缩结果。'),
 'save_compression':('Save full output and its provenance atomically.','完整保存输出及来源，避免半写入。'),
 'create_engine':('Load only the selected local compressor.','仅加载被选中的本地压缩器。'),
 'local_candidates':('Tune native retention using measured length alone.','只按实测长度调整原生保留率。'),
 'select_local':('Keep the longest feasible complete local output; never trim it.','保留最长合规完整本地输出，绝不裁剪。'),
 'compress_one':('Expose the single source-only compression boundary.','提供唯一的纯原文压缩边界。'),
 'summary_messages':('Construct the exact original prompt without a question.','构造准确的原始提示词，不带问题。'),
 'output_ceiling':('Allow room for a complete generation beyond the target budget.','在目标预算之外留出空间以得到完整生成。'),
 'generate_summary':('Accept the first complete draft, even if over target.','接收第一份完整稿，即使超过目标。'),
 'request_identity':('Represent every generation parameter in the cache identity.','把所有生成参数纳入缓存身份。'),
 'make_client':('Choose cache-only or explicitly enabled paid execution.','选择仅缓存执行或显式允许的付费执行。'),
 'predict_one':('Pass context and question to the unchanged reader prompt, then parse p_yes.','将上下文与问题放入未变的 reader 提示词，再解析 p_yes。'),
 'analyze':('Calculate paired scores, the full directed matrix, and uncertainty.','计算配对评分、完整有向矩阵和不确定性。'),
 'import_firstdraft':('Verify and copy old evidence without generating any new model output.','验证并复制旧证据，不生成新模型输出。'),
 'record':('Add a read-only source fingerprint to the migration proof.','把只读来源哈希加入迁移证明。'),
 'copy_api':('Validate the exact historical request before copying its response.','复制响应前验证准确的历史请求。'),
 'build_report':('Publish only metrics whose saved audit hashes still match.','仅发布保存审计哈希仍一致的指标。'),
 'build_audit':('Embed actual texts, requests and responses in an offline viewer.','把实际文本、请求和响应嵌入离线审计页。'),
 'heatmap':('Use one shared color scale for both directional matrices.','为双向矩阵使用统一色阶。'),
 'page':('Assemble a standalone bilingual document.','组装独立双语页面。'),
 'infogap':('Sweep exact breakpoints to find the largest directed decision advantage.','扫描准确断点，找到最大的有向决策优势。'),
 'forecast_scores':('Compute accuracy, Brier and log loss from raw probabilities.','从原始概率计算准确率、Brier 和 log loss。'),
 'bootstrap_gaps':('Resample context clusters while keeping the two predictors paired.','按上下文成组重采样，并保持两预测器配对。'),
 'payoff_curve':('Keep both limits at each threshold rather than smoothing the jump.','保留每个阈值两侧极限，不平滑跳跃。'),
 'parse_probability':('Reject invalid probability JSON rather than substituting a default.','拒绝无效概率 JSON，不用默认值填补。')}


def bi(en,zh): return '<span class="en">'+en+'</span><span class="zh">'+zh+'</span>'
def esc(s): return html.escape(str(s))
def anchor(s): return ''.join(c if c.isalnum() else '-' for c in s)


def code_block(text,start=1):
    return '<pre><code>'+''.join(f'<span class="line"><i>{i}</i>{esc(line)}</span>' for i,line in enumerate(text.splitlines(),start))+'</code></pre>'


def params_table(node):
    rows=[]
    # Include nested helpers before showing a block that contains their first definition.
    funcs=[n for n in ast.walk(node) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    for f in funcs:
        args=[*f.args.posonlyargs,*f.args.args]
        defaults=[None]*(len(args)-len(f.args.defaults))+list(f.args.defaults)
        all_args=list(zip(args,defaults))+list(zip(f.args.kwonlyargs,f.args.kw_defaults))
        if f.args.vararg: all_args.append((f.args.vararg,None))
        if f.args.kwarg: all_args.append((f.args.kwarg,None))
        for a,default in all_args:
            if a.arg not in PARAMS: raise ValueError('Missing parameter explanation: '+a.arg)
            en,zh=PARAMS[a.arg]
            label=(f.name+'.' if f is not node else '')+a.arg
            value=ast.unparse(default) if default is not None else ('automatic / 自动' if a.arg=='self' else 'required / 必填')
            rows.append('<tr><td><code>'+esc(label)+'</code></td><td><code>'+esc(value)+'</code></td><td>'+bi(esc(en),esc(zh))+'</td></tr>')
    if not rows: return '<p>'+bi('No explicit input parameters. The function reads CLI settings or fixed project resources.','没有显式参数；函数读取命令行设置或固定项目资源。')+'</p>'
    return '<div class="scroll"><table><tr><th>'+bi('Parameter','参数')+'</th><th>'+bi('Default','默认值')+'</th><th>'+bi('Meaning','含义')+'</th></tr>'+''.join(rows)+'</table></div>'


def build():
    config=json.loads((ROOT/'configs/clean/mini100.json').read_text(encoding='utf-8'))
    body='<header id="top"><p class="eyebrow">BLACKWELL COMPACTION · CLEAN CODE</p><h1>'+bi('Read the experiment, one step at a time','一步一步读懂实验代码')+'</h1><p class="lead">'+bi('Start with the six-stage flow. Each box links to actual code, with parameter explanations before the first source block.','先看六步流程。点击框图跳到真实代码，每个源码块前都有参数解释。')+'</p><button onclick="switchLanguage(\'en\')">English</button> <button onclick="switchLanguage(\'zh\')">中文</button> <button onclick="expandAll()">'+bi('Expand source blocks','展开全部源码')+'</button></header>'
    flow=[('run.py','main','1 · Command','1 · 命令入口'),('experiment/config.py','load_config','2 · Settings','2 · 读取配置'),('experiment/pipeline.py','prepare','3 · Data','3 · 准备数据'),('experiment/pipeline.py','compress','4 · Compress','4 · 压缩'),('experiment/pipeline.py','predict','5 · Predict','5 · 预测'),('experiment/pipeline.py','evaluate','6 · Evaluate','6 · 评估')]
    body+='<nav class="flow">'+''.join('<a href="#'+anchor(p+'-'+fn)+'">'+bi(en,zh)+'</a>' for p,fn,en,zh in flow)+'</nav>'
    body+='<section><h2>'+bi('Keep these three objects separate','先分清三个对象')+'</h2><div class="cards">'
    for en,zh,example in [('context → compressor','context → 压缩器','"A full passage or paper..."'),('context + question → reader','context + question → reader','{"p_yes": 0.73}'),('probabilities + label → evaluator','概率 + label → 评估器','G(A, B), Brier, log loss')]:
        body+='<article><h3>'+bi(en,zh)+'</h3><code>'+esc(example)+'</code></article>'
    body+='</div><p>'+bi('The compressor only sees context and a budget. The question first appears at the reader. The label is used to select a declared dataset subset and to evaluate; it never enters a compressor or a reader prompt.','压缩器只看 context 和预算。question 到 reader 阶段才出现。label 用于定义声明的数据子集及评价，绝不进入压缩器或 reader 提示词。')+'</p><p>'+bi('A parameter is an input slot in a function definition. An argument is the actual value passed into it. A returned dict is a bundle of named outputs; a Path points to a file or folder.','参数是函数定义中的输入位置；调用时填进去的实际值叫实参。返回的 dict 是一组带名字的输出；Path 表示文件或文件夹位置。')+'</p></section>'
    body+='<section><h2>'+bi('Try the safe path first: CMD','先试不收费的路径：CMD')+'</h2><p>'+bi('Open CMD inside the project folder. These commands reuse verified local caches; they do not call the API or run local compression.','在项目目录打开 CMD。下列命令复用已验证缓存，不调用 API、不跑本地压缩。')+'</p>'+code_block('call .venv\\Scripts\\activate.bat\nset PYTHONHASHSEED=0\npython run.py --config configs/clean/mini100.json --stage all --offline')+'<p>'+bi('A fresh paid run uses a copy of the config with a new run_name, then --stage all --allow-api. Without --allow-api, paid cache misses fail. --offline additionally forbids downloads and local inference.','从头付费跑时，复制配置并改用新的 run_name，再加 --stage all --allow-api。没有 --allow-api，付费缓存缺失即报错；--offline 还禁止下载和本地推理。')+'</p></section>'
    body+='<section><h2>'+bi('The configuration you edit','你日常修改的配置')+'</h2>'+code_block(json.dumps(config,ensure_ascii=False,indent=2))+'<p>'+bi('run_name selects a new result folder. sample_size is per dataset. retained_ratio means KEEP. reader and summary are different roles. summary_policy fixes first-draft behavior; local budgets remain upper caps. Frozen settings cannot be edited in an existing run.','run_name 选择新的结果文件夹；sample_size 是每个数据集题数；retained_ratio 表示保留。reader 和 summary 是不同角色。summary_policy 固定第一稿策略，本地方法仍用上限。已有运行不允许修改冻结设置。')+'</p></section>'
    body+='<section><h2>'+bi('A practical reading order','建议阅读顺序')+'</h2><ol>'
    for en,zh in [('Read main, prepare and load_examples; explain where the 200 questions come from.','先读 main、prepare、load_examples，能说清 200 题从哪来。'),('Read compress_one and summary_messages; identify where query-blindness is enforced.','读 compress_one 和 summary_messages，找出不传 query 的边界。'),('Read predict_one and parse_probability; distinguish text generation from a validated probability.','读 predict_one 和 parse_probability，区分模型文本与通过校验的概率。'),('Read analyze and infogap; connect each matrix direction to paired lists of numbers.','读 analyze 和 infogap，把矩阵方向对应到两组配对数值。'),('Read audit last, then inspect API caching and GPU adapters only when needed.','最后读 audit；需要时再深入 API 缓存与 GPU 适配。')]:body+='<li>'+bi(en,zh)+'</li>'
    body+='</ol></section><nav class="toc"><h2>'+bi('Files and source blocks','文件与源码块')+'</h2>'+''.join('<a href="#'+anchor(p)+'">'+esc(p)+'</a>' for p,*_ in FILES)+'</nav>'
    blocks=[];source_hashes={};function_count=0
    for path,en,zh,desc_en,desc_zh in FILES:
        text=(ROOT/path).read_text(encoding='utf-8-sig');lines=text.splitlines();tree=ast.parse(text)
        source_hashes[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        body+='<section class="file" id="'+anchor(path)+'"><p class="eyebrow">'+esc(path)+'</p><h2>'+bi(en,zh)+'</h2><p>'+bi(desc_en,desc_zh)+'</p>'
        nodes=[]
        for n in tree.body:
            if isinstance(n,ast.FunctionDef): nodes.append((n,n.name))
            elif isinstance(n,ast.ClassDef):
                nodes.extend((child,n.name+'.'+child.name) for child in n.body if isinstance(child,ast.FunctionDef))
        first=min((n.lineno for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))),default=len(lines)+1)
        if first>1:
            body+='<details><summary>'+bi('Imports and constants','导入与常量')+'</summary>'+code_block('\n'.join(lines[:first-1]))+'</details>'
        for node,qualified in nodes:
            function_count+=len([n for n in ast.walk(node) if isinstance(n,ast.FunctionDef)])
            aid=anchor(path+'-'+node.name)
            # __init__/compress names may repeat across classes; anchor includes qualified name if needed.
            if aid in blocks: aid=anchor(path+'-'+qualified)
            blocks.append(aid)
            action=ACTIONS.get(node.name,(desc_en,ast.get_docstring(node) or desc_zh))
            body+='<article class="block" id="'+aid+'"><h3><code>'+esc(qualified)+'</code></h3><p>'+bi(esc(action[0]),esc(action[1]))+'</p>'+params_table(node)
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            body+='<details class="source"><summary>'+bi('Read actual source · lines ','阅读真实源码 · 行 ')+str(start)+'–'+str(node.end_lineno)+'</summary>'+code_block('\n'.join(lines[start-1:node.end_lineno]),start)+'</details><div class="steps"><button onclick="moveBlock(this,-1)">← '+bi('Previous','上一块')+'</button><a href="#top">'+bi('Overview','总览')+'</a><button onclick="moveBlock(this,1)">'+bi('Next','下一块')+' →</button></div></article>'
        body+='</section>'
    body+='<section><h2>'+bi('Check your understanding','读完检查自己是否理解')+'</h2>'
    for q_en,q_zh,a_en,a_zh in [
      ('Why is question absent from compress_one?','为什么 compress_one 没有 question 参数？','It makes evaluation-query leakage structurally difficult; the compressor receives only source text and length settings.','通过接口隔离评测问题；压缩器只收到原文和长度设置。'),
      ('Does 50% guarantee an equally long summary?','50% 是否保证摘要等长？','No. It is a requested summary target; actual output may be under or over it. Local methods enforce a cap.','不保证。摘要只把它作为提示词目标，实际可偏短或偏长；本地方法执行上限。'),
      ('Does --offline rerun a model?','--offline 是否重新运行模型？','No. It replays cached evidence and recomputes mathematical results; missing caches are errors.','不会。它重放缓存证据并重算数学结果；缺缓存就报错。'),
      ('Why are G(A,B) and G(B,A) both saved?','为什么要同时保存 G(A,B) 和 G(B,A)？','Different decision thresholds may favor different predictors; one positive direction alone is not Blackwell dominance.','不同决策阈值可能偏好不同预测器；单个方向为正不等于 Blackwell 支配。')]:
        body+='<details><summary>'+bi(q_en,q_zh)+'</summary><p>'+bi(a_en,a_zh)+'</p></details>'
    body+='</section><footer>'+bi('Generated from actual saved source. Historical tools remain frozen for provenance; the current run path imports none of those runners.','由真实保存源码生成。历史 tools 为追溯而冻结保留；当前运行路径不导入旧运行脚本。')+f'<p>{len(FILES)} files · {function_count} functions including nested helpers · {len(blocks)} source blocks</p></footer>'
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:20px}body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#20384b;background:#f2f5f8;max-width:1200px;margin:auto;padding:30px}h1{font-size:38px;line-height:1.2}h2{font-size:26px}h3{font-size:20px}header,section,.toc{background:white;border:1px solid #dbe4eb;border-radius:16px;padding:28px;margin:24px 0}.lead{font-size:20px;color:#506c80}.eyebrow{font:13px ui-monospace,Consolas,monospace;color:#59778d;overflow-wrap:anywhere}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.flow a{padding:22px 10px;text-align:center;background:#214d70;color:white;border-radius:10px;text-decoration:none}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cards article{padding:18px;background:#eff5fa;border-radius:10px}.toc{display:flex;flex-wrap:wrap;gap:12px}.toc h2{width:100%}.toc a{padding:5px 10px;background:#eef4f9;border-radius:6px;font-size:14px}.block{border-top:1px solid #dbe4eb;margin-top:28px;padding-top:20px}a{color:#1b668e}button{font:inherit;border:1px solid #b9cedb;border-radius:7px;padding:7px 13px;background:white;color:#235678;cursor:pointer}summary{cursor:pointer;color:#235678;padding:12px 0;font-weight:600}pre{font:14px/1.65 Consolas,ui-monospace,monospace;overflow:auto;background:#102638;color:#e5f0f7;border-radius:10px;padding:16px;tab-size:4}pre .line{display:block;min-height:1.65em;white-space:pre}pre i{display:inline-block;width:42px;text-align:right;margin-right:18px;user-select:none;color:#83a0b6;font-style:normal}code{font-family:Consolas,ui-monospace,monospace}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dfe8ef;vertical-align:top}th{background:#eef4f9}.scroll{overflow:auto}td:first-child{min-width:160px}.steps{display:flex;align-items:center;justify-content:space-between;padding-top:14px}.zh{display:none}html[data-lang=zh] .en{display:none}html[data-lang=zh] .zh{display:inline}footer{color:#526b7c;margin:36px 0}.file:target,.block:target{outline:3px solid #71a9ce;outline-offset:7px}@media(max-width:800px){body{padding:14px}.flow{grid-template-columns:repeat(2,1fr)}.cards{grid-template-columns:1fr}section,header,.toc{padding:18px}h1{font-size:30px}}'''
    js='''function switchLanguage(l){document.documentElement.dataset.lang=l;document.documentElement.lang=l;}function expandAll(){document.querySelectorAll('details.source').forEach(d=>d.open=true);}function reveal(){const id=decodeURIComponent(location.hash.slice(1));const el=document.getElementById(id);if(el){const d=el.querySelector('details.source');if(d)d.open=true;}}window.addEventListener('hashchange',reveal);function moveBlock(button,direction){const blocks=[...document.querySelectorAll('.block')];const current=button.closest('.block');const next=blocks[blocks.indexOf(current)+direction];if(next)location.hash=next.id;}reveal();'''
    OUT.write_text('<!doctype html><html lang="en" data-lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Clean experiment code walkthrough</title><style>'+css+'</style>'+body+'<script>'+js+'</script></html>',encoding='utf-8')
    (ROOT/'docs/Clean_Code_Source_Index.json').write_text(json.dumps({'files_sha256':source_hashes,'function_count':function_count,'blocks':blocks},indent=2),encoding='utf-8')
    print('Built guide:',OUT,'functions',function_count,'blocks',len(blocks))


if __name__=='__main__': build()
