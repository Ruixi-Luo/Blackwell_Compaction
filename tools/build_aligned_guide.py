"""把新实验的真实源码和参数说明嵌入可离线阅读的双语导读。"""
import ast
import hashlib
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from blackwell.reporting import STYLE,bi,table

PARAMS = {
 'title': ('The human-readable page title, escaped before insertion.', '页面标题，插入 HTML 前会转义。'),
 'body': ('Already assembled report HTML from the other presentation functions.', '由其他展示函数组装好的报告 HTML。'),
 'metrics': ('Saved scores, exact InfoGap values, bootstrap ranges and payoff curves.', '已保存的评分、精确 InfoGap、bootstrap 区间及收益曲线。'),
 'scale': ('Largest displayed InfoGap, shared across datasets for heat-map colors.', '当前展示数据集中的最大 InfoGap，用于统一热力图色阶。'),
 'data': ('One method’s probability-diagnostic dictionary, including reliability bins.', '一个方法的概率诊断字典，含可靠性分箱。'),
 'name': ('Human-readable method label used in a plot.', '图中的可读方法名称。'),
 'diagnostics': ('Per-method probability and actual-length diagnostics.', '每种方法的概率诊断及实际长度诊断。'),
 'usage': ('Attempt counts and returned token usage, separated by API stage.', '按 API 阶段区分的尝试次数与返回 token 用量。'),
 'methods': ('Ordered method identifiers used to retrieve paired conditions.', '按顺序排列的方法标识，用于读取同一题的配对条件。'),
 'path': ('Repository-relative source filename displayed with each code block.', '相对仓库的源码文件名，显示在各代码块旁。'),
 'node': ('Python syntax-tree node identifying a function and its source lines.', 'Python 语法树中的函数节点，记录函数参数和源码行号。'),
 'lines': ('The source file split into a list of original lines.', '源码按原始行拆开的列表。'),
 'index': ('A unique function-card number used to create an HTML anchor.', '函数卡片的唯一编号，用来生成 HTML 跳转锚点。'),
 'args': ('Collects extra positional arguments; this test double deliberately ignores them.', '收集额外的位置参数；这个测试替身会故意忽略它们。'),
 'kwargs': ('Collects extra named arguments into a dictionary; unused by the failure test double.', '把额外的命名参数收集成字典；失败测试替身不使用它们。'),

 'compacted':('Dictionary of method names to context-only compression records, before the common-length step.','方法名到压缩记录的字典；只有上下文及长度，尚未统一最终长度。'),
 'contexts':('Deduplicated context strings assigned to this worker; no queries or labels.','分配给此 worker 的去重上下文字符串，不含问题和标签。'),
 'directory':('This worker’s private API cache and ledger directory.','当前 worker 独立的 API 缓存与账本目录。'),
 'workers':('Number of independent rewrite workers; their attempt caps sum to the fixed total.','独立改写 worker 数；各自上限之和等于固定总上限。'),
 'worker':('Zero-based worker index for deterministic context assignment and quota splitting.','从 0 起的 worker 编号，用于确定性分配原文及额度。'),
 'stop':('Shared stop signal; after a failure, other workers stop before their next context.','共享停止信号；发生错误后，其他 worker 在下个上下文前停止。'),
 'messages':('The context-only rewrite messages, including grammar and length instructions.','仅含上下文的改写消息，包括语法和长度指令。'),
 'max_tokens':('API output ceiling, also used to conservatively pace requests.','API 输出上限，同时用于保守估计节流额度。'),
 'self':('The current object; Python supplies it automatically.','当前对象，Python 自动传入。'),
 'context':('Only the original context string; not an example, query or answer.','仅原始上下文字符串，不是包含问题答案的整个样本。'),
 'budget':('Maximum number of Qwen tokens allowed in the compressed context.','压缩上下文允许的 Qwen token 数上限。'),
 'tokenizer':('The frozen Qwen tokenizer used to count and clamp text consistently.','固定 Qwen tokenizer，统一计算长度及执行裁剪。'),
 'engine':('A loaded local compressor with compress(context, retained_ratio).','已加载的本地压缩器，提供 compress(context, retained_ratio)。'),
 'max_trials':('Hard limit on length-only candidate attempts; not answer-quality trials.','长度试探的最大次数，不是根据答题表现试选次数。'),
 'candidates':('Draft texts, measured lengths and native rates or API request hashes.','候选文本、实测长度及原生比例或 API 请求哈希。'),
 'client':('The existing cached API client, with its attempt ledger and hard cap.','已有的缓存 API 客户端，携带尝试账本和硬性上限。'),
 'method':('The compressor name selecting a local algorithm or rewrite prompt.','方法名，选择本地算法或改写提示词。'),
 'previous':('The preceding draft, used only to repair its length.','上一版草稿，仅用于修订长度。'),
 'previous_tokens':('Actual Qwen token count of that draft.','上一版草稿的实际 Qwen token 数。'),
 'probabilities':('Original p_yes values, in exactly the same order as labels.','原始 p_yes 列表，与标签顺序一一对应。'),
 'labels':('Binary gold labels; these enter evaluation, never compression.','二元真实标签；用于评估，从不进入压缩。'),
 'bins':('Number of fixed equal-width probability bins; ten by default.','固定等宽概率箱数量，默认十个。'),
 'conditions':('All per-question records for one condition, including text lengths.','某一条件的全部逐题记录，包含文本长度。'),
 'condition':('One method’s recorded text, probability and request metadata.','一个方法保存的文本、概率和请求元数据。'),
 'config':('The JSON configuration: model, dataset, methods, rate and API caps.','JSON 配置：模型、数据集、方法、保留率及 API 上限。'),
 'run':('The new run directory under runs/; old run directories are read-only.','runs/ 下的新实验目录；旧运行目录只读。'),
 'run_dir':('The directory for this run’s caches, predictions and reports.','当前实验的缓存、预测和报告目录。'),
 'request':('Compression cache identity: context, budget, method and protocol hash.','压缩缓存身份：原文、预算、方法及协议哈希。'),
 'manifest':('The frozen identity: config, source sample hashes, code and runtime.','冻结身份记录：配置、来源样本哈希、代码及运行环境。'),
 'examples':('The frozen paired sample list; only context is passed to compression.','冻结的配对样本列表；压缩阶段只取 context。'),
 'rows':('Completed paired predictions, one row per question.','完成的配对预测，每题一行。'),
 'result':('A compression result including selected text and length diagnostics.','压缩结果，含选定文本和长度诊断。'),
 'seconds':('Measured time for all attempts for one unique context.','单份独立上下文的全部尝试耗时。'),
 'api':('False selects local GPU compressors; True selects API rewrites.','False 运行本地 GPU 压缩器；True 运行 API 改写。'),
 'offline':('True forbids new API requests and requires complete response caches.','True 禁止新增 API 请求，必须命中完整缓存。'),
 'message':('The explanation raised when a required invariant fails.','必要条件不成立时抛出的解释。'),
 'retained_ratio':('Requested native fraction to keep; actual Qwen length is measured separately.','请求保留的原生比例；实际 Qwen 长度另行测量。'),
 'local_cache':('The original runner’s local compression cache.','原运行器的本地压缩缓存。'),
 'text':('The input or compressed text, depending on this function.','当前函数处理的输入文本或压缩文本。'),
 'hashes':('IDs of the compression API requests that produced drafts.','产生候选的压缩 API 请求标识。'),
 'repaired':('Whether an additional draft was requested to repair length.','是否额外请求草稿来修订长度。'),
}

EXPLAIN = {
 'section_title':('Select the English or Chinese presentation of a static section label.','为静态章节名称生成可切换的英文和中文展示。'),
 'page': ('Wrap prepared content with the shared style and English/Chinese controls. No API calls occur here.', '给已组装内容加上公共样式和中英文按钮；此处不会调用 API。'),
 'matrix': ('Render the saved directional values as an 8×8 heat map with a common color scale.', '把已保存的双向数值画成使用共同色阶的 8×8 热力矩阵。'),
 'reliability_plot': ('Draw nonempty probability bins against their observed Yes frequencies.', '把非空概率箱的平均预测值与实际 Yes 频率画出来。'),
 'protocol_section': ('State the frozen experiment and its interpretation limits in both languages.', '用两种语言说明固定实验设置及解释边界。'),
 'method_panels': ('Assemble scores, matrices, length diagnostics, probability plots, uncertainty and usage from saved results.', '从已保存结果组装评分、矩阵、长度诊断、概率图、波动区间与用量。'),
 'sample_section': ('Create expandable question records with their probabilities and escaped context texts.', '生成可展开的逐题记录，显示概率及经过转义的上下文文本。'),
 'report_main': ('Publish only completed, audited runs; add pre-match length statistics and copy shareable outputs.', '只发布完整且通过审计的运行；补充等长处理前的长度统计，并复制可分享结果。'),
 'function_card': ('Explain every parameter, then embed genuine source in short blocks with original line numbers and content hashes.', '先说明每个参数，再按短代码块嵌入真实源码，保留原行号和内容指纹。'),
 'guide_main': ('Read source files without executing them, parse function boundaries, and build this offline walkthrough.', '读取但不执行源码，解析函数边界，生成这份离线导读。'),
 'count': ('A word-count stand-in used only by tests; production uses the frozen Qwen tokenizer.', '仅供测试使用的单词计数替身；正式实验使用固定的 Qwen tokenizer。'),
 'head_tail': ('A simple prefix-trimming stand-in for length tests, not the production tokenizer implementation.', '长度测试用的简单前缀截断替身，不是正式 tokenizer 的实现。'),
 'test_actual_lengths_are_equal_without_changing_controls': ('Check exact equal lengths, unchanged controls and preserved original records.', '检查严格等长、对照组不变，以及原记录未被原地修改。'),
 'test_empty_compression_is_flagged_not_silently_dropped': ('Check that an empty output produces an explicit zero common budget rather than a silently removed sample.', '检查空输出会产生明确的零共同预算，而非悄悄删除样本。'),
 'test_parallel_worker_caps_sum_to_global_cap': ('Check that splitting an uneven allowance across workers preserves the exact total cap.', '检查不能整除的额度拆给各 worker 后，总上限仍完全不变。'),
 'test_rewrite_pacing_preserves_the_request': ('Mock the clock and API client to verify pacing arithmetic and unchanged request messages.', '替换时钟与 API 客户端，核验节流计算及请求消息不被改写。'),
 'test_cached_rewrites_need_no_pacing_or_new_api_call': ('Check that cached responses require neither sleeping nor a fresh API request.', '检查缓存命中既不等待，也不产生新的 API 请求。'),
 'test_compression_model_does_not_change_reader': ('Check that a copied rewrite configuration changes the compressor without mutating the reader.', '检查改写配置通过副本改变压缩模型，而不修改 reader。'),
 'test_feasible_candidate_and_explicit_clipping': ('Check length-based candidate selection and explicit cap trimming.', '检查按长度选择候选，以及显式执行上限裁剪。'),
 'test_native_rate_is_length_only': ('Use a synthetic compressor to check bounded native-rate search without answer feedback.', '用模拟压缩器检查有限的原生比例搜索，不提供答题反馈。'),
 'test_tiny_overshoot_preferred_to_large_underfill': ('Check that a nearly feasible long draft is selected over a substantially short draft, then explicitly capped.', '检查轻微超长稿优先于大幅欠填稿，并明确裁剪到上限。'),
 'test_query_boundary': ('Inspect public signatures and sentinel messages to check the query/label boundary.', '检查公共函数签名与哨兵消息，核验 query 和标签的隔离边界。'),
 'test_rewrite_length_repairs_and_cache_hashes': ('Exercise the optional repair path with synthetic drafts and verify every recorded request hash. Formal runs keep this option disabled.', '用模拟草稿测试可选的长度修订路径，并核验全部请求指纹；正式运行关闭这个选项。'),
 'test_api_failure_does_not_silently_skip_context': ('Check that an API exception propagates instead of producing a substitute or skipped context.', '检查 API 异常会向外抛出，不产生替代文本或跳过上下文。'),
 'test_bins_include_one_and_keep_empty_bins': ('Check endpoint binning, empty bins, confident errors and log-loss clipping sensitivity.', '检查端点分箱、空箱、极端错判及 log loss 对截断值的敏感性。'),

 'match_lengths':('Choose the shortest feasible compressed output for each context, then deterministically trim the other compressed conditions until all have exactly the same Qwen token count. Full and no-context controls stay unchanged. Local algorithms are not rerun at the smaller common budget; the measured system is the compressor plus this trimming adapter. The resulting common budget is at most 50% of the original and may be lower; no question or label enters this step.','针对每份上下文取最短可行压缩长度，确定性裁剪其他压缩条件，直到 Qwen token 数完全相同。原文与空上下文对照不变。不会在更低的共同预算下重跑本地算法，测量对象是压缩器与裁剪适配器的组合。共同预算最多为原文 50%，可能更低；此步骤不接收问题或标签。'),
 'rewrite_worker':('Each worker processes a deterministic shard sequentially and owns a separate API cache and ledger. Only the standard-library thread pool is shared. Attempt caps sum to the configured total; API failures stop other workers before the next context.','每个 worker 顺序处理确定分片，拥有独立 API 缓存和账本，仅使用标准库线程池。各尝试上限之和等于配置总额；API 失败会通知其他 worker 在下一份上下文前停止。'),
 'complete':('Estimate input plus maximum output tokens and space this worker’s calls conservatively. Then delegate the unchanged messages to the original serial cached client. Worker parallelism does not change the compressor prompt or reader.','估计输入与最大输出 token，并保守安排当前 worker 的调用间隔，然后把原样消息交给已有串行缓存客户端。并行不改变压缩提示词或 reader。'),
 'compression_config':('Create a separate configuration for the rewrite model and its ledger. Both new compressors use DeepSeek-V3.2; the reader remains Qwen. This is a copy, so changing the compressor does not silently change the reader.','为改写模型及其账本创建单独配置。两个新增压缩器均用 DeepSeek-V3.2，reader 保持 Qwen。这是配置副本，不会因为更换压缩模型而悄悄更换 reader。'),
 'select_candidate':('Length is the only selection criterion. Prefer a draft within the target band, otherwise choose the closest length. If the selected draft is too long, use the same deterministic head–tail cap and expose post_clip_used. A short but feasible output is not padded.','只根据长度选择：优先目标区间内候选，否则选择距离预算最近者。若选中草稿超长，使用统一确定性首尾裁剪并记录 post_clip_used。短但合规的文本不补无意义内容。'),
 'align_local':('Translate a Qwen-token cap into a native ratio through bounded search. Every trial sees the same original context. The search is heuristic because lengths need not be monotone; the saved candidates make this auditable.','通过有限搜索把 Qwen token 上限映射为原生保留率。每次尝试都看同一原文。长度不保证单调，所以搜索是启发式；保存所有候选便于审计。'),
 'rewrite_messages':('Construct a query-free prompt. Ordinary summary and Telegrapher share the budget instruction; TE additionally loads the exact versioned grammar. Measured length and a word-count estimate guide later drafts, without answer feedback.','构造不含 query 的提示词。普通摘要与 Telegrapher 共用预算指令；后者额外加载固定版本语法。实测长度与词数估计指导修订，不使用答题反馈。'),
 'compress':('Produce candidates from the same original text, then choose only by measured length. API failures propagate; no empty substitute, silent omission or p=0.5 fallback.','从同一原文生成候选，再仅按实测长度选择。API 失败会向外抛出，不用空文本替代、不漏段、不补 p=0.5。'),
 'probability_diagnostics':('Compare forecast probabilities with binary outcomes in fixed bins. ECE is descriptive; it does not recalibrate the model. Endpoint-error and log-loss sensitivity counts reveal where confident mistakes dominate a score.','在固定箱内对照预测概率与二元结果。ECE 只作描述，不重新校准模型。端点错判与 log loss 敏感性揭示极端自信错误是否主导分数。'),
 'retention_diagnostics':('Summarize the distribution of realized retention and count underfilled or clamped cases. A common cap is not exact equal-length output.','汇总实际保留率分布，并统计欠填和裁剪。共同上限不等于输出严格等长。'),
 'prepare':('Read the old sample order, independently reload the pinned dataset, and record old file hashes. A new directory freezes code/config/runtime. Changing any of them requires a new run name.','读取旧样本顺序，独立重载固定数据，并记录旧文件哈希。新目录冻结代码、配置和环境；改变其中任何一项需换运行名。'),
 'compression_request':('The cache identity deliberately excludes question, label, supporting facts and reader outputs. Equal contexts share a result even when asked different questions.','缓存身份有意排除问题、标签、证据标注及 reader 输出。同一上下文面对不同问题时复用同一压缩结果。'),
 'read_compaction':('Verify both the request and the stored result digest before accepting a cache hit. Missing cache is explicit, never an alternative compressor.','命中缓存前校验请求及结果指纹。缓存缺失明确返回，不擅自换压缩方法。'),
 'save_compaction':('Save every candidate and its final selection atomically. Timing counts the complete length-search procedure.','原子保存全部候选及最终选择；耗时统计完整长度搜索过程。'),
 'compress_stage':('Deduplicate identical contexts and process one method at a time to bound GPU memory. Local inference blocks network; API rewrites use a separate ledger. No evaluation question is read here.','去重相同上下文，每次只装一个方法以控制显存。本地推理禁网；API 改写使用独立账本。此处不读取评测问题。'),
 'condition_text':('During prediction, read the already chosen compression from cache. Full, empty-context and head–tail are deterministic controls. No compression model is loaded here.','预测时读取已选定的压缩缓存。原文、空上下文及首尾截断是确定性对照。此处不加载压缩模型。'),
 'predict':('Only now attach the evaluation question. All methods receive the same question and reader settings. Strictly parse p_yes and persist paired completed rows; cached responses make interruption resumable.','到这里才加入评测问题。所有方法使用相同问题和 reader 设置。严格解析 p_yes 并保存完整配对行；响应缓存支持中断续跑。'),
 'finish':('Reuse the existing exact InfoGap and bootstrap implementation, then add descriptive probability and budget diagnostics. No forecast is modified by these diagnostics.','复用已有精确 InfoGap 及 bootstrap，再追加概率和预算诊断。诊断不修改任何预测概率。'),
 'audit':('Reconstruct complete API requests, check each saved candidate’s token count and length-only selection, verify old files stayed unchanged, independently check all 64 InfoGap cells, and replay forecasts with network blocked.','重建完整 API 请求，核对候选长度和纯长度选择，确认旧文件不变，独立核验全部 64 个 InfoGap 单元格，并在断网下重放预测。'),
 'require':('A failed invariant stops the run with a readable error instead of silently weakening the protocol.','必要条件失败就停止并给出可读错误，不悄悄放宽协议。'),
 'main':('The command line chooses one stage or the full sequence. Preparation and local inference are network-blocked; only rewrite and reader stages use API access.','命令行选择单一阶段或完整顺序。准备与本地推理禁止网络，只有改写及 reader 阶段使用 API。'),
 '__init__':('Store the dependencies and settings on the object; actual requests happen later through compress or complete.','把依赖和设置保存在对象上；之后调用 compress 或 complete 才执行推理或请求。'),
 '_result':('Package actual lengths and request metadata for the original runner.','为原运行器打包实际长度和请求元数据。'),
 'selected_methods':('Validate the method list, including the two new API compressors. The new aligned runner additionally freezes its eight paired conditions.','验证方法列表，包括新增两种 API 压缩器。预算对齐运行器还会冻结八组配对条件。'),
}


def section_title(title):
    english, _, chinese = title.partition(' / ')
    return bi(html.escape(english), html.escape(chinese or english))


def function_card(path, node, lines, index):
    args = [*node.args.posonlyargs,*node.args.args]
    if node.args.vararg:
        args.append(node.args.vararg)
    args.extend(node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(node.args.kwarg)
    parameters = []
    for arg in args:
        meaning = PARAMS.get(arg.arg)
        if node.name == 'require' and arg.arg == 'condition':
            meaning = ('The Boolean invariant that must be true to continue.', '必须为真的布尔条件；否则停止运行。')
        if node.name == 'method_panels' and arg.arg == 'rows':
            meaning = ('Optional paired records for probability scatterplots; omitting them skips those plots.','可选的配对记录，用来画概率散点图；省略时只跳过这组图。')
        if node.name == 'section_title' and arg.arg == 'title':
            meaning = ('A static English / Chinese label from the module list.','模块列表中的静态“英文 / 中文”名称。')
        if meaning is None:
            raise ValueError(f'Missing parameter explanation: {path}:{node.name}:{arg.arg}')
        prefix = '*' if arg is node.args.vararg else '**' if arg is node.args.kwarg else ''
        parameters.append([f'<code>{prefix}{arg.arg}</code>',bi(*meaning)])
    key = node.name
    if node.name == 'main' and path == 'tools/build_aligned_docs.py':
        key = 'report_main'
    if node.name == 'main' and path == 'tools/build_aligned_guide.py':
        key = 'guide_main'
    meaning = EXPLAIN.get(key)
    if path.startswith('tests/') and node.name in {'compress','__init__','complete'}:
        meaning = ('A test double that records calls, returns synthetic drafts or raises deliberately; it makes no real API request.','测试替身：记录调用、返回模拟草稿或故意抛错，不会产生真实 API 请求。')
    if meaning is None:
        raise ValueError(f'Missing function explanation: {node.name}')
    card = f'<article id="function-{index}"><h3><code>{node.name}</code></h3><p>'+bi(*meaning)+'</p>'
    if parameters:
        card += table([bi('Parameter','参数'),bi('Purpose','含义及作用')],parameters)
    for start in range(node.lineno-1,node.end_lineno,18):
        stop = min(start+18,node.end_lineno)
        source = '\n'.join(lines[start:stop])
        digest = hashlib.sha256(source.encode()).hexdigest()
        code = '\n'.join(f'<span class="line-number">{i+1:>3}</span> '+html.escape(lines[i]) for i in range(start,stop))
        card += f'<p class="muted">{path} · {start+1}–{stop}</p><pre data-code-sha256="{digest}"><code>{code}</code></pre>'
    return card+'</article>'


def main():
    modules = [
        ('prepare','1 · Freeze samples / 冻结样本','tools/run_aligned_suite.py',['prepare','compression_request','compression_config']),
        ('budget','2 · Native budget search / 原生预算搜索','blackwell/budgeting.py',['select_candidate','align_local']),
        ('rewrite','3 · Summary & Telegrapher / API 改写','blackwell/compressors/llm_rewrite.py',None),
        ('pacing','3b · Bounded concurrency / 有界并行','blackwell/rewrite_client.py',None),
        ('cache','4 · Compress & cache / 压缩与缓存','tools/run_aligned_suite.py',['read_compaction','save_compaction','compress_stage','rewrite_worker']),
        ('match','5 · Match actual lengths / 统一实际长度','blackwell/budgeting.py',['match_lengths']),
        ('reader','6 · Predict / 预测','tools/run_aligned_suite.py',['condition_text','predict']),
        ('diagnose','7 · Diagnose / 诊断','blackwell/diagnostics.py',None),
        ('audit','8 · Evaluate & audit / 评估与审计','tools/run_aligned_suite.py',['finish','audit','require','main']),
        ('reports','9 · HTML reports / HTML 报告','tools/build_aligned_docs.py',None),
        ('interface','Support · Shared interface / 共用接口','blackwell/compression.py',None),
        ('registry','Support · Method registry / 方法注册','run_experiment.py',['selected_methods']),
        ('tests','Appendix · Tests / 测试','tests/test_aligned.py',None),
        ('guide_builder','Appendix · Guide generation / 导读生成','tools/build_aligned_guide.py',None),
    ]
    body = '<header><div class="toolbar"><button onclick="language(\'en\')">English</button><button onclick="language(\'zh\')">中文</button></div><h1>'+bi('Read the aligned-budget experiment','逐块阅读预算对齐实验')+'</h1><p>'+bi('Click a stage to read the actual source code and every parameter. This is the current extension; the previous full-repository tutorial remains available below.','点击流程节点，阅读真实源码与每个参数。此页讲解当前扩展；下方保留上一版全仓库导读。')+'</p></header>'
    body += '<section id="overview"><h2>'+bi('Execution map','运行总览')+'</h2><div class="flow">'+''.join(f'<a class="node" href="#{key}">{section_title(title)}</a>' for key,title,_,_ in modules)+'</div><p>'+bi('context → compression → saved text; saved text + question → reader → p_yes; p_yes + gold label → InfoGap and diagnostics. Labels never enter compression or the reader prompt.','context → 压缩 → 保存文本；保存文本＋question → reader → p_yes；p_yes＋真实标签 → InfoGap 与诊断。标签不会进入压缩或 reader 提示词。')+'</p><p><a href="Code_Walkthrough_20261023.html">'+bi('Earlier full-repository guide','旧版全仓库代码导读')+'</a> · <a href="Experiments_20261024_Aligned/Experiment_Overview.html">'+bi('New experiment results','新实验结果')+'</a></p></section>'
    body += '<section><h2>'+bi('How to run and resume','运行与续跑')+'</h2><pre>$env:PYTHONHASHSEED="0"\n.\\.venv\\Scripts\\python.exe tools/run_aligned_suite.py --config configs/aligned_hotpotqa.json --stage all</pre><p>'+bi('Stages: prepare, local, rewrite, predict, audit, all. Rerunning the same command uses validated caches. Editing the protocol requires a new run_name; do not overwrite old results. Model weights and caches remain on D:.','阶段包括 prepare、local、rewrite、predict、audit、all。相同命令续跑会复用校验后的缓存；更改协议必须使用新 run_name，不能覆盖旧结果。模型和缓存保留在 D 盘。')+'</p><details><summary>configs/aligned_hotpotqa.json</summary><pre>'+html.escape((ROOT/'configs/aligned_hotpotqa.json').read_text(encoding='utf-8-sig'))+'</pre></details><p>'+bi('sample_size freezes the existing sample count; retained_ratio sets the 0.5 cap; max_api_attempts limits reader attempts; compression_max_api_attempts separately limits rewrite attempts; min_interval_seconds controls pacing, not sample size. rewrite_max_trials=1 fixes one draft per API compressor in this experiment; the class supports optional repairs for separate future runs. Failed and timed-out calls also count.','sample_size 冻结既有样本数；retained_ratio 设置 0.5 上限；max_api_attempts 限制 reader 尝试；compression_max_api_attempts 独立限制改写尝试；min_interval_seconds 控制请求间隔而非样本数；rewrite_max_trials=1 将本轮每种 API 压缩器固定为一稿，类中可选修订仅供另行设计的运行使用。失败及超时也计入尝试。')+'</p></section>'
    index = 0
    for key,title,relative,names in modules:
        lines = (ROOT/relative).read_text(encoding='utf-8').splitlines()
        tree = ast.parse('\n'.join(lines))
        nodes = sorted([n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and (names is None or n.name in names)],key=lambda n:n.lineno)
        body += f'<section id="{key}"><a href="#overview">↑ '+bi('Map','总览')+f'</a><h2>{section_title(title)}</h2>'
        body += '<details><summary>'+bi('Complete current file (imports and constants included)','完整当前文件（含导入与常量）')+'</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details>'
        for node in nodes:
            body += function_card(relative,node,lines,index)
            index += 1
        if key == 'rewrite':
            prompt_path = ROOT/'blackwell/compressors/prompts/telegrapher_v5.txt'
            lock = json.loads((prompt_path.parent/'telegrapher.lock.json').read_text(encoding='utf-8-sig'))
            if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != lock['sha256']:
                raise ValueError('Pinned grammar hash mismatch')
            url = lock['repository']+'/blob/'+lock['commit']+'/'+lock['source_path']
            body += '<details><summary>Telegrapher v5 · pinned grammar</summary><p><a href="'+html.escape(url)+'">'+html.escape(lock['authors'])+'</a> · '+html.escape(lock['license'])+'</p><pre>'+html.escape(prompt_path.read_text(encoding='utf-8'))+'</pre></details><p>'+bi('The lock records the grammar source and license. Its adaptation note dates from an earlier Qwen pilot; the current rewrite model is compression_model in the experiment configuration (DeepSeek-V3.2).','版本锁记录语法来源和许可；其中的 adaptation 备注来自早期 Qwen 试验。当前改写模型由实验配置的 compression_model 指定，本轮为 DeepSeek-V3.2。')+'</p>'
        body += '</section>'
    body += '<section><h2>'+bi('Where outputs live','输出在哪里')+'</h2><pre>runs/aligned50_&lt;dataset&gt;_20261024/\n  manifest.json / examples.json\n  aligned_compression_cache/     # context-only requests + all drafts\n  compression_api/              # rewrite API requests, responses and ledger\n  api_cache/ / api_attempts.json # reader requests, responses and ledger\n  predictions.json / .csv\n  metrics.json / diagnostics.json / audit.json\n  report.html\ndocs/Experiments_20261024_Aligned/Experiment_Overview.html</pre><p>'+bi('InfoGap is unchanged: exact breakpoints and one-sided limits, both directions, factor 2 plus saved unscaled values. ECE is a new diagnostic, not the objective used to choose compressed text. The six compressed conditions are then exactly length-matched per context at their common feasible budget, at most 50% of the original. Inspect initial and matching trims separately.','InfoGap 算法未改变：精确断点与单侧极限、双向、系数 2 及未缩放值。ECE 是新增诊断，不用于挑选压缩文本。六种压缩条件还会按每份上下文的共同可行预算严格等长，最多为原文 50%。需分别检查初次预算裁剪和共同长度裁剪。')+'</p></section>'
    page = '<!doctype html><html lang="en" data-language="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aligned experiment · Code walkthrough</title><style>'+STYLE+'.flow{display:flex;flex-wrap:wrap;gap:12px}.node{padding:20px;border:1px solid #abc;border-radius:8px;background:#edf4fb;text-decoration:none}.line-number{color:#8798a9;user-select:none}article{border-top:1px solid #dce5ee;margin-top:25px}pre{white-space:pre;overflow:auto}section{scroll-margin-top:15px}</style><body>'+body+'<script>function language(x){document.documentElement.dataset.language=x;document.documentElement.lang=x}</script></body></html>'
    target = ROOT/'docs/Code_Walkthrough_20261024_Aligned.html'
    target.write_text(page,encoding='utf-8')
    print(f'{target}: {index} functions with actual code and parameter explanations')


if __name__ == '__main__':
    main()
