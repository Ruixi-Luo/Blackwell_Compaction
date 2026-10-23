# Blackwell Compaction

一个面向 **query-blind 上下文压缩、二元概率预测与 InfoGap** 的小型科研项目。目标是让每一步能读懂、检查和复跑。

```text
数据集 → context → 压缩器 → 压缩后的 context
         question ───────→ reader → p_yes
         label ──────────────────→ InfoGap / Brier / log loss
```

压缩器只能看到上下文与保留率，看不到问题、标签、gold evidence 或 reader 的预测。`p_yes` 表示 Bernoulli(p_yes) 的概率参数；目前由模型直接报告，不预设为真实贝叶斯后验。

## 先看结果与导读

- **[完整代码导读（零基础版）](docs/Code_Walkthrough_20261023.html)**：按照运行逻辑讲解 47 个 Python 文件、183 个函数/方法和 332 个参数；内嵌源码、配置说明、SALT 追加流程及 InfoGap 教学计算器。可搜索、切换中英文，默认英文；单独分享 HTML 即可阅读。
- **[SALT-summary 补充实验](docs/Experiments_20261023_SALT/Experiment_Overview.html)**：相同三个数据集，追加 query-blind SALT-summary；5×5 InfoGap 矩阵、实际 token 保留率、新旧调用分开统计。原四组预测和下方历史报告保留。
- **[三数据集实验与代码导读](docs/Experiments_20261018/Experiment_Overview.html)**：HotpotQA 458 题、QASPER 283 题、LongBench 52 题；四组输入、双向 InfoGap、收益差曲线、概率评分与逐题审计。
- **[数据集与代码导读](docs/Datasets_and_Experiment_Guide.html)**：HotpotQA 输入规则、QASPER 的真实可用规模、代码阅读顺序，中英文切换，默认英文。
- **[HotpotQA 30 条配对实验](docs/HotpotQA_Results.html)**：原文、空上下文、Selective Context、LLMLingua-2；可逐条检查原文、压缩文本与概率。
- [本地压缩器模块导读](docs/Local_Compressors.html)：官方算法封装、安装兼容性与调用示例。
- 历史结果保留：[BoolQ 六组实验](docs/Local_Compressors_Results.html)、[初始四组实验](docs/Pilot_Results.html)、[最初代码审计文档](docs/Code_Walkthrough_and_Audit.html)。这些是当时版本的快照。

GitHub 文件预览不直接执行 HTML；下载后用浏览器打开。HTML 自包含且默认英文，可切换中文。

## 当前实验边界

| 用途 | 配置 | 数据与方法 |
|---|---|---|
| 快速检查 | `configs/quickcheck.json` | BoolQ train 2 条；full、no_context、head_tail；不加载压缩模型 |
| 较长上下文 pilot | `configs/hotpotqa.json` | HotpotQA distractor dev 原生 yes/no 30 条；full、no_context、Selective Context、LLMLingua-2 |
| 全文数据可用性核验 | `tools/inspect_datasets.py` | QASPER v0.3 全部 train/dev/test；只统计，不调用 LLM |
| 多段落探索实验 | `configs/hotpotqa_full.json` | HotpotQA dev 全部 458 道 yes/no |
| 全文探索实验 | `configs/qasper_train.json` | QASPER train 的 283 道文字证据可核验 yes/no；233 篇论文 |
| 长上下文补充 | `configs/longbench_binary.json` | LongBench v1 标准英文版的 52 道一致 yes/no；不含 E/v2 |

Reader 固定为硅基流动 `Qwen/Qwen3-30B-A3B-Instruct-2507`，关闭思考、temperature=0。SC 使用官方 GPT-2 / phrase 算法；LLMLingua-2 使用官方 XLM-R-large MeetingBank 权重。新实验设置 `compression_device: cuda`，在 RTX 5070 Laptop GPU 上以 float32 运行；旧配置省略该字段时默认 CPU。没有训练模型。设备不可用会明确报错，不偷偷切换。

HotpotQA：7,405 个 dev 样本中有 **458 道原生 yes/no**（225 Yes / 233 No），全是 comparison 类型。保留全部提供段落及标题，不用 supporting facts 筛选；447 道有 10 段，另外 11 道只有 2–9 段。平均上下文约 **1,234 Qwen token**，是从 BoolQ 过渡到更长上下文的测试，还不是 4k–16k benchmark。原数据检索依赖问题，只有我们的压缩步骤与 query 隔离。

QASPER：要求正文非空、所有标注都是一致的 boolean 后，train/dev/test 分别为 **389 / 89 / 126** 题；再要求每份标注的文字证据都出现在全文输入中，剩 **283 / 51 / 88** 题。此次只用 train 的 283 题探索，dev/test 保留。证据仅用于子集资格判断，输入仍是完整标题、摘要、章节和段落；不按证据裁剪。保守筛选改变了标签比例（192 Yes / 91 No），不能代表全部 QASPER。相同全文共用压缩缓存，bootstrap 按全文成组。

LongBench：从固定 SHA-256 的官方标准 v1 归档筛选所有参考答案均为同一英文 yes/no 的题，得到 HotpotQA 14、2WikiMQA 19、QASPER 18、MultiFieldQA-en 1，共 52 题。原样保留 `context` 和 `input`。这不是官方完整 benchmark 的评测协议；不同数据集不合并算一个总排名。

长度适配：仅对超过 GPT-2 1024-token 上限的 SC 长句进行保留全部字符的分段，再使用全文统一短语阈值；仅对 LLMLingua-2 原生分块重新分词后超过 511 正文 token 的块进行同类分段。正常输入保留原生分块。QASPER 的 233 篇论文中 4 篇需要适配，HotpotQA/LongBench 均不需要。适配规则、GPU 型号、CUDA/PyTorch 版本都写入运行清单。GPU 耗时经过同步，不把异步提交时间当成实际推理时间。

## 第一次运行

在仓库根目录打开 PowerShell。数据、环境、模型及缓存都留在仓库所在的 D 盘，不上传 GitHub。

```powershell
# 基础环境；已经安装过则跳过。
.\setup.ps1

# HotpotQA 镜像是 Parquet，额外装这一个依赖。
$env:PIP_CACHE_DIR = "$PWD/.cache/pip"
$env:TEMP = "$PWD/.cache/tmp"
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt

# 两个本地压缩器：本机使用 NVIDIA GPU，安装 CUDA 12.8 构建。
# CUDA 包初次下载约 3.5 GB，模型权重约 2.8 GB；本机已安装，无需重装。
.\setup-local.ps1 -Device cuda
# 没有 NVIDIA GPU 的电脑才改用：.\setup-local.ps1 -Device cpu

# 免费检查代码，再准备数据（只下载公开数据/tokenizer，不调用 LLM）。
.\run.ps1 -Mode Test
.\run.ps1 -Mode Prepare -Config configs/hotpotqa.json

# 以下 Run 命令在缓存缺失时会消耗硅基流动 API 余额。
.\run.ps1 -Mode Run -Config configs/quickcheck.json
.\run.ps1 -Mode Run -Config configs/hotpotqa.json
```

原四组三数据集实验流程（历史说明）：

下方命令展示原四组实验的执行顺序。其配置中的运行名已对应冻结的旧源码；加入 SALT 后，当前核心代码身份发生变化，不能用这些旧运行名直接续跑。**本轮请使用下节的 SALT 补跑命令**。若要用当前代码重新运行原四组，先复制配置并设置新的 `run_name`；旧运行保持不动，复现旧源码则使用文末说明的历史提交。

```powershell
# 不花 API 费用，先检查全部模型输入长度。
.\.venv\Scripts\python.exe tools/preflight_inputs.py --config configs/hotpotqa_full.json
.\.venv\Scripts\python.exe tools/preflight_inputs.py --config configs/qasper_train.json
.\.venv\Scripts\python.exe tools/preflight_inputs.py --config configs/longbench_binary.json

# 会产生 API 费用。各数据集串行执行，避免同时调用 reader 触发共享限流。
# 458×4 + 283×4 + 52×4 = 3172 次名义调用；每组另有限额，重试计入。
.\.venv\Scripts\python.exe tools/run_current_suite.py

# 全部完成、审计及断网重放通过后，生成可分享报告。
.\.venv\Scripts\python.exe tools/build_suite_docs.py
```

每个数据集仍可单独用 `run.ps1 -Mode Run -Config ...` 执行。`tools/run_current_suite.py` 只组织现有步骤，不另建实验框架。完整结果与缓存保存在 `runs/<run_name>/`；报告、CSV 和指标快照在 `docs/Experiments_20261018/`。默认英文，可切换中文。CUDA 切换前中止的 CPU 试运行单独保留，不混入结果。最初并行调用 reader 触发了共享 HTTP 429 限流，后续改为串行并从缓存继续，失败尝试在账本中保留。

QASPER 在完成 282/283 题后因限流重试达到 1,350 次尝试上限；LongBench 在 49/52 题后达到 250 次。续跑分别另建运行名，将上限小幅提高到 1,380 / 280，逐文件验证并保留原 API 响应、压缩缓存和整份账本，仅补齐剩余题目。数据、模型、提示词和压缩设置均未改变；转移及前后结果一致性核验保存在续跑目录的 `continuation_receipt.json`，原运行保留。

辅助工具 `tools/continue_after_limit.py` 整理了这一续跑过程，`--help` 可查看参数。它只接受已达到尝试上限、已有本地压缩预检的运行；复制过程禁止网络，并要求数据、核心源码和压缩环境保持一致。

密钥从 `SILICONFLOW_API_KEY` 环境变量读取，也支持本地 `.env`，不要写进代码或配置。新运行的 BoolQ 快速检查名义上 6 次 reader 调用，HotpotQA 为 30 × 4 = 120 次，配置分别限制最多 20 / 150 次尝试（包括重试）。完全相同的请求会复用缓存。ChatGPT 订阅与 OpenAI API 计费独立，本项目当前不会调用 OpenAI。

如 Windows 阻止 `.ps1`，可用 `powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode Test`。`setup.ps1` 可用 `-PythonPath` 指定 Python；当前本地压缩环境验证于 Python 3.12。

## 追加 SALT-summary，而不重跑旧四组

`blackwell/compressors/salt_summary.py` 是一个独立压缩模块，接口仍为 `compress(context_text, retained_ratio)`。它使用固定版本的官方 **legacy** 选择器：清洗、分句与去重 → 主题组织与句子筛选 → 按原文顺序输出。所有 query 相关参数均为 `None`，模块不接收问题、标签或 reader 输出。模型是 BGE-small-en-v1.5；源码与权重版本、哈希保存在 `configs/salt_models.lock.json`，准备脚本负责下载并校验。数据、模型和缓存仍在 D 盘，无需更换现有 PyTorch/Transformers。

SALT 的名义 `0.5` 预算按**清洗去重后、激进过滤前的句子词数**计算，不是原文词数或 Qwen token 的一半。保留官方清洗和过滤后，实际长度可能明显偏离 50%；报告会如实展示。超长句在编码打分时可能截断，但选中句子的输出文本不因此截尾。该 query-blind 条件不是对论文 query-aware QA 主表的直接复现。

```powershell
# 保证官方实现的集合遍历次序可重复；每次启动新 Python 进程前设置。
$env:PYTHONHASHSEED = '0'

# 下载固定版本的公开代码和权重；不调用付费 reader。
.\.venv\Scripts\python.exe tools/prepare_salt.py
# 已有资产时可只校验：.\.venv\Scripts\python.exe tools/prepare_salt.py --check-only

# 先运行小样本；只有 predict 阶段在缓存缺失时调用付费 API。
.\.venv\Scripts\python.exe tools/extend_with_salt.py --config configs/salt_smoke.json --stage compress
.\.venv\Scripts\python.exe tools/extend_with_salt.py --config configs/salt_smoke.json --stage predict
.\.venv\Scripts\python.exe tools/extend_with_salt.py --config configs/salt_smoke.json --stage audit
.\.venv\Scripts\python.exe tools/extend_with_salt.py --config configs/salt_smoke.json --stage replay

# 三个正式数据集依次处理，避免 API 并发限流。
$saltConfigs = @('configs/salt_hotpotqa_full.json', 'configs/salt_qasper_train.json', 'configs/salt_longbench_binary.json')
foreach ($saltConfig in $saltConfigs) {
    .\.venv\Scripts\python.exe tools/extend_with_salt.py --config $saltConfig --stage compress
    if ($LASTEXITCODE -ne 0) { throw 'Compression failed; inspect the run before continuing.' }
    .\.venv\Scripts\python.exe tools/extend_with_salt.py --config $saltConfig --stage predict
    if ($LASTEXITCODE -ne 0) { throw 'Prediction failed; inspect the run before continuing.' }
    .\.venv\Scripts\python.exe tools/extend_with_salt.py --config $saltConfig --stage audit
    if ($LASTEXITCODE -ne 0) { throw 'Audit failed.' }
    .\.venv\Scripts\python.exe tools/extend_with_salt.py --config $saltConfig --stage replay
    if ($LASTEXITCODE -ne 0) { throw 'Offline replay failed.' }
}

# 发布到新目录；原报告保留。此步骤不调用 API。
.\.venv\Scripts\python.exe tools/build_suite_docs.py --configs $saltConfigs --destination docs/Experiments_20261023_SALT
```

补跑主流程集中在 `tools/extend_with_salt.py`：核验来源运行和代码身份 → 复制原样本及四组结果 → 只新增 SALT 压缩与预测 → 重新计算五组指标 → 审计、断网重放。每个正式数据集使用独立新运行名，`extension_receipt.json` 记录来源文件哈希与旧预测完全一致性；新 API 账本只记录补跑调用。可先读这一小工具，再读压缩模块和原有指标模块。

**时间差需要注意**：原四组预测采集于 10 月 18 日，SALT 在 10 月 23 日补跑。虽保持模型名称、提示词和配对题目一致，托管 reader 可能存在无法观察的版本或服务变化。因此这些结果仍有时间混杂，不能把差异全部归因于压缩方法。旧 SC/LLMLingua-2 压缩耗时是复用的历史记录，报告与新增 SALT 耗时分开列出。

## 不花 API 费用的审计与复跑

```powershell
# 核验原始数据、抽样、实际请求、概率、压缩长度及独立计算的 InfoGap。
.\run.ps1 -Mode Audit -Config configs/hotpotqa.json

# 仅从已保存预测重算指标和 HTML。
.\run.ps1 -Mode Report -Config configs/hotpotqa.json

# 禁止网络、清除本进程密钥、禁止重载压缩模型，验证逐字节一致的重放。
.\.venv\Scripts\python.exe tools/verify_offline_replay.py --config configs/hotpotqa.json

# 全量数据核验：首次去掉 --offline 以下载作者的 QASPER 归档。
.\.venv\Scripts\python.exe tools/inspect_datasets.py --offline

# 完成实验、审计和重放后，再生成可分享导读及结果快照。
.\.venv\Scripts\python.exe tools/build_dataset_docs.py
```

配置、源码或数据改变后必须换 `run_name`，避免混用实验；旧运行保留。历史报告可用 `Report` 重算，但严格 `Replay` 需要匹配该运行的源码、依赖版本和设备设置。本轮三数据集的核心源码保存在提交 `87391df`；历史 HotpotQA 30 题对应 `47c5b29`，BoolQ 六组对应 `1c3c58a`，初始四组对应 `fb641ef`。运行清单中的 `files_sha256` 记录实际执行文件的哈希；`git_commit_at_run` 记录准备运行时的 HEAD，代码当时可能尚未提交。需要复跑历史源码时使用独立 checkout，不覆盖当前工作目录。

## 学习代码的阅读顺序

1. **`configs/hotpotqa.json`**：先看实验中哪些量固定、哪些量允许改变。
2. **`blackwell/data.py` → `blackwell/datasets/`**：原始资料如何变成统一样本。每种数据集一个文件，BoolQ 的原有抽样逻辑保留；新增 QASPER 和 LongBench 沿用相同接口。
3. **`run_experiment.py`**：`prepare_run` → `generate_predictions` → `analyze`，串起整个实验。
4. **`blackwell/compression.py` → `blackwell/compressors/`**：不同算法怎样遵守同一个文本输入/输出边界。
5. **`blackwell/prompts.py` 与 `blackwell/client.py`**：reader 看到什么，以及 API 缓存、重试、限额如何实现。
6. **`blackwell/metrics.py`**：不依赖网络的数学计算。`tests/` 展示如何检查标签、信息隔离、边界值和回归。

不必先理解所有辅助工具。`tools/` 负责核验和文档，实验主流程没有依赖它们；`docs/` 保存给人读的快照；`runs/<run_name>/` 保存完整本地实验。

新增数据集只需写一个 loader 并在 `blackwell/data.py` 注册。它返回普通字典列表（含 `id, context, question, label, context_group`）及来源记录。**不要把整个样本字典传给压缩器**。新增模型则需另行核验 tokenizer、API 参数及概率格式；当前 reader 有明确的模型限制，不是改任意链接就保证兼容。

```python
from blackwell.compressors import create_compressor

compressor = create_compressor("llmlingua2", device="cuda")  # 或 selective_context；CPU 用 device="cpu"
compressed_text = compressor.compress(context_text, retained_ratio=0.5)
```

## 读指标时注意什么

`retained_ratio` 是名义保留率。SC 的删除率参数取 `1 - retained_ratio`；两个官方方法的词法单位不同，实际 Qwen token 数不一定等于预算。本项目保留原生输出、不事后截断，报告实际长度及超预算次数，因此不是严格等长度排名。

InfoGap 的实现约定为：

```text
G(A,B) = 2 × sup over t∈[0,1] of mean[(t-y)(1[p_A≤t] − 1[p_B≤t])]
```

必须同时看 `G(A,B)` 与 `G(B,A)`。实现扫描全部概率断点和单侧极限，保存带系数 2 与未缩放的结果。arXiv v2 的定理 4.1 和定理 5.1 证明含系数 2，但定理 5.1 展示式未写；这是我们明确选择的实现约定，不是作者确认的勘误。小样本 bootstrap 只作描述性波动范围，不是已验证的支配检验。

新报告另给出阈值收益差曲线（含左右极限）、Brier / log loss / accuracy 及相对原文的变化量。耗时按独立压缩结果累计，不把同篇论文多道题复用的耗时重复相加。InfoGap 在各数据集分别计算；LongBench 同时给出各来源任务的小样本描述。

SC 0.1.4 的旧 spaCy / click 依赖约束不兼容当前环境，安装时显式绕过，使用已经验证的 spaCy 3.8.7 / click 8.5.0；`pip check` 会显示这两项已知偏差。模型版本、官方源码及权重校验值均有记录，详见本地压缩器导读。

## 来源与数据许可

- [HotpotQA（EMNLP 2018）](https://hotpotqa.github.io/) · CC BY-SA 4.0；原服务器不可用时使用已固定 revision 和 SHA-256 的 [Hugging Face 镜像](https://huggingface.co/datasets/hotpotqa/hotpot_qa)。
- [QASPER（NAACL 2021）](https://aclanthology.org/2021.naacl-main.365/) · CC BY 4.0；[官方 evaluator](https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py)。
- [BoolQ](https://github.com/google-research-datasets/boolean-questions) · CC BY-SA 3.0。
- [Selective Context](https://github.com/liyucheng09/Selective_Context)、[LLMLingua-2](https://github.com/microsoft/LLMLingua)、[SALT](https://github.com/oteomamo/SALT)。
- [InfoGap · Feng, Qian, Tang](https://arxiv.org/html/2507.12094v2)。

报告中的数据摘录及实验性压缩改写遵循对应数据许可，不能将数据许可误认为代码许可。原始数据、模型、密钥和 API 响应缓存均由 Git 忽略；公开 HTML、统计和概率表保存在 `docs/`。
