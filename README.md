# Blackwell Compaction

一个面向 **query-blind 上下文压缩、二元概率预测与 InfoGap** 的小型科研项目。目标是让每一步能读懂、检查和复跑。

```text
数据集 → context → 压缩器 → 压缩后的 context
         question ───────→ reader → p_yes
         label ──────────────────→ InfoGap / Brier / log loss
```

压缩器只能看到上下文与保留率，看不到问题、标签、gold evidence 或 reader 的预测。`p_yes` 表示 Bernoulli(p_yes) 的概率参数；目前由模型直接报告，不预设为真实贝叶斯后验。

## 先看结果与导读

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

新一轮三数据集实验（已安装 CUDA 环境后）：

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

密钥从 `SILICONFLOW_API_KEY` 环境变量读取，也支持本地 `.env`，不要写进代码或配置。新运行的 BoolQ 快速检查名义上 6 次 reader 调用，HotpotQA 为 30 × 4 = 120 次，配置分别限制最多 20 / 150 次尝试（包括重试）。完全相同的请求会复用缓存。ChatGPT 订阅与 OpenAI API 计费独立，本项目当前不会调用 OpenAI。

如 Windows 阻止 `.ps1`，可用 `powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode Test`。`setup.ps1` 可用 `-PythonPath` 指定 Python；当前本地压缩环境验证于 Python 3.12。

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

配置、源码或数据改变后必须换 `run_name`，避免混用实验；旧运行保留。历史报告可用 `Report` 重算，但严格 `Replay` 需要匹配该运行的源码。HotpotQA 本次核心源码在提交 `47c5b29`；BoolQ 六组历史运行对应 `1c3c58a`，初始四组对应 `fb641ef`。需要复跑历史源码时使用独立 checkout，不覆盖当前工作目录。

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
- [Selective Context](https://github.com/liyucheng09/Selective_Context)、[LLMLingua-2](https://github.com/microsoft/LLMLingua)。
- [InfoGap · Feng, Qian, Tang](https://arxiv.org/html/2507.12094v2)。

报告中的数据摘录及实验性压缩改写遵循对应数据许可，不能将数据许可误认为代码许可。原始数据、模型、密钥和 API 响应缓存均由 Git 忽略；公开 HTML、统计和概率表保存在 `docs/`。
