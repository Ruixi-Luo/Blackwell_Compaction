# Blackwell Compaction

一个适合逐行阅读的 **query-blind 上下文压缩 + 二元概率预测 + InfoGap** 基础实验。

同一道 BoolQ 问题使用 4 种输入：原文 `full`、无上下文 `no_context`、首尾截断 `head_tail`、API 摘要 `summary`。两个压缩 baseline 都不能看到后续问题或真实标签。Reader 固定为硅基流动 `Qwen/Qwen3-30B-A3B-Instruct-2507`，输出 `{"p_yes": 0.73}`，表示 Bernoulli(0.73)。这个原始 API 版不训练模型、不下载模型权重、不需要 GPU。新增的可选本地版还支持官方 **Selective Context** 和 **LLMLingua-2**：压缩器在本地运行，reader 继续使用 API。

这是验证代码与协议的小样本实验，不是论文级方法比较。摘要 baseline 是通用摘要，不声称复现 ACON 或 LLMLingua。

## 新增：Selective Context 与 LLMLingua-2

- **[模块导读和对比结论](docs/Local_Compressors.html)**：中英文切换、接口示例、兼容性说明、可展开源码。
- **[六组配对实验结果](docs/Local_Compressors_Results.html)**：同一批 30 条样本，保留旧四组预测，增加两个官方方法。
- 官方代码：[Selective Context](https://github.com/liyucheng09/Selective_Context)（EMNLP 2023）与 [LLMLingua-2](https://github.com/microsoft/LLMLingua)（ACL Findings 2024）。

```python
from blackwell.compressors import create_compressor

# 第一次创建会加载本地权重；创建一次后可反复压缩。
compressor = create_compressor("llmlingua2")
# 或 create_compressor("selective_context")
compressed_text = compressor.compress(context_text, retained_ratio=0.5)
```

输入、输出均为字符串。接口不接收 query、label 或 reader 预测。Selective Context 使用 GPT-2 + phrase；LLMLingua-2 使用官方 XLM-R-large MeetingBank 模型，明确启用二代算法。无训练，CPU 即可运行。本地权重约 2.8 GB（另加依赖和缓存），全部在 D 盘；压缩不消耗 LLM API。

```powershell
# 在基础环境 setup.ps1 之后，安装一次可选模型和依赖。
.\setup-local.ps1

# 本机已有的六组实验：重算报告、审计和完全离线重放。
.\run.ps1 -Mode Report -Config configs/local_comparison.json
.\run.ps1 -Mode Audit -Config configs/local_comparison.json
.\run.ps1 -Mode Replay -Config configs/local_comparison.json

# 新克隆没有旧 runs 缓存时：原文、空上下文、SC、LLMLingua-2 四组。
# 这个命令会产生付费 API 调用；运行前核对配置和密钥。
.\run.ps1 -Mode Run -Config configs/local_fresh.json
```

`configs/local_comparison.json` 会复用原 pilot 的完全相同 API 请求缓存；本机预计仅需为两个新方法增加 60 次 reader 调用。`configs/local_fresh.json` 不依赖旧运行。两者都使用 30 条相同 seed 的 BoolQ dev 样本，实际 API 次数由去重和重试决定。

**长度解释**：`retained_ratio` 表示保留率；SC 的原生参数是删除率，因此使用 `reduce_ratio = 1 - retained_ratio`。原生短语/词筛选不能精确保证 Qwen token 预算，本项目记录实际 token 数并保留超预算输出，不追加截断。这是相同名义比例的 pilot，不是严格等 token 预算排名。

**依赖兼容性**：SC 0.1.4 原始 spaCy 3.2.0 与当前 Python 3.12 不兼容。使用官方包的算法，覆盖模型加载钩子以固定本地权重，采用已验证的 spaCy 3.8.7。安装时显式绕过其旧 spaCy / click 依赖锁定；`pip check` 会报告这两项已知偏差。实际版本、模型 revision、文件 SHA256 和官方源码哈希会写入运行记录。详见导读。这不声称精确复现论文的原始软件环境。

旧 `docs/Pilot_Results.html` 和原导读保留不变。旧配置仍可用于 Report / Audit；旧运行的严格 Replay 需要与其记录一致的原源码（`fb641ef`，使用独立 checkout），否则哈希检查会按设计拒绝。后续更改协议或源码仍需新 `run_name`。

## 原始四组 pilot：先看哪里

- **[代码导读与审计文档](docs/Code_Walkthrough_and_Audit.html)**：支持中英文，默认英文；含流程、公式、真实结果和可展开源码。
- **[已运行的实验结果](docs/Pilot_Results.html)**：逐条查看问题、标签、压缩文本、四组概率和 InfoGap。
- [实验配置](configs/pilot.json)：固定随机种子，抽取 30 个 BoolQ dev 样本，目标保留原文 50% token。
- [InfoGap 实现](blackwell/metrics.py)：核心数学与网络、LLM 完全分离。

## 在这台电脑运行

在仓库文件夹打开 PowerShell。所有命令都从仓库根目录执行；不用激活虚拟环境。

```powershell
# 第一次安装。环境和依赖缓存都在仓库所在的 D 盘。
.\setup.ps1

# 不消耗 LLM API：检查数学、缓存、数据隔离和预算约束。
.\run.ps1 -Mode Test

# 不消耗 LLM API：核对真实请求、概率与独立计算的决策收益。
.\run.ps1 -Mode Audit

# 不消耗 LLM API：从已有预测重算指标和 HTML。
.\run.ps1 -Mode Report

# 不访问网络：严格使用已经保存的 API 响应重放完整流程。
.\run.ps1 -Mode Replay

# 真正调用 API：已有的完全相同请求会命中本地缓存。
.\run.ps1 -Mode Run
```

若 Windows 拒绝运行 `.ps1`，可使用 `powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode Report`。在其他电脑上，安装 Python 3.10+，然后执行 `python -m venv .venv`、`.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`；本机 `setup.ps1` 默认复用已安装的 Codex Python，也可传入 `-PythonPath` 指定自己的 Python。

API 密钥从 `SILICONFLOW_API_KEY` 环境变量读取；也支持将 `.env.example` 复制成 `.env` 后在本地填写。不要将密钥写入代码、配置或聊天消息。`.env` 已被 Git 忽略。OpenAI 的 ChatGPT Pro 订阅与 OpenAI API 计费不同；本版本只实现并验证了 SiliconFlow，不会自动改用 OpenAI。

## 实验怎样运行

1. 从 SuperGLUE 官方发行包读取 BoolQ，核验下载文件 SHA-256。`dev` 对应包内的 `val.jsonl`。
2. 按固定 seed 从全体样本均匀无放回抽取，不根据标签或结果筛选。
3. 只取 `passage` 为上下文。`question` 留给 reader；`label` 只留给评估。
4. `head_tail` 按官方 Qwen tokenizer 保留首尾，截断标记也计入预算。
5. `summary` 用同一 API 模型压缩，仅传入 passage 和预算；超长最多修复一次，仍超预算则明确标记并保留，不偷偷截断。
6. 同一 reader 对四种上下文回答同一个问题，严格解析 JSON 概率。
7. 汇总配对预测，计算双向 InfoGap、Brier、Log loss 和辅助 Accuracy。

模型直接报告的概率不等同于真实贝叶斯后验，也不是 token log probability。我们的结论只涉及固定 reader、提示词、数据分布下的概率预测；不直接证明压缩文本本身对任意 query 的 Blackwell 支配。

## 保存了什么

```text
configs/                 样本数、seed、模型和预算
blackwell/               核心模块，每个文件只负责一件事
tests/                   不调用真实模型的测试
docs/                    可分享的导读与已运行结果
data/                    数据与 tokenizer，仅本地，Git 忽略
.venv/ .cache/           本地 Python 环境和缓存，Git 忽略
runs/<run_name>/
  manifest.json          配置、数据/代码校验值、版本与运行状态
  examples.json          固定样本及其来源行号
  predictions.json       每条样本的完整上下文、压缩文本和概率
  predictions.csv        方便查看的长表，每条样本对应 methods 中各方法的一行
  metrics.json           精确数值、双向矩阵、阈值和 bootstrap 范围
  audit.json             对实际请求、概率、预算与独立决策收益的核验
  local_compression_cache/  本地压缩输入、输出与模型/依赖指纹（本地版）
  cache_import.json     旧 API 响应的来源及逐文件哈希（复用时）
  api_cache/*.json       不含密钥的实际请求正文与原始 API 响应
  api_attempts.json      含失败/重试在内的 API 次数与返回 usage
  report.html            自包含交互结果页
```

`runs/` 不自动上传，公开结果快照在 `docs/`。JSON 保存计算精度，HTML 显示时才四舍五入。配置或 Python 源码改变后，必须换一个 `run_name`，防止把不同实验混在一起。`Run` 会使用匹配的缓存；如果想获取新的随机实现，也要换 `run_name`。即使 temperature=0，远端模型也不保证逐字复现。

## InfoGap 的方向与公式

定义 `h_t(p,y) = (t-y) × 1[p ≤ t]`，本实现计算

```text
G(A,B) = 2 × sup over t∈[0,1] of mean[h_t(p_A,y) − h_t(p_B,y)]
```

`G(A,B)` 是 A 相对 B 的最大收益优势，不能与 `G(B,A)` 交换。两边都可能为正。估计值为 0 不等于在小样本下证明了总体支配。

论文 arXiv v2 的定理 4.1 及定理 5.1 证明含系数 2，但定理 5.1 展示的 estimator 未写 2。这里明确采用前者，`metrics.json` 同时保存 `value` 与 `unscaled`，后者为前者的一半。等待作者最终版本核实，而不是隐藏该差异。实现扫描全部概率断点及左右极限；不先校准、不分桶、不用粗网格。

## 怎么扩展

- 先复制 `configs/pilot.json`，修改 `run_name` 与 `sample_size`；`max_api_attempts` 要覆盖约 `5 × 样本数` 加长度修复与重试，默认上限会中止超额调用。
- `retained_ratio` 表示保留率，例如 `0.25` 对应目标 4 倍压缩。
- 添加 baseline：参照 `blackwell/compressors/` 的独立模块，在配置 `methods` 中选择；只接收上下文和长度参数，不要传整个带标签的 example。
- 更换模型前需同时核对 tokenizer、API 参数、JSON 格式和价格。第一版显式拒绝未经验证的模型设置。
- 下一阶段再增加 HotpotQA 原生 yes/no 子集；不把本次 BoolQ 小样本结果解释为长上下文 benchmark 结论。

## 来源

- [InfoGap · Feng, Qian, Tang](https://arxiv.org/html/2507.12094v2)
- [BoolQ 数据与 CC BY-SA 3.0 许可](https://github.com/google-research-datasets/boolean-questions)
- [SuperGLUE 官方发行包](https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip)
- [Qwen tokenizer 固定版本](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe)
- [SiliconFlow API 文档](https://docs.siliconflow.cn/docs/api/chat-completions-post)

报告中重现的 BoolQ 问题与 passage 归原数据作者所有，沿用其 CC BY-SA 3.0 许可；压缩文本是该材料的实验性改写。不要将数据许可误认为代码许可。
