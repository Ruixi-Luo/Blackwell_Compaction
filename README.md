# Blackwell Compaction

一个适合逐行阅读的 **query-blind 上下文压缩 + 二元概率预测 + InfoGap** 基础实验。

同一道 BoolQ 问题使用 4 种输入：原文 `full`、无上下文 `no_context`、首尾截断 `head_tail`、API 摘要 `summary`。两个压缩 baseline 都不能看到后续问题或真实标签。Reader 固定为硅基流动 `Qwen/Qwen3-30B-A3B-Instruct-2507`，输出 `{"p_yes": 0.73}`，表示 Bernoulli(0.73)。不训练模型、不下载模型权重、不需要 GPU。

这是验证代码与协议的小样本实验，不是论文级方法比较。摘要 baseline 是通用摘要，不声称复现 ACON 或 LLMLingua。

## 先看哪里

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
  predictions.csv        方便查看的长表，每条样本对应 4 行
  metrics.json           精确数值、双向矩阵、阈值和 bootstrap 范围
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
- 添加 baseline：在 `compression.py` 中只接收 `context` 和 `budget`，不要把整个带标签的 example 传进去。
- 更换模型前需同时核对 tokenizer、API 参数、JSON 格式和价格。第一版显式拒绝未经验证的模型设置。
- 下一阶段再增加 HotpotQA 原生 yes/no 子集；不把本次 BoolQ 小样本结果解释为长上下文 benchmark 结论。

## 来源

- [InfoGap · Feng, Qian, Tang](https://arxiv.org/html/2507.12094v2)
- [BoolQ 数据与 CC BY-SA 3.0 许可](https://github.com/google-research-datasets/boolean-questions)
- [SuperGLUE 官方发行包](https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip)
- [Qwen tokenizer 固定版本](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe)
- [SiliconFlow API 文档](https://docs.siliconflow.cn/docs/api/chat-completions-post)

报告中重现的 BoolQ 问题与 passage 归原数据作者所有，沿用其 CC BY-SA 3.0 许可；压缩文本是该材料的实验性改写。不要将数据许可误认为代码许可。
