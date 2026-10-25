# Blackwell Compaction

一个用于 **query-blind 上下文压缩 → 二元概率预测 → 双向 InfoGap** 的科研实验仓库。当前日常入口只有 **`run.py`**；先读下面的流程，再看具体算法。

- **[Clean 代码导读：流程图、真实源码和逐参数讲解](docs/Clean_Code_Walkthrough.html)**（中英文切换，默认英文）
- **[100+100 题老师版报告](docs/Mini100_FirstDraft_20261025/Mini100_Report.html)** · [PDF](docs/Mini100_FirstDraft_20261025/Mini100_Report.pdf)
- [历史运行与旧导读索引](docs/Historical_README_20261025.md)：旧结果和旧协议完整保留。

HTML 下载后直接用浏览器打开。GitHub 的源文件预览不执行 HTML。

## 先理解数据怎样流动

```text
配置 → 原始数据集 → 固定抽样 → context → 压缩器 → 压缩文本
                            question ──────────→ reader → p_yes
                            label ─────────────────────→ 评估 → 报告
```

压缩器只接收完整 `context` 和长度预算，不接收问题、标签或 reader 反馈。Reader 共享同一个提示词，输出 JSON 概率 `p_yes`。评估使用相同题目的配对预测；这些口头概率不预设为真实或已校准后验。

## 在本机 CMD 复核本次实验：不收费

```bat
cd /d "D:\Research\Ph.D\2026.8.4-context compression\Experiments\2026.10.17 main Exp\Balckwell Compaction"
call .venv\Scripts\activate.bat
set PYTHONHASHSEED=0
python run.py --config configs/clean/mini100.json --stage all --offline
```

这会检查缓存、重放 1,200 个预测、重算双向 InfoGap、审计并重建 HTML。**不调用 API、不运行 GPU 压缩**。输出目录是 `runs/clean_mini100_verified/`，其中：

```text
manifest.json             配置、源码哈希、依赖版本、抽样记录
hotpotqa/、qasper/
  examples.json           固定的题目和完整上下文
  compressions/           压缩文本、实际长度和来源
  predictions.json        每题每方法的 p_yes
  metrics.json            评分、双向 InfoGap、bootstrap 区间
  diagnostics.json        概率可靠性与极端预测诊断
  audit.json              真实请求与结果的审计结论
api/reader/、api/summary/   实际 API 请求/响应和尝试账本
spending_reservations.json  累计费用与未知用量预留
Report.html               双语结果报告
Audit.html                离线逐题审计
```

本机已完成首次显式导入。迁移同一份旧结果时才需要以下命令；它不是新实验的前置条件：

```bat
python run.py --config configs/clean/mini100.json --import-run runs/mini100_firstdraft_20261025_v5 --stage all --offline
```

导入会重新从原始数据独立抽样，并检查与历史选题一致；验证原始摘要 prompt 和真实响应后复制缓存，继承累计费用和去重后的尝试记录。缺少旧运行目录时会报错，不会偷偷重新生成。

## 从头跑一个新实验

先复制配置 `configs/clean/smoke.json`，给它一个从未使用的 `run_name`（必须以 `clean_` 开头），先用每数据集 2 题检查。修改配置的主要字段：

| 字段 | 含义 |
|---|---|
| `run_name` | 新结果目录名称；改协议/源码/依赖版本后换名字，禁止覆盖旧实验 |
| `datasets` | `hotpotqa`、`qasper`，可只选一个 |
| `sample_size` | **每个数据集**抽几题 |
| `sample_seed` | 按 ID 排序后无放回抽样的固定种子 |
| `methods` | 条件列表；保留 `full` 和 `no_context` 对照 |
| `retained_ratio` | 保留比例，如 `0.5`，不是删除比例 |
| `device` | `cuda` 或 `cpu`；CUDA 不可用就报错 |
| `reader` / `summary` | 下游预测器/摘要模型请求设置；当前仅校验既有 Qwen/DeepSeek 组合 |
| `cost_reserve_cny` | 包含未知用量预留的累计预算，平台账单仍为准 |

本机依赖和模型已安装好。新终端里按上一节激活环境，然后：

```bat
python run.py --config configs/clean/smoke.json --stage prepare
python run.py --config configs/clean/smoke.json --stage all --allow-api
```

第二条会在缺少缓存时**真实调用 API、产生费用**，也会运行本地压缩。这次整理没有执行它。若不传 `--allow-api`，API 缓存缺失就失败；`--offline` 进一步禁止数据下载及本地模型推理。两者不能同时使用。

阶段可选 `prepare`、`compress`、`predict`、`evaluate`、`audit`、`report`、`all`。新实验无需任何旧 v3/v4 结果；不启动后台队列。相同配置重跑会续用准确缓存。一个运行目录只允许一个进程写入；意外强退后如留下 `.run.lock`，先确认里面记录的 PID 已结束，再处理锁文件。

仅重算指标和报告：

```bat
python run.py --config configs/clean/mini100.json --stage evaluate --offline
python run.py --config configs/clean/mini100.json --stage audit --offline
python run.py --config configs/clean/mini100.json --stage report --offline
```

新结果报告只需要 Python 标准库和既有实验依赖，不需要 ReportLab。本次老师版 PDF 保留为固定发布版本；clean 日常报告生成的是 HTML，可在浏览器另存为 PDF。

## 当前协议

- 方法：Full、No context、Selective Context、LLMLingua-2、普通 LLM summary、DAC。
- Reader：`Qwen/Qwen3-30B-A3B-Instruct-2507`；摘要：`deepseek-ai/DeepSeek-V3.2`。temperature=0，关闭 thinking，reader 输出上限128。
- **摘要只取原始 prompt 第一份完整输出**。不为凑长度修订、不裁剪、不按预测效果筛选。输出因长度截断时失败，不自动提高上限重新生成。网络层最多三次传输尝试，可能发生未知费用；不等于保证服务端只生成一次。
- 本地方法只按实际长度试探原生比例，SC/LLMLingua-2 最多8次、DAC最多4次；选最长完整合规输出，没有合规输出则失败。各方法独立预算，绝不向最短输出对齐。
- 50% 对摘要是**目标**、对本地方法是**上限**；报告实际长度与超目标数量，不能称等长排名。
- HotpotQA distractor dev；QASPER train 中标签一致且所有文字证据可在正文核验的二元子集。证据只定义样本池，不标记或裁剪压缩输入。相同正文复用压缩结果。
- InfoGap 保存双向、factor-2 与未归一化值，扫描准确断点和单侧极限；配对 bootstrap 按上下文成组。双向正值意味着不同决策的取舍，不证明 Blackwell 支配。

## 代码阅读地图

```text
run.py                      唯一日常命令行入口
configs/clean/              当前配置（mini100、smoke）
experiment/
  config.py                 参数验证、路径和运行身份
  data.py                   从原始数据集抽样
  pipeline.py               串联 prepare/compress/predict/evaluate/audit
  compression.py            统一压缩接口、独立预算与缓存
  summary.py                原始摘要 prompt + 第一稿策略
  api.py                    创建客户端、reader 单次预测
  evaluation.py             汇总指标和矩阵
  reporting.py              双语 HTML + 逐题审计
  import_results.py         可选历史导入；新实验不依赖它
blackwell/                  已验证、继续复用的底层库
  datasets/                 官方数据格式适配
  compressors/              各算法与 GPU/长度适配
  metrics.py                InfoGap、Brier、log loss、bootstrap 数学
  client.py、streaming.py    精确请求缓存、完整响应和失败处理
  cost_guard.py             费用预留
  storage.py、tokenization.py 文件与长度计量
```

`experiment/` 不导入 `tools/run_*` 或旧 `run_experiment.py`。复用已经验证的底层实现，避免再复制一套算法。历史入口保留原路径，因为旧 manifest 和缓存证据引用这些文件；**平时只用 run.py，历史脚本不是可替换的新入口**。

新增方法时：写独立压缩器，接入 `experiment/compression.py` 的工厂和方法表，再补边界测试。新增数据集时：写完整上下文加载器，接入 `experiment/data.py`。新增模型前需检查 tokenizer、响应格式及费用估算；目前不能只替换 URL 就宣称协议不变。

## 安装、测试与代码导读

本机已配置虚拟环境、CUDA和固定权重。新机器使用 `setup.ps1`、`setup-local.ps1 -Device cuda`，然后安装 `requirements-data.txt`，运行 `tools/setup_dac.py` 下载被锁定的 DAC 资产。所有数据、权重、缓存放在仓库的 `data/` 和 `.cache/`。API key 放本机 `.env` 或环境变量，不提交 Git。

```bat
python -m unittest discover -s tests -v
python tools/build_clean_guide.py
python tools/setup_dac.py --offline
```

[Clean 源码导读](docs/Clean_Code_Walkthrough.html) 首页流程框可点击；每块包含真实源码、参数说明、上一块/下一块。默认英文，可切中文。数学及底层网络/GPU代码在进阶部分。

本次验收记录见 [Clean_Validation.json](docs/Clean_Validation.json)：离线新流程与旧实验的选题、压缩文本、1,200 条概率、72 格 InfoGap 及 bootstrap 区间完全一致；不增加付费调用。不把缓存重放等同于独立重新生成。
