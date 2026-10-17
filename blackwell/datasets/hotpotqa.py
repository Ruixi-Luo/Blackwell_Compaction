"""HotpotQA 原生 yes/no：保留 distractor 输入，不使用 gold supporting facts。"""
import random

from ..storage import ROOT, download, fingerprint

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SOURCE_URL = (f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{REVISION}/"
              "distractor/validation-00000-of-00001.parquet")
SOURCE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
HOME = "https://hotpotqa.github.io/"


def build_context(context):
    """只接收段落字段：标题 + 原顺序句子；不接收问题、答案或 supporting_facts。"""
    titles, paragraphs = context["title"], context["sentences"]
    if not titles or len(titles) != len(paragraphs):
        raise ValueError("Empty or mismatched HotpotQA paragraph fields.")
    blocks = []
    for title, sentences in zip(titles, paragraphs):
        if not sentences or not all(isinstance(s, str) for s in sentences):
            raise ValueError("Malformed HotpotQA paragraph.")
        blocks.append(title + "\n" + " ".join(s.strip() for s in sentences))
    return "\n\n".join(blocks)


def convert_rows(rows):
    """先筛原生标签，再统一格式；不根据长度、类别比例或模型结果挑选。"""
    examples = []
    for index, row in enumerate(rows):
        answer = row["answer"].strip().lower()
        if answer not in {"yes", "no"}:
            continue
        context = build_context(row["context"])
        examples.append({
            "id": f"hotpotqa-dev-{row['id']}", "source_row_zero_based": index,
            "context": context, "context_group": fingerprint(context),
            "question": row["question"], "label": int(answer == "yes"),
            "question_type": row["type"], "paragraph_count": len(row["context"]["title"]),
        })
    if len({e["id"] for e in examples}) != len(examples):
        raise ValueError("Duplicate HotpotQA IDs.")
    return examples


def load_pool(offline=False):
    # 延迟导入：BoolQ 和 QASPER 不需要这个可选依赖。
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("HotpotQA requires: python -m pip install -r requirements-data.txt") from error
    path = download(SOURCE_URL, ROOT / "data/hotpotqa/validation.parquet", SOURCE_SHA256, offline)
    rows = parquet.read_table(path).to_pylist()
    examples = convert_rows(rows)
    provenance = {
        "name": "HotpotQA yes/no (distractor)", "split": "dev", "dataset_home": HOME,
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256, "revision": REVISION,
        "source_format": "Hugging Face Parquet mirror; dev is named validation",
        "raw_rows": len(rows), "available_rows": len(examples), "license": "CC BY-SA 4.0",
        "selection_rule": "answer.strip().lower() in {yes, no}; no balancing or outcome filtering",
        "context_definition": "All supplied titles and paragraphs (usually 10), original order; sentences joined by spaces. No supporting_facts.",
        "scope": "The supplied retrieved context is question-conditioned; only the compressor is query-blind.",
    }
    return examples, provenance


def load_hotpotqa(split, sample_size, seed, offline=False):
    if split != "dev":
        raise ValueError("This pilot supports HotpotQA distractor dev only.")
    pool, provenance = load_pool(offline)
    if not 1 <= sample_size <= len(pool):
        raise ValueError("sample_size exceeds the eligible native yes/no pool.")
    indices = random.Random(seed).sample(range(len(pool)), sample_size)
    return [pool[i] for i in indices], {**provenance, "sample_seed": seed}
