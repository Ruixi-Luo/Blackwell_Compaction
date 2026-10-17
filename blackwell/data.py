"""BoolQ：保留原生标签，不制造答案，不按实验结果挑样本。"""
import json
import random
import zipfile

from .storage import ROOT, download, file_hash, fingerprint, write_json

DATASET_HOME = "https://github.com/google-research-datasets/boolean-questions"
DATASET_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
DATASET_SHA256 = "853fbe7922f70c59629f06a39e8d9ca440c3d740e760fd3b87a5ddf3dcba2436"


def load_boolq(split, sample_size, seed):
    if split not in {"train", "dev"}:
        raise ValueError("Only labeled BoolQ train/dev splits are supported.")
    # 使用 SuperGLUE 官方发行包；原始 Google bucket 当前返回 403。
    source_path = download(DATASET_URL, ROOT / "data" / "boolq" / "BoolQ.zip", DATASET_SHA256)
    member = "BoolQ/train.jsonl" if split == "train" else "BoolQ/val.jsonl"
    with zipfile.ZipFile(source_path) as archive:
        all_rows = [json.loads(line) for line in archive.read(member).decode("utf-8").splitlines() if line]
    if not 1 <= sample_size <= len(all_rows):
        raise ValueError("sample_size must be between 1 and the dataset size.")
    # 不分层、不重新平衡标签；固定随机种子，记录原始行号以便复查。
    indices = random.Random(seed).sample(range(len(all_rows)), sample_size)
    examples = []
    for index in indices:
        row = all_rows[index]
        if type(row["label"]) is not bool:
            raise ValueError("BoolQ answer must be a boolean.")
        context = row["passage"]  # 本版本只压缩 passage，不加入可选 title。
        examples.append({
            "id": f"boolq-{split}-{index:05d}",
            "source_row_zero_based": index,
            "context": context,
            "context_group": fingerprint(context),
            "question": row["question"],
            "label": int(row["label"]),
            "title": row.get("title", ""),
        })
    provenance = {
        "name": "BoolQ", "split": split, "source_url": DATASET_URL, "archive_member": member,
        "dataset_home": DATASET_HOME, "source_sha256": file_hash(source_path),
        "available_rows": len(all_rows), "sample_seed": seed,
        "license": "CC BY-SA 3.0 (as stated by the dataset authors)",
        "context_definition": "passage only; title excluded",
    }
    write_json(source_path.with_suffix(".provenance.json"), provenance)
    return examples, provenance

