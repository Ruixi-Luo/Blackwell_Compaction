"""统一文件读写：路径都相对于仓库，默认在 D 盘。"""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent


def fingerprint(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """先写临时文件，再替换；意外退出不会留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def download(url, path, expected_sha256=None):
    """只下载明确指定的公开数据或 tokenizer；不下载模型权重。"""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers={"User-Agent": "Blackwell-Compaction/0.1"})
        with urlopen(request, timeout=90) as response:
            content = response.read()
        if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("Downloaded file has an unexpected SHA-256.")
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    if expected_sha256 and file_hash(path) != expected_sha256:
        raise ValueError(f"File checksum mismatch: {path.name}")
    return path

