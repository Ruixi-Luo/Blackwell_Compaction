"""下载固定 SALT 源码与 BGE-small 权重到 D 盘；不安装聊天依赖，不调用 API。"""
import argparse
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.compressors.runtime import asset_spec, configure_cache, file_digest, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    configure_cache()
    spec = asset_spec("salt_summary")
    if not args.check_only:
        for name, record in spec["files"].items():
            target = ROOT / spec["directory"] / name
            if target.exists():
                continue  # 最后的完整性检查拒绝已存在但被改动的文件。
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            print(f"Downloading {name} ({record['bytes']:,} bytes)", flush=True)
            request = Request(record["url"], headers={"User-Agent": "Blackwell-Compaction/0.3"})
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                while block := response.read(4 * 1024 * 1024):
                    output.write(block)
            if partial.stat().st_size != record["bytes"] or file_digest(partial) != record["sha256"]:
                raise ValueError(f"SALT asset checksum mismatch: {name}")
            os.replace(partial, target)
        # 上游模块 import 时使用此编码器作日志计数，预缓存后压缩本身可离线。
        import tiktoken
        tiktoken.encoding_for_model("gpt-4")
    verify_assets(["salt_summary"])
    print("Pinned SALT source and BGE-small verified. No LLM API calls made.")


if __name__ == "__main__":
    main()
