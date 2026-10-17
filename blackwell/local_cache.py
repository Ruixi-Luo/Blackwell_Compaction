"""缓存本地压缩结果；缓存身份只含 context、长度设置、模型和软件版本。"""
import time

from .compressors import create_compressor
from .storage import fingerprint, read_json, write_json


class LocalCompressionCache:
    def __init__(self, run_dir, identity, offline=False):
        self.directory = run_dir / "local_compression_cache"
        self.identity = identity
        self.offline = offline
        self.engines = {}

    def compress(self, context, budget, method, retained_ratio):
        request = {"context": context, "budget_tokens": budget, "method": method,
                   "retained_ratio": retained_ratio, "runtime": self.identity}
        key = fingerprint(request)
        path = self.directory / f"{key}.json"
        if path.exists():
            entry = read_json(path)
            if entry["request"] != request or entry["result_sha256"] != fingerprint(entry["result"]):
                raise ValueError("Local compression cache failed integrity checks.")
            return {**entry["result"], "local_compression_hash": key}
        if self.offline:
            raise RuntimeError("Offline local compression cache miss; inference is disabled.")
        if method not in self.engines:
            self.engines[method] = create_compressor(method)
        start = time.perf_counter()
        text = self.engines[method].compress(context, retained_ratio)
        if not isinstance(text, str):
            raise TypeError("Local compressor must return text.")
        result = {"text": text, "compression_seconds": time.perf_counter() - start,
                  "native_retained_ratio": retained_ratio}
        write_json(path, {"request": request, "result": result, "result_sha256": fingerprint(result)})
        return {**result, "local_compression_hash": key}
