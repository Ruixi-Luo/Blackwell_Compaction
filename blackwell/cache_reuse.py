"""从已完成的运行复用原始 API 响应；仍由 client 的完整请求指纹决定能否命中。"""
import re
import shutil

from .storage import ROOT, file_hash, fingerprint, read_json, write_json


def import_api_cache(config, destination, offline=False):
    source_name = config.get("reuse_api_cache_from")
    if not source_name:
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_name) or source_name == config["run_name"]:
        raise ValueError("Invalid source run name.")
    receipt = destination / "cache_import.json"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["source_run"] != source_name:
            raise ValueError("Cache import source changed.")
        for name, digest in saved["files_sha256"].items():
            if file_hash(destination / "api_cache" / name) != digest:
                raise ValueError("Imported API response changed.")
        return
    if offline:
        raise RuntimeError("Offline replay needs the saved cache-import receipt.")
    source = ROOT / "runs" / source_name
    manifest = read_json(source / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only complete runs can supply reusable API responses.")
    old = manifest["identity"]["config"]
    for name in ("provider", "base_url", "model", "temperature", "enable_thinking", "prediction_max_tokens"):
        if old[name] != config[name]:
            raise ValueError(f"Reader configuration differs: {name}")
    files = {}
    for path in sorted((source / "api_cache").glob("*.json")):
        entry = read_json(path)
        if fingerprint(entry["request"]) != path.stem:
            raise ValueError("Source API cache request fingerprint differs.")
        target = destination / "api_cache" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = file_hash(path)
        if target.exists() and file_hash(target) != digest:
            raise ValueError("Refusing to overwrite a different API response.")
        if not target.exists():
            shutil.copyfile(path, target)
        files[path.name] = digest
    write_json(receipt, {"source_run": source_name, "source_manifest_sha256": file_hash(source / "manifest.json"),
                         "files_sha256": files, "count": len(files)})
