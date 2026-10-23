"""固定权重版本和本地依赖；所有缓存放在仓库的 D 盘目录。"""
import hashlib
import importlib.metadata
import os
import tempfile
from pathlib import Path

from ..storage import ROOT, read_json

PACKAGES = ("torch", "transformers", "tokenizers", "accelerate", "llmlingua",
            "selective-context", "spacy", "en-core-web-sm", "numpy", "nltk",
            "tiktoken", "safetensors", "sentencepiece", "click")


def configure_cache():
    paths = {"HF_HOME": ".cache/huggingface", "TIKTOKEN_CACHE_DIR": ".cache/tiktoken",
             "NLTK_DATA": ".cache/nltk", "TORCH_HOME": ".cache/torch",
             "TEMP": ".cache/tmp", "TMP": ".cache/tmp"}
    for variable, relative in paths.items():
        folder = ROOT / relative
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(folder)
    tempfile.tempdir = os.environ["TEMP"]
    # 模型只从已下载、已固定的本地目录加载；推理时不临时追踪远端 main。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def asset_spec(method):
    lock_name = "salt_models.lock.json" if method == "salt_summary" else "local_models.lock.json"
    return read_json(ROOT / "configs" / lock_name)[method]


def model_directory(method):
    folder = ROOT / asset_spec(method)["directory"]
    if method == "salt_summary":
        folder = folder / "model"
    if not (folder / "model.safetensors").exists():
        raise RuntimeError("Local model missing. Run .\\setup-local.ps1, then prepare_local_models.py.")
    return folder


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_assets(methods):
    """运行开始时逐块验哈希，不把数 GB 权重一次性读进内存。"""
    for method in methods:
        spec = asset_spec(method)
        for name, record in spec["files"].items():
            path = ROOT / spec["directory"] / name
            if not path.exists() or path.stat().st_size != record["bytes"] or file_digest(path) != record["sha256"]:
                raise RuntimeError(f"Missing or changed local asset: {method}/{name}")


def runtime_identity(methods, device='cpu'):
    if device not in {'cpu', 'cuda'}:
        raise ValueError('compression_device must be cpu or cuda.')
    hardware = {}
    if device == 'cuda':
        torch = prepare_torch(device)
        hardware = {'gpu_name': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
                    'compute_capability': list(torch.cuda.get_device_capability(0)), 'tf32': False}
    try:
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Optional local dependencies missing; run .\\setup-local.ps1 first.") from error
    upstream_files = {}
    for distribution, relative in [("selective-context", "selective_context/__init__.py"),
                                   ("llmlingua", "llmlingua/prompt_compressor.py"),
                                   ("llmlingua", "llmlingua/utils.py")]:
        path = importlib.metadata.distribution(distribution).locate_file(relative)
        upstream_files[relative] = file_digest(path)
    identity = {"device": device, "hardware": hardware, "dtype": "float32", "torch_threads": 4,
            "upstream_files_sha256": upstream_files,
            "packages": versions, "assets": {m: asset_spec(m) for m in methods},
            "selective_context": {"language": "en", "reduce_level": "phrase", "model_type": "gpt2",
                                  "overlong_sentence_policy": "lossless whitespace-first split to <=1024 tokens including BOS; one global phrase threshold"},
            "llmlingua2": {"use_context_level_filter": False, "token_to_word": "mean",
                           "force_tokens": [], "force_reserve_digit": False,
                           "drop_consecutive": False, "max_batch_size": 1,
                           "overlong_chunk_policy": "lossless whitespace-first split only if native chunk exceeds 511 body tokens"},
            "budget_policy": "one native call at requested retained_ratio; no post-truncation"}
    if "salt_summary" in methods:
        from .salt_summary import SALT_SETTINGS
        identity["salt_summary"] = SALT_SETTINGS
    return identity


def prepare_torch(device='cpu'):
    configure_cache()
    import torch
    if device not in {'cpu', 'cuda'}:
        raise ValueError('compression_device must be cpu or cuda.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install CUDA PyTorch with setup-local.ps1 -Device cuda.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch
