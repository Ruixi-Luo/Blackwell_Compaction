"""从这里开始阅读：准备数据 -> 压缩 -> 预测 -> 配对评估 -> HTML 报告。"""
import argparse
import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from blackwell.client import CachedChatClient, parse_probability
from blackwell.compression import Compactor
from blackwell.compressors import LOCAL_METHODS
from blackwell.compressors.llm_rewrite import API_METHODS
from blackwell.cache_reuse import import_api_cache
from blackwell.data import load_dataset
from blackwell.metrics import bootstrap_gaps, forecast_scores, infogap, payoff_curve
from blackwell.prompts import prediction_messages
from blackwell.storage import ROOT, file_hash, fingerprint, read_json, write_json

DEFAULT_METHODS = ("full", "no_context", "head_tail", "summary")


def selected_methods(config):
    methods = config.get("methods", list(DEFAULT_METHODS))
    allowed = set(DEFAULT_METHODS) | set(LOCAL_METHODS) | set(API_METHODS)
    if (not isinstance(methods, list) or len(methods) != len(set(methods))
            or not {"full", "no_context"}.issubset(methods) or set(methods) - allowed):
        raise ValueError("methods must be a unique list of supported names including full and no_context.")
    return methods


def load_local_keys():
    """可选 .env 仅允许两种密钥；已有环境变量优先，不打印任何值。"""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in {"SILICONFLOW_API_KEY", "OPENAI_API_KEY"}:
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def load_config(path):
    config = read_json(path)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_name"]):
        raise ValueError("run_name must contain only letters, digits, _ or -.")
    if type(config["sample_size"]) is not int or config["sample_size"] < 1:
        raise ValueError("sample_size must be a positive integer.")
    if not 0 < config["retained_ratio"] < 1:
        raise ValueError("retained_ratio must be between 0 and 1.")
    # 第一版只验证这一组合；避免改模型后仍悄悄使用错误 tokenizer/参数。
    if (config["provider"], config["base_url"], config["model"], config["api_key_env"]) != (
        "siliconflow", "https://api.siliconflow.cn/v1", "Qwen/Qwen3-30B-A3B-Instruct-2507", "SILICONFLOW_API_KEY"
    ):
        raise ValueError("Version 0.1 supports SiliconFlow Qwen/Qwen3-30B-A3B-Instruct-2507 only. See the audit guide before adding models.")
    if config["enable_thinking"] is not False:
        raise ValueError("This pilot freezes enable_thinking=false.")
    for key in ("prediction_max_tokens", "max_api_attempts", "bootstrap_repetitions"):
        if type(config[key]) is not int or config[key] < 2:
            raise ValueError(f"Invalid setting: {key}")
    if config["min_interval_seconds"] < 0 or config["timeout_seconds"] <= 0:
        raise ValueError("Invalid timing setting.")
    selected_methods(config)
    if config.get('compression_device', 'cpu') not in {'cpu', 'cuda'}:
        raise ValueError('compression_device must be cpu or cuda.')
    return config


def code_provenance():
    paths = [ROOT / "run_experiment.py", *sorted((ROOT / "blackwell").rglob("*.py"))]
    hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"python": platform.python_version(), "files_sha256": hashes,
            "git_commit_at_run": result.stdout.strip() if result.returncode == 0 else None}


def prepare_run(config, run_dir, offline=False):
    # 延迟导入：离线重算报告不需要 tokenizer 或下载。
    from blackwell.tokenization import ContextTokenizer, TOKENIZER_REVISION, TOKENIZER_SHA256
    tokenizer = ContextTokenizer()
    examples, provenance = load_dataset(config, offline)
    identity = {"config": config, "dataset": provenance, "examples_sha256": fingerprint(examples),
                "tokenizer_revision": TOKENIZER_REVISION, "tokenizer_sha256": TOKENIZER_SHA256,
                "code": code_provenance()}
    local_methods = [m for m in selected_methods(config) if m in LOCAL_METHODS]
    if local_methods:
        from blackwell.compressors.runtime import configure_cache, runtime_identity, verify_assets
        configure_cache()
        verify_assets(local_methods)
        identity["local_compressors"] = runtime_identity(local_methods, config.get('compression_device', 'cpu'))
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        # git 提交与否不改变语义；代码内容和配置改变时必须换 run_name。
        old_identity = manifest["identity"]
        comparable_old = {**old_identity, "code": old_identity["code"]["files_sha256"]}
        comparable_new = {**identity, "code": identity["code"]["files_sha256"]}
        if comparable_old != comparable_new:
            raise ValueError("This run_name already belongs to different code/config/data. Choose a new run_name.")
    else:
        manifest = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared"}
        write_json(manifest_path, manifest)
        write_json(run_dir / "examples.json", examples)
    return tokenizer, examples, manifest


def generate_predictions(config, run_dir, tokenizer, examples, offline=False):
    client = CachedChatClient(config, run_dir, offline)
    methods = selected_methods(config)
    local_cache = None
    if any(m in LOCAL_METHODS for m in methods):
        from blackwell.local_cache import LocalCompressionCache
        identity = read_json(run_dir / "manifest.json")["identity"]["local_compressors"]
        local_cache = LocalCompressionCache(run_dir, identity, offline)
    compactor = Compactor(client, tokenizer, local_cache, config["retained_ratio"])
    rows = []
    for number, example in enumerate(examples, 1):
        context = example["context"]
        original_tokens = tokenizer.count(context)
        budget = max(1, math.floor(config["retained_ratio"] * original_tokens))
        # 所有压缩先完成，再让 reader 看到问题。compress() 不接收整个 example。
        compacted = {
            "full": {"text": context, "tokens": original_tokens, "budget_tokens": None,
                     "budget_met": None, "length_repair_used": False, "compression_request_hashes": []},
            "no_context": {"text": "", "tokens": 0, "budget_tokens": None,
                           "budget_met": None, "length_repair_used": False, "compression_request_hashes": []},
        }
        for method in methods:
            if method in {"full", "no_context"}:
                continue
            compacted[method] = compactor.compress(context, budget, method)
        conditions = {}
        for method in methods:
            compressed = compacted[method]
            raw, request_hash = client.complete(
                prediction_messages(compressed["text"], example["question"]),
                max_tokens=config["prediction_max_tokens"], json_mode=True,
            )
            probability = parse_probability(raw)
            conditions[method] = {
                **compressed, "p_yes": probability, "predicted_label": int(probability >= 0.5),
                "prediction_request_hash": request_hash,
                "retained_ratio_actual": compressed["tokens"] / original_tokens,
            }
        rows.append({**example, "original_tokens": original_tokens, "conditions": conditions})
        write_json(run_dir / "predictions.json", rows)
        print(f"[{number:02d}/{len(examples):02d}] {example['id']}  " +
              "  ".join(f"{m}={conditions[m]['p_yes']:.3f}" for m in methods), flush=True)
    return rows, {"new_api_attempts": client.new_attempts, "cache_hits": client.cache_hits}


def analyze(rows, config):
    if len(rows) != config["sample_size"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate samples: refusing to report a silently reduced dataset.")
    methods = selected_methods(config)
    if any(set(r["conditions"]) != set(methods) for r in rows):
        raise ValueError("Conditions must match the configured methods on every sample.")
    labels = [r["label"] for r in rows]
    groups = [r["context_group"] for r in rows]
    probabilities = {method: [r["conditions"][method]["p_yes"] for r in rows] for method in methods}
    scores = {}
    for method in methods:
        conditions = [r["conditions"][method] for r in rows]
        # 论文的多个问题复用同一压缩结果；耗时不能按题目重复累计。
        unique_compressions = {c['local_compression_hash']: c['compression_seconds']
                               for c in conditions if 'local_compression_hash' in c}
        scores[method] = {
            **forecast_scores(probabilities[method], labels),
            "mean_context_tokens": sum(c["tokens"] for c in conditions) / len(rows),
            "pooled_retained_ratio": sum(c["tokens"] for c in conditions) / sum(r["original_tokens"] for r in rows),
            "over_budget_samples": sum(c["budget_met"] is False for c in conditions),
            "length_repairs": sum(c["length_repair_used"] for c in conditions),
            "unique_compressions": len(unique_compressions),
            "compression_seconds_unique_total": sum(unique_compressions.values()),
        }
    for method in methods:
        for metric in ('accuracy', 'brier', 'log_loss'):
            scores[method][metric + '_delta_vs_full'] = scores[method][metric] - scores['full'][metric]
    matrix = {a: {b: infogap(probabilities[a], probabilities[b], labels) for b in methods} for a in methods}
    uncertainty = {method: bootstrap_gaps(
        probabilities["full"], probabilities[method], labels, groups,
        config["bootstrap_repetitions"], config["bootstrap_seed"],
    ) for method in methods if method != "full"}
    curves = {m: payoff_curve(probabilities['full'], probabilities[m], labels)
              for m in methods if m != 'full'}
    subsets = {}
    for task in sorted({r['source_task'] for r in rows if 'source_task' in r}):
        indices = [i for i, r in enumerate(rows) if r.get('source_task') == task]
        y = [labels[i] for i in indices]
        p = {m: [probabilities[m][i] for i in indices] for m in methods}
        subsets[task] = {'n': len(indices), 'positive_labels': sum(y),
                         'scores': {m: forecast_scores(p[m], y) for m in methods},
                         'infogap_vs_full': {m: {'full_over_method': infogap(p['full'], p[m], y),
                                                'method_over_full': infogap(p[m], p['full'], y)}
                                             for m in methods if m != 'full'}}
    return {"n": len(rows), "positive_labels": sum(labels), "scores": scores,
            "context_clusters": len(set(groups)), "payoff_curves_vs_full": curves, "source_tasks": subsets,
            "infogap": matrix, "bootstrap_vs_full": uncertainty,
            "normalization": "2 * sup_t mean[(t-y)(1[p_A<=t]-1[p_B<=t])]; also save unscaled.",
            "scope": "Small pilot of verbalized probabilities, not true-posterior recovery or a dominance test."}


def write_outputs(run_dir, rows, config):
    from blackwell.reporting import write_report
    methods = selected_methods(config)
    metrics = analyze(rows, config)
    write_json(run_dir / "metrics.json", metrics)
    fields = ["id", "label", "method", "p_yes", "p_no", "predicted_label", "context_tokens", "budget_tokens", "retained_ratio", "budget_met"]
    with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in methods:
                c = row["conditions"][method]
                writer.writerow({"id": row["id"], "label": row["label"], "method": method,
                                 "p_yes": c["p_yes"], "p_no": 1 - c["p_yes"], "predicted_label": c["predicted_label"],
                                 "context_tokens": c["tokens"], "budget_tokens": c["budget_tokens"],
                                 "retained_ratio": c["retained_ratio_actual"], "budget_met": c["budget_met"]})
    write_report(run_dir, rows, metrics, config)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="Download dataset/tokenizer, but do not call the LLM.")
    modes.add_argument("--offline", action="store_true", help="Replay exactly the saved API responses; fail on cache misses.")
    modes.add_argument("--report-only", action="store_true", help="Recompute metrics/report from saved predictions, without network.")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    run_dir = ROOT / "runs" / config["run_name"]
    if args.report_only:
        saved_config = read_json(run_dir / "manifest.json")["identity"]["config"]
        if saved_config != config:
            raise ValueError("Report config differs from the saved experiment config.")
        write_outputs(run_dir, read_json(run_dir / "predictions.json"), config)
        print(f"Report: {run_dir / 'report.html'}")
        return
    if args.offline:
        # 离线重放必须已准备好数据，不允许触发 download() 的网络分支。
        required = [ROOT / "data" / "tokenizer" / "qwen3-30b-a3b-instruct-2507.json",
                    run_dir / "manifest.json"]
        if not all(p.exists() for p in required):
            raise RuntimeError("Offline replay needs the previously prepared local files.")
    tokenizer, examples, manifest = prepare_run(config, run_dir, args.offline)
    print(f"{len(examples)} {manifest['identity']['dataset']['name']} {config['dataset_split']} samples; {len(selected_methods(config))} paired conditions.")
    if args.prepare_only:
        print("Prepared. No LLM API calls were made.")
        return
    import_api_cache(config, run_dir, args.offline)
    load_local_keys()
    manifest.update(status="running")
    write_json(run_dir / "manifest.json", manifest)
    try:
        rows, execution = generate_predictions(config, run_dir, tokenizer, examples, args.offline)
        write_outputs(run_dir, rows, config)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), last_execution=execution)
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(execution))
        print(f"Report: {run_dir / 'report.html'}")
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        write_json(run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        sys.exit(1)
