"""Clean 命令行入口。先看 main()，再沿阶段名称读 experiment/pipeline.py。"""
import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from experiment.config import load_config, run_directory
from experiment import pipeline


@contextmanager
def run_lock(run):
    """run：本轮目录。独占锁防止两个终端同时写缓存/记账；正常退出自动删除。"""
    run.mkdir(parents=True, exist_ok=True)
    path = run/'.run.lock'
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError(f'Run is already locked: {path}. Check the recorded PID before removing a stale lock.') from None
    with handle:
        json.dump({'pid': os.getpid()}, handle)
    try:
        yield
    finally:
        path.unlink()


def main():
    """解析命令 -> 验证配置 -> 执行阶段。没有隐藏的后台队列。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/clean/mini100.json')
    parser.add_argument('--stage', choices=['prepare', 'compress', 'predict', 'evaluate', 'audit', 'report', 'all'], default='all')
    parser.add_argument('--offline', action='store_true', help='No downloads, GPU inference or API calls; fail if caches are missing.')
    parser.add_argument('--allow-api', action='store_true', help='Explicitly allow paid API cache misses within the configured reservation.')
    parser.add_argument('--import-run', type=Path, help='Explicit one-time import of an audited mini100 first-draft source; requires --offline.')
    args = parser.parse_args()
    if args.offline and args.allow_api:
        parser.error('--offline and --allow-api cannot be combined.')
    if args.import_run and not args.offline:
        parser.error('--import-run requires --offline; migration must never generate new outputs.')
    config = load_config(args.config)
    run = run_directory(config)
    with run_lock(run):
        tokenizer, manifest = pipeline.prepare(config, run, args.offline)
        if args.import_run:
            from experiment.import_results import import_firstdraft
            import_firstdraft(args.import_run, run, config, tokenizer)
        if args.stage in ('compress', 'all'):
            pipeline.compress(config, run, tokenizer, args.offline, args.allow_api)
        if args.stage in ('predict', 'all'):
            pipeline.predict(config, run, tokenizer, args.allow_api)
        if args.stage in ('evaluate', 'all'):
            pipeline.evaluate(config, run)
        if args.stage in ('audit', 'all'):
            pipeline.audit(config, run, tokenizer, manifest)
        if args.stage in ('report', 'all'):
            from experiment.reporting import build_report
            build_report(config, run)
    print('Output:', run)


if __name__ == '__main__':
    main()
