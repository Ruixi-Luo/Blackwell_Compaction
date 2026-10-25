"""配置和路径：唯一需要日常修改的实验设置来自 JSON。"""
import math
import importlib.metadata
import os
import re
from blackwell.storage import ROOT, read_json, file_hash

METHODS = ('full', 'no_context', 'selective_context', 'llmlingua2', 'llm_summary', 'dac')


def load_config(path):
    """path：配置文件路径；返回已检查的 dict，不下载、不调用模型。"""
    config = read_json(path)
    if not re.fullmatch(r'clean_[A-Za-z0-9_-]+', config['run_name']):
        raise ValueError('run_name must start with clean_ and contain only letters, digits, _ or -.')
    methods = config['methods']
    if not isinstance(methods, list) or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('Unknown or repeated method.')
    if not {'full', 'no_context'} <= set(methods):
        raise ValueError('Keep Full and No context as controls.')
    for key in ('sample_size', 'sample_seed', 'bootstrap_repetitions', 'bootstrap_seed'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError('Expected a nonnegative integer: '+key)
    if config['sample_size'] < 1 or config['bootstrap_repetitions'] < 2:
        raise ValueError('Invalid sample size or bootstrap repetitions.')
    if type(config['retained_ratio']) not in (float, int) or not 0 < config['retained_ratio'] < 1:
        raise ValueError('retained_ratio is the fraction to KEEP, between zero and one.')
    if config['summary_policy'] != 'original_first_draft':
        raise ValueError('Clean workflow only implements original_first_draft, without revisions or clipping.')
    if config['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda.')
    if not config['datasets'] or set(config['datasets']) - {'hotpotqa', 'qasper'}:
        raise ValueError('Clean workflow supports HotpotQA and QASPER.')
    if len(config['datasets']) != len(set(config['datasets'])):
        raise ValueError('Repeated dataset.')
    if not math.isfinite(config['cost_reserve_cny']) or config['cost_reserve_cny'] <= 0:
        raise ValueError('Positive cost reservation required.')
    for role in ('reader', 'summary'):
        api = config[role]
        if api['temperature'] != 0 or api['enable_thinking'] is not False:
            raise ValueError('This protocol fixes temperature=0 and thinking=false.')
        if type(api['max_api_attempts']) is not int or api['max_api_attempts'] < 1:
            raise ValueError('API attempt limit must be a positive integer.')
        if api['timeout_seconds'] <= 0 or api['min_interval_seconds'] < 0:
            raise ValueError('Invalid request timing.')
        # Price accounting and token counting are currently validated for these models only.
        expected = 'Qwen/Qwen3-30B-A3B-Instruct-2507' if role == 'reader' else 'deepseek-ai/DeepSeek-V3.2'
        if api['model'] != expected or api['base_url'] != 'https://api.siliconflow.cn/v1':
            raise ValueError('A new model/provider needs a tokenizer and pricing review first.')
    if config['reader']['stream'] or config['reader']['max_tokens'] != 128:
        raise ValueError('Reader protocol is non-streaming, max_tokens=128.')
    return config


def run_directory(config):
    """config：已验证配置；每个 run_name 对应独立目录。"""
    return ROOT / 'runs' / config['run_name']


def code_hashes():
    """冻结计算代码和模型锁文件；报告样式变化不影响已保存预测。"""
    paths = [ROOT/'run.py', *sorted((ROOT/'experiment').glob('*.py')),
             *sorted((ROOT/'blackwell').rglob('*.py')),
             ROOT/'configs/local_models.lock.json', ROOT/'configs/dac.lock.json']
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths
            if p != ROOT/'experiment/reporting.py'}


def load_keys():
    """只读本机 .env，保留已设置环境变量，不打印密钥。"""
    path = ROOT/'.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.partition('=')
            if sep and name.strip() in ('SILICONFLOW_API_KEY', 'OPENAI_API_KEY'):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def package_versions():
    """记录实际已安装的依赖版本；只读元数据，不加载 GPU。"""
    versions = {}
    for name in ('tokenizers', 'pyarrow', 'torch', 'transformers', 'llmlingua',
                 'selective-context', 'spacy', 'numpy', 'accelerate'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
