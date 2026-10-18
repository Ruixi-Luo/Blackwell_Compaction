"""数据入口：按配置选 loader，后续压缩、预测和评估使用相同数据格式。"""
from .datasets.boolq import load_boolq
from .datasets.hotpotqa import load_hotpotqa
from .datasets.qasper import load_qasper
from .datasets.longbench import load_longbench


def load_dataset(config, offline=False):
    # 旧 BoolQ 配置没有 dataset 字段，保留原行为。
    name = config.get("dataset", "boolq")
    loaders = {"boolq": load_boolq, "hotpotqa_yesno": load_hotpotqa,
               "qasper_yesno_text": load_qasper, "longbench_yesno": load_longbench}
    if name not in loaders:
        raise ValueError(f"Unsupported dataset: {name}.")
    return loaders[name](config["dataset_split"], config["sample_size"], config["sample_seed"], offline)
