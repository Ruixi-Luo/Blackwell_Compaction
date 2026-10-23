"""官方 SALT legacy 的无 query 纯文本接口；只调用固定版本的 prose 路径。"""
import os
import sys
from pathlib import Path

from . import validate_input
from .runtime import ROOT, asset_spec, model_directory, prepare_torch, verify_assets


# 用官方 CLI 的 prose 默认值，而不是 retrieval.py 内不同的函数默认值。
SALT_SETTINGS = {
    "selector": "legacy", "query": None, "dataset_name": "",
    "max_keywords_ratio": 0.4, "theme_percentile": 0.9,
    "min_keywords": 2, "pass1_budget_pct": 0.5,
    "theme_prec_min": 0.0, "redundancy_thresh": 1.0, "intro_pct": 0.03,
    "neighbor_window": 0, "query_budget_uncapped": False,
    "query_budget_pct": 0.75, "branch_floor": 0,
    "attention_implementation": "eager", "python_hash_seed": 0,
    "budget_basis": "floor(retained_ratio * words after native clean/split/dedup, before filtering)",
    "preprocessing": "native prose clean/split/aggressive filter; no dataset-specific anchors or few-shot bypass",
    "long_sentence_policy": "native split target 400 BGE tokens; native encoder truncation to 512 incl specials; selected sentence text kept",
    "output_policy": "selected sentences in document order, joined with spaces; no post-truncation",
}


class SaltSummaryCompressor:
    def __init__(self, device="cpu"):
        # 官方 set 的并列主题选择受 Python hash seed 影响，须在进程启动前固定。
        if os.environ.get("PYTHONHASHSEED") != "0" or sys.flags.hash_randomization:
            raise RuntimeError("Start Python with PYTHONHASHSEED=0 for reproducible SALT selection.")
        torch = prepare_torch(device)
        verify_assets(["salt_summary"])
        source = ROOT / asset_spec("salt_summary")["directory"] / "source"
        sys.path.insert(0, str(source))
        from salt.engine import compressor
        from salt.engine.retrieval import trie_select
        if not compressor.__file__ or not Path(compressor.__file__).resolve().is_relative_to(source.resolve()):
            raise RuntimeError("SALT was imported from a different installation, not the pinned source.")
        from transformers import AutoModel, AutoTokenizer
        folder = str(model_directory("salt_summary"))
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            output_attentions=True, attn_implementation="eager", torch_dtype=torch.float32,
        ).to(device).eval()
        self.pipeline = compressor
        self.select = trie_select
        self.device = device

    def compress(self, context, retained_ratio=0.5):
        """输入 context 字符串和保留率，输出字符串；不接收问题、标签或数据集。"""
        validate_input(context, retained_ratio)
        if not context.strip() or retained_ratio == 1:
            return context
        # dataset_name 恒为空：没有借用数据集名选择规则，更没有读取 input/query。
        sentences, _, _, word_budget, _ = self.pipeline.prep_prose_sentences(
            context, "", self.tokenizer, retained_ratio)
        if not sentences:
            return ""  # 忠实保留官方清洗后为空的结果，不偷偷换成截断或其他方法。
        sent_data, frequencies, themes = self.pipeline.build_prose_sent_data(
            sentences, self.tokenizer, self.model, self.device,
            SALT_SETTINGS["max_keywords_ratio"], SALT_SETTINGS["theme_percentile"])
        if sum(sentence["n_words"] for sentence in sent_data) <= word_budget:
            return " ".join(sentence["text"] for sentence in sent_data)
        selected, _ = self.select(
            sent_data, frequencies, themes, word_budget,
            query_keywords=None, query_embedding=None, query_proper_nouns=None, qtype=None,
            min_keywords=2, pass1_budget_pct=0.5, theme_prec_min=0.0,
            redundancy_thresh=1.0, intro_pct=0.03, neighbor_window=0,
            query_budget_uncapped=False, query_budget_pct=0.75, branch_floor=0,
        )
        return " ".join(sentence.text for sentence in sorted(selected, key=lambda sentence: sentence.sent_idx))
