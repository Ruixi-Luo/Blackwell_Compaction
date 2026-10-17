"""官方 selective-context 0.1.4 的薄封装：只替换固定版本的模型加载位置。"""
import re

from . import validate_input
from .runtime import model_directory, prepare_torch


class SelectiveContextCompressor:
    def __init__(self):
        prepare_torch()
        from selective_context import SelectiveContext
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        folder = str(model_directory("selective_context"))

        class PinnedSelectiveContext(SelectiveContext):
            def _prepare_model(self):
                # 保留官方句内 self-information、短语聚合与百分位筛选代码。
                # 覆盖此钩子，只为 CPU、safetensors、离线固定模型路径。
                self.device = "cpu"
                self.tokenizer = GPT2Tokenizer.from_pretrained(folder, local_files_only=True)
                self.model = GPT2LMHeadModel.from_pretrained(
                    folder, local_files_only=True, use_safetensors=True
                ).to(self.device).eval()
                self.max_token_length = self.model.config.n_positions
                self.get_self_information = self._get_self_info_via_gpt2

        self.engine = PinnedSelectiveContext(model_type="gpt2", lang="en")

    def compress(self, context, retained_ratio=0.5):
        """仅输入上下文和保留率，输出压缩文本；不接收 query。"""
        validate_input(context, retained_ratio)
        if not context.strip() or retained_ratio == 1:
            return context
        # 官方 GPT-2 按句计算，有 1024-token 位置上限。超长单句明确失败。
        clean = self.engine.beautify_context(context)
        for sentence in re.split(self.engine.sent_tokenize_pattern, clean):
            if sentence.strip():
                ids = self.engine.tokenizer.encode("<|endoftext|>" + sentence.strip(), add_special_tokens=False)
                if len(ids) > self.engine.max_token_length:
                    raise ValueError("Selective Context: a sentence exceeds GPT-2's 1024-token limit; no silent truncation.")
        # 注意：官方参数表示删去比例；本项目统一参数表示保留比例。
        text, _removed_phrases = self.engine(
            context, reduce_ratio=1 - retained_ratio, reduce_level="phrase"
        )
        return text
