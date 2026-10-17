"""用 Qwen 官方 tokenizer 计数；只需要约 11 MB 的词表文件。"""
from tokenizers import Tokenizer

from .storage import ROOT, download

TOKENIZER_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
TOKENIZER_URL = f"https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/resolve/{TOKENIZER_REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


class ContextTokenizer:
    def __init__(self):
        self.path = download(
            TOKENIZER_URL, ROOT / "data" / "tokenizer" / "qwen3-30b-a3b-instruct-2507.json", TOKENIZER_SHA256
        )
        self.tokenizer = Tokenizer.from_file(str(self.path))

    def count(self, text):
        # 不包括 system prompt、问题或 chat template；API usage 另行记录。
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def head_tail(self, text, budget):
        """按 token 分配首尾额度，但在字符边界截取，避免切坏 Unicode。"""
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded.ids) <= budget:
            return text
        left = (budget + 1) // 2
        right = budget // 2
        while left or right:
            prefix = text[:encoded.offsets[left - 1][1]] if left else ""
            suffix = text[encoded.offsets[-right][0]:] if right else ""
            candidate = prefix + ("\n[...]\n" if left and right else "") + suffix
            # 截断标记也占预算；拼接后重新计数，而非假设 token 数可相加。
            if self.count(candidate) <= budget:
                return candidate
            if left >= right:
                left -= 1
            else:
                right -= 1
        return ""

