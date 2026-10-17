"""两个 baseline；公共接口只接收上下文和长度预算。"""
from .prompts import compression_messages


class Compactor:
    def __init__(self, client, tokenizer):
        self.client = client
        self.tokenizer = tokenizer

    def compress(self, context, budget, method):
        """故意不接收 question、label 或原文预测，确保 query-blind。"""
        if method == "head_tail":
            text = self.tokenizer.head_tail(context, budget)
            return self._result(text, budget, [], False)
        if method != "summary":
            raise ValueError(f"Unknown compaction method: {method}")
        hashes = []
        previous = None
        previous_tokens = None
        for attempt in range(2):  # 最多一次长度修复；不根据预测效果挑摘要。
            messages = compression_messages(context, budget, previous, previous_tokens)
            text, request_hash = self.client.complete(
                messages, max_tokens=min(2048, max(256, budget * 3 + 100))
            )
            hashes.append(request_hash)
            count = self.tokenizer.count(text)
            if count <= budget:
                return self._result(text, budget, hashes, attempt == 1)
            previous, previous_tokens = text, count
        # 仍然超预算就明确标记并保留原输出，不偷偷追加截断策略。
        return self._result(previous, budget, hashes, True)

    def _result(self, text, budget, hashes, repaired):
        count = self.tokenizer.count(text)
        return {
            "text": text, "tokens": count, "budget_tokens": budget,
            "budget_met": count <= budget, "length_repair_used": repaired,
            "compression_request_hashes": hashes,
        }

