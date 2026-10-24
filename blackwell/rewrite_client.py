"""给独立改写 worker 节流；基础 API 客户端保持串行、简单且无需锁。"""
import time

from .client import CachedChatClient


class PacedRewriteClient:
    def __init__(self, config, directory, tokenizer, workers, worker):
        # 每个 worker 有自己的账本；额度之和恰好等于总上限。
        total = config['max_api_attempts']
        allowance = total // workers + int(worker < total % workers)
        self.client = CachedChatClient({**config, 'max_api_attempts': allowance}, directory)
        self.tokenizer = tokenizer
        self.tokens_per_minute = config.get('compression_tokens_per_minute', 35000) / workers
        self.next_start = 0.

    def complete(self, messages, max_tokens):
        """按输入 token 估计＋最大输出额度保守安排下一次调用；不改变 API 请求。"""
        wait = self.next_start-time.monotonic()
        if wait > 0:
            time.sleep(wait)
        estimated = sum(self.tokenizer.count(m['content']) for m in messages) + max_tokens + 100
        self.next_start = time.monotonic() + 60*estimated/self.tokens_per_minute
        return self.client.complete(messages, max_tokens)
