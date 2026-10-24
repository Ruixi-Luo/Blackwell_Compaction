"""透明、顺序调用的 SiliconFlow 客户端：缓存请求/响应，但绝不保存密钥。"""
import json
import math
import os
import time
from http.client import HTTPException
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .storage import fingerprint, read_json, write_json
from .streaming import read_stream, verify_stream_cache


class OutputLimitError(ValueError):
    """响应因输出上限被截断；原始响应已缓存，但不可作为有效结果。"""
    def __init__(self, request_hash):
        super().__init__('API output reached its token ceiling; incomplete text is not accepted.')
        self.request_hash = request_hash


class CachedChatClient:
    def __init__(self, config, run_dir, offline=False):
        self.config = config
        self.run_dir = run_dir
        self.offline = offline
        self.cache_dir = run_dir / "api_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = run_dir / "api_attempts.json"
        self.ledger = read_json(self.ledger_path) if self.ledger_path.exists() else []
        self.last_call = 0.0
        self.new_attempts = 0
        self.cache_hits = 0

    def complete(self, messages, max_tokens, json_mode=False):
        payload = {
            "model": self.config["model"], "messages": messages,
            "temperature": self.config["temperature"], "max_tokens": max_tokens,
            "enable_thinking": self.config["enable_thinking"], "stream": self.config.get("stream", False),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        identity = {"base_url": self.config["base_url"], "payload": payload}
        request_hash = fingerprint(identity)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            self.cache_hits += 1
            record = read_json(cache_path)
            if record["request"] != identity:
                raise ValueError("Cache content does not match the requested input.")
            return self._extract(record["response"], request_hash)
        # 恢复运行可复用完全相同生成参数的旧非流式响应，保留其真实请求和哈希。
        # 新请求仍以 stream=True 单独记账，绝不把旧响应伪装成流式响应。
        if payload['stream']:
            legacy_identity = {**identity, 'payload': {**payload, 'stream': False}}
            legacy_hash = fingerprint(legacy_identity)
            legacy_path = self.cache_dir / f'{legacy_hash}.json'
            if legacy_path.exists():
                record = read_json(legacy_path)
                if record['request'] != legacy_identity:
                    raise ValueError('Legacy cache content does not match the requested input.')
                self.cache_hits += 1
                return self._extract(record['response'], legacy_hash)
        if self.offline:
            raise RuntimeError("Offline replay is missing a cached response. No API call was made.")
        key = os.environ.get(self.config["api_key_env"], "").strip()
        if not key:
            raise RuntimeError(f"Set the environment variable {self.config['api_key_env']} first.")

        # 只对网络错误、限流、服务端错误重试；参数/认证错误立即停止。
        for attempt in range(3):
            retry_delay = 2 ** (attempt + 1)
            if len(self.ledger) >= self.config["max_api_attempts"]:
                raise RuntimeError("Hard API-attempt limit reached; stopped before the next request.")
            wait = self.config["min_interval_seconds"] - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            entry = {
                "request_hash": request_hash, "status": "started",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "max_output_tokens": max_tokens,
            }
            # 先记账后发送，即使请求超时也计入调用上限。
            self.ledger.append(entry)
            write_json(self.ledger_path, self.ledger)
            self.new_attempts += 1
            request = Request(
                self.config["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            self.last_call = time.monotonic()
            try:
                with urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                    body = read_stream(response) if payload["stream"] else json.load(response)
                entry["status"] = "received"
                entry["usage"] = body.get("usage", {})
                write_json(self.ledger_path, self.ledger)
                write_json(cache_path, {
                    "request": identity, "response": body,
                    "received_utc": datetime.now(timezone.utc).isoformat(),
                })
                return self._extract(body, request_hash)
            except HTTPError as error:
                entry.update(status="http_error", http_status=error.code)
                write_json(self.ledger_path, self.ledger)
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    # 不输出响应正文/请求头，避免服务端错误信息意外包含凭证。
                    raise RuntimeError(f"API returned HTTP {error.code}; see api_attempts.json.") from None
                if error.code == 429:
                    # 免费端点的每分钟限额较低；不以密集重试加重限流。
                    retry_delay = 15 * (attempt + 1)
            except (URLError, TimeoutError, HTTPException, ConnectionError) as error:
                entry.update(status="network_error", error_type=type(error).__name__)
                write_json(self.ledger_path, self.ledger)
                if attempt == 2:
                    raise RuntimeError("API connection failed after three attempts.") from None
            time.sleep(retry_delay)
        raise RuntimeError("No API response.")

    @staticmethod
    def _extract(body, request_hash):
        if "_stream_chunks" in body:
            verify_stream_cache(body)
        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            raise OutputLimitError(request_hash)
        if choice.get("finish_reason") != "stop":
            raise ValueError("API output did not finish normally; truncated predictions are not accepted.")
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("API returned an empty response.")
        return content.strip(), request_hash


def parse_probability(text):
    """严格解析；绝不把失败样本自动填成 0.5，也不把字符串或硬标签当概率。"""
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"p_yes"}:
        raise ValueError('Expected exactly {"p_yes": number}.')
    value = obj["p_yes"]
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("p_yes must be a finite numeric probability in [0, 1].")
    return float(value)

