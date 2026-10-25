"""Shared conservative spending reservations; exact provider charges remain authoritative."""
import threading
from datetime import datetime, timezone
from .storage import read_json, write_json
from .client import CachedChatClient

class CostGuard:
    def __init__(self, path, tokenizer, limit=20):
        self.path, self.tokenizer, self.limit = path, tokenizer, limit
        self.lock=threading.Lock()
        self.items=read_json(path) if path.exists() else []
        # A crash leaves reservations outstanding. Do not assume they were free.

    def reserve(self, model, messages, maximum):
        deepseek='DeepSeek' in model
        inputs=sum(self.tokenizer.count(m['content']) for m in messages)+200
        # Provider tokenizer may differ. Input multiplier and price margins are explicit.
        estimated_input=inputs*(1.5 if deepseek else 1.1)
        unit=(estimated_input*(4 if deepseek else 1)+maximum*(6 if deepseek else 3))/1e6
        amount=3*unit  # base client has at most three transport attempts
        with self.lock:
            used=sum(x['accounted_cny'] for x in self.items)
            if used+amount>self.limit:
                raise RuntimeError(f'Cost reservation would exceed {self.limit} CNY: {used:.2f}+{amount:.2f}')
            index=len(self.items)
            self.items.append({'model':model,'status':'reserved','accounted_cny':amount,
                'unknown_usage_per_attempt_cny':unit,'created_utc':datetime.now(timezone.utc).isoformat()})
            write_json(self.path,self.items)
            return index

    def settle(self, index, attempts):
        with self.lock:
            item=self.items[index]; total=0; unknown=0
            for a in attempts:
                u=a.get('usage')
                if not u:
                    unknown+=1; total+=item['unknown_usage_per_attempt_cny']; continue
                hit=u.get('prompt_cache_hit_tokens',u.get('prompt_tokens_details',{}).get('cached_tokens',0)) or 0
                if 'DeepSeek' in item['model']:
                    total+=((u['prompt_tokens']-hit)*4+hit*.4+u['completion_tokens']*6)/1e6
                else:
                    total+=(u['prompt_tokens']*1+u['completion_tokens']*3)/1e6
            item.update(status='settled',accounted_cny=total,unknown_usage_attempts=unknown,
                        attempts=len(attempts))
            write_json(self.path,self.items)

class GuardedClient:
    def __init__(self, config, folder, guard):
        self.client=CachedChatClient(config,folder)
        self.offline=CachedChatClient(config,folder,offline=True)
        self.guard=guard

    def complete(self, messages, max_tokens, json_mode=False):
        try:
            return self.offline.complete(messages,max_tokens,json_mode)
        except RuntimeError as error:
            if 'Offline replay is missing' not in str(error): raise
        reservation=self.guard.reserve(self.client.config['model'],messages,max_tokens)
        before=len(self.client.ledger)
        try:
            return self.client.complete(messages,max_tokens,json_mode)
        finally:
            self.guard.settle(reservation,self.client.ledger[before:])
