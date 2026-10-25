"""只处理模型请求、缓存和预算；不决定压缩算法或评价指标。"""
from blackwell.client import CachedChatClient, parse_probability
from blackwell.cost_guard import CostGuard, GuardedClient
from blackwell.prompts import prediction_messages
from .config import load_keys


def request_identity(config, messages, maximum, json_mode=False):
    """config：单个模型配置；messages：请求；maximum：输出上限；json_mode：JSON 回答。"""
    payload = {k: config[k] for k in ('model', 'temperature', 'enable_thinking', 'stream')}
    payload.update(messages=messages, max_tokens=maximum)
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    return {'base_url': config['base_url'], 'payload': payload}


def make_client(config, role, run, tokenizer, allow_api=False):
    """role=reader/summary；run：本轮目录；allow_api=False 时缓存缺失就报错。"""
    folder = run/'api'/role
    if not allow_api:
        return CachedChatClient(config[role], folder, offline=True)
    load_keys()
    guard = CostGuard(run/'spending_reservations.json', tokenizer, config['cost_reserve_cny'])
    return GuardedClient(config[role], folder, guard)


def predict_one(context, question, client, maximum=128):
    """预测器此时才接收 question；返回原始概率、分类和请求哈希。"""
    raw, key = client.complete(prediction_messages(context, question), maximum, json_mode=True)
    probability = parse_probability(raw)
    return {'p_yes': probability, 'predicted_label': int(probability >= .5),
            'prediction_request_hash': key}
