"""只加载 LLMLingua-2 tokenizer，核验原生分块不会丢失正文 token。"""
import argparse
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.compressors.runtime import configure_cache, model_directory
from blackwell.storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hotpotqa.json')
    args = parser.parse_args()
    config = read_json(ROOT/args.config)
    run = ROOT/'runs'/config['run_name']
    configure_cache()
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden')):
        from transformers import AutoTokenizer
        from llmlingua import PromptCompressor
        # 只调用官方分块方法，不初始化分类模型或重新压缩。
        engine = PromptCompressor.__new__(PromptCompressor)
        engine.tokenizer = AutoTokenizer.from_pretrained(str(model_directory('llmlingua2')), local_files_only=True)
        engine.max_seq_len = 512
        records = []
        for example in read_json(run/'examples.json'):
            chunks = engine._PromptCompressor__chunk_context(example['context'], {'.', '\n'})
            sizes = [len(engine.tokenizer.tokenize(chunk))+2 for chunk in chunks]
            # 官方分块偶尔保留 511 个正文 token。TokenClfDataset 在其前后加
            # CLS/SEP 再切到 512；长度 513 只丢 SEP，超过 513 才丢正文。
            if not sizes or max(sizes) > 513:
                raise ValueError(f"Native chunk loses content tokens: {example['id']}, lengths={sizes}")
            records.append({'id': example['id'], 'chunk_tokens_including_special_tokens': sizes})
    result = {'status': 'passed', 'network_blocked': True, 'classifier_weights_loaded': False,
              'max_chunk_tokens_including_special_tokens': max(max(r['chunk_tokens_including_special_tokens']) for r in records),
              'chunks_with_only_final_separator_trimmed': sum(n == 513 for r in records for n in r['chunk_tokens_including_special_tokens']),
              'content_tokens_truncated_by_classifier': 0,
              'records': records}
    write_json(run/'local_chunk_check.json', result)
    print('Verified classifier input lengths for',len(records),'contexts; max=',result['max_chunk_tokens_including_special_tokens'])


if __name__ == '__main__':
    main()
