"""付费前检查全部输入的本地模型长度限制；只加载 tokenizer，不加载模型。"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from blackwell.compressors.runtime import configure_cache, model_directory
from blackwell.compressors.chunking import split_overlong
from blackwell.data import load_dataset
from blackwell.storage import read_json, write_json
from blackwell.tokenization import ContextTokenizer


def inspect(config_path):
    configure_cache()
    from transformers import AutoTokenizer, GPT2Tokenizer
    from llmlingua import PromptCompressor
    config = read_json(ROOT / config_path)
    examples, provenance = load_dataset(config, offline=True)
    counter = ContextTokenizer()
    sc = GPT2Tokenizer.from_pretrained(str(model_directory('selective_context')), local_files_only=True)
    engine = PromptCompressor.__new__(PromptCompressor)
    engine.tokenizer = AutoTokenizer.from_pretrained(str(model_directory('llmlingua2')), local_files_only=True)
    engine.max_seq_len = 512
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    records, seen = [], set()
    for example in examples:
        if example['context_group'] in seen:
            continue
        seen.add(example['context_group'])
        context = example['context']
        sentences = [s.strip() for s in re.split(pattern, re.sub(r'\s+', ' ', context)) if s.strip()]
        sizes_sc = [len(sc.encode('<|endoftext|>' + s, add_special_tokens=False)) for s in sentences]
        chunks = engine._PromptCompressor__chunk_context(context, {'.', '\n'})
        sizes_llm = [len(engine.tokenizer.tokenize(c)) + 2 for c in chunks]
        sc_count = lambda s: len(sc.encode('<|endoftext|>' + s.strip(), add_special_tokens=False))
        llm_count = lambda s: len(engine.tokenizer.tokenize(s))
        safe_sc = [part for s in sentences for part in split_overlong(s, sc_count, 1024)]
        safe_llm = [part for c in chunks for part in split_overlong(c, llm_count, 511)]
        if ''.join(safe_sc) != ''.join(sentences) or ''.join(safe_llm) != ''.join(chunks):
            raise ValueError('Length adapter lost text.')
        records.append({'id': example['id'], 'context_group': example['context_group'],
                        'qwen_tokens': counter.count(context), 'sc_max_sentence_tokens': max(sizes_sc),
                        'llmlingua2_max_chunk_tokens_with_special': max(sizes_llm),
                        'sc_overlong_sentences': sum(n > 1024 for n in sizes_sc),
                        'llmlingua2_overlong_chunks': sum(n > 513 for n in sizes_llm),
                        'sc_max_after_adapter': max(sc_count(s) for s in safe_sc),
                        'llmlingua2_max_after_adapter': max(llm_count(s)+2 for s in safe_llm)})
    problems = [r for r in records if r['sc_max_after_adapter'] > 1024 or r['llmlingua2_max_after_adapter'] > 513]
    result = {'status': 'failed' if problems else 'passed', 'questions': len(examples),
              'unique_contexts': len(records), 'provenance': provenance, 'problems': problems,
              'records': records, 'api_calls': 0, 'model_weights_loaded': False}
    result['contexts_with_length_adapter'] = sum(bool(r['sc_overlong_sentences'] or r['llmlingua2_overlong_chunks']) for r in records)
    write_json(ROOT / '.cache/preflight' / (Path(config_path).stem + '.json'), result)
    print(f"{config_path}: {len(examples)} questions, {len(records)} contexts, {len(problems)} length problems", flush=True)
    for problem in problems:
        print(problem, flush=True)
    return not problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    sys.exit(0 if inspect(parser.parse_args().config) else 1)
