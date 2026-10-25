"""DAC official selection, with full-context attention reduced in memory-bounded blocks.

Query blocks attend to ALL preceding source tokens. This is not document chunking,
windowed attention, or truncation. Scores and hidden states use float32.
"""
import importlib.util
import math
import types
from pathlib import Path

from ..storage import ROOT, read_json
from .runtime import prepare_torch, file_digest


class DACCompressor:
    def __init__(self, device='cuda', query_block=128, logit_block=64):
        self.torch = prepare_torch(device)
        spec = read_json(ROOT/'configs/dac.lock.json')
        folder = ROOT/spec['directory']
        for name, record in spec['files'].items():
            if file_digest(folder/name) != record['sha256']:
                raise ValueError('DAC asset hash mismatch: '+name)
        module_spec = importlib.util.spec_from_file_location('pinned_dac', folder/'source/compressor.py')
        upstream = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(upstream)
        self.engine = upstream.PromptCompressor(str(folder/'model'), device_map=device,
            model_config={'local_files_only': True, 'trust_remote_code': False, 'torch_dtype': self.torch.float32})
        self.query_block, self.logit_block = query_block, logit_block
        self.original_get_ppl = self.engine.get_ppl
        self.original_forwards = [layer.self_attn.forward for layer in self.engine.model.model.layers]
        self.column_sums = []
        self.enable_bounded()

    def enable_bounded(self):
        # SDPA mask preparation omits the unnecessary L x L causal mask. Actual
        # attention below uses explicit causal masks for only query_block rows.
        self.engine.model.config._attn_implementation = 'sdpa'
        for layer in self.engine.model.model.layers:
            layer.self_attn.forward = types.MethodType(self.attention_forward, layer.self_attn)
        self.engine.get_ppl = self.get_ppl

    def enable_reference(self):
        """Only for short-input equivalence checks against unmodified official code."""
        self.engine.model.config._attn_implementation = 'eager'
        for layer, forward in zip(self.engine.model.model.layers, self.original_forwards):
            layer.self_attn.forward = forward
        self.engine.get_ppl = self.original_get_ppl

    def attention_forward(self, attention, hidden_states, position_embeddings,
                          attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
        torch = self.torch
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        if past_key_values is not None:
            raise ValueError('DAC scoring must not use a KV cache.')
        shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
        q = attention.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = attention.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k = repeat_kv(k, attention.num_key_value_groups)
        v = repeat_kv(v, attention.num_key_value_groups)
        length = q.shape[-2]
        output = torch.empty_like(q)
        columns = torch.zeros(length, device=q.device, dtype=torch.float32)
        for start in range(0, length, self.query_block):
            end = min(start+self.query_block, length)
            scores = torch.matmul(q[:,:,start:end], k.transpose(2,3))*attention.scaling
            causal = torch.arange(length,device=q.device)[None,:] > torch.arange(start,end,device=q.device)[:,None]
            scores.masked_fill_(causal, torch.finfo(scores.dtype).min)
            if attention_mask is not None:
                scores += attention_mask[:,:,start:end,:length]
            weights = torch.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
            output[:,:,start:end] = torch.matmul(weights,v)
            columns += weights.sum(dim=(0,1,2))
        if self.collect_attention:
            self.column_sums.append(columns)
        output = output.transpose(1,2).contiguous().reshape(*hidden_states.shape[:-1],-1)
        return attention.o_proj(output), None

    def get_ppl(self, context='', input_ids=None, attention_mask=None, return_attn=False):
        torch = self.torch
        self.column_sums, self.collect_attention = [], return_attn
        if input_ids is None:
            encoded = self.engine.tokenizer(context,return_tensors='pt',add_special_tokens=False)
            input_ids = encoded['input_ids'].to(self.engine.device)
        else:
            input_ids = input_ids.to(self.engine.device)
        mask = torch.ones_like(input_ids)
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError('This adapter supports a single unpadded context only.')
        if input_ids.shape[1] > self.engine.max_position_embeddings:
            raise ValueError('Source exceeds DAC model window; no truncation allowed.')
        with torch.inference_mode():
            hidden = self.engine.model.model(input_ids=input_ids,attention_mask=None,
                use_cache=False,output_attentions=False,return_dict=True).last_hidden_state
            losses=[]
            for start in range(0,input_ids.shape[1]-1,self.logit_block):
                end=min(start+self.logit_block,input_ids.shape[1]-1)
                logits=self.engine.model.lm_head(hidden[:,start:end]).float()
                losses.append(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                    input_ids[:,start+1:end+1].reshape(-1),reduction='none'))
            loss=torch.cat(losses).reshape(1,-1)
            if return_attn:
                # Preserve official implementation's normalization exactly;
                # its matrix counter is (layers * query heads - 1).
                denominator=len(self.column_sums)*self.engine.model.config.num_attention_heads-1
                sums=torch.stack(self.column_sums).sum(dim=0)[1:]/denominator
                return loss,input_ids,mask,sums
            return loss,input_ids,mask

    def compress(self, context, retained_ratio=.5):
        """Only source text and retention ratio; never a question or answer."""
        if not isinstance(context,str) or not context.strip() or not 0<retained_ratio<1:
            raise ValueError('Expected nonempty source and a retention ratio in (0,1).')
        result=self.engine.compress(context=context,compress_ratio=1-retained_ratio,
            method='dynamic_attn_ppl',fusion='additive',alpha=.8,dyn_time=10,
            preserve_punct=False,return_info=True)
        self.last_metadata={k:v for k,v in result.items() if k not in {'compressed_text','kept_indices'}}
        return result['compressed_text']
