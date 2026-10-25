"""Mini study: prepare -> DAC -> summary revision -> prediction -> audit.

All compressors receive only source text and its ex-ante budget. Old runs are
read-only. Every output is resumable, and all reader requests use exact caching.
"""
import argparse, concurrent.futures, json, math, os, random, shutil, sys, time
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from blackwell.storage import read_json, write_json, fingerprint, file_hash
from blackwell.tokenization import ContextTokenizer
from blackwell.client import CachedChatClient, parse_probability, OutputLimitError
from blackwell.prompts import prediction_messages
from blackwell.fixed_budget import in_band, choose_candidate, revision_messages
from blackwell.cost_guard import CostGuard, GuardedClient
from blackwell.diagnostics import probability_diagnostics, retention_diagnostics
from run_experiment import load_local_keys, analyze

METHODS=['full','no_context','selective_context','llmlingua2','llm_summary','dac']
SOURCES={'hotpotqa':'aligned50_hotpotqa_20261024_v2','qasper':'aligned50_qasper_20261024_v2'}
SUITE=ROOT/'runs/mini100_fixed50_20261025_v2'

def sources_index():
    return {p.stem:p for p in (ROOT/'runs').glob('*/api_cache/*.json') if SUITE not in p.parents}

def payload(config,messages,maximum,json_mode=False):
    p={'model':config['model'],'messages':messages,'temperature':0.,'max_tokens':maximum,
       'enable_thinking':False,'stream':config.get('stream',False)}
    if json_mode: p['response_format']={'type':'json_object'}
    return {'base_url':config['base_url'],'payload':p}

def core_hashes():
    paths=[Path(__file__),ROOT/'run_experiment.py',ROOT/'configs/mini100_fixed50_v2.json',
           ROOT/'configs/dac.lock.json',*sorted((ROOT/'blackwell').rglob('*.py'))]
    return {p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}

def entry_path(folder,context,method):
    return folder/'compressions'/f'{fingerprint({"context":context,"method":method})}.json'

def save_entry(folder,context,method,budget,candidates):
    selected=choose_candidate(candidates,budget)
    result={'text':selected['text'],'tokens':selected['tokens'],'budget_tokens':budget,
        'budget_met':True,'within_target_band':in_band(selected['tokens'],budget),
        'budget_fill':selected['tokens']/budget,'post_clip_used':False,'matching_trim_used':False,
        'length_repair_used':len(candidates)>1,'compression_request_hashes':[
            c['request_hash'] for c in candidates if 'request_hash' in c],
        'selected_candidate':candidates.index(selected),'compression_seconds':sum(c.get('seconds',0) for c in candidates)}
    request={'context':context,'budget_tokens':budget,'method':method}
    entry={'request':request,'candidates':candidates,'result':result,'result_sha256':fingerprint(result)}
    write_json(entry_path(folder,context,method),entry)
    return result

def prepare(config):
    tokenizer=ContextTokenizer()
    if (SUITE/'manifest.json').exists():
        manifest=read_json(SUITE/'manifest.json')
        if manifest['config']!=config or manifest['core_hashes']!=core_hashes():
            raise ValueError('Frozen configuration or code changed: use a new run identity.')
        return tokenizer,manifest
    SUITE.mkdir(parents=True,exist_ok=True)
    old_files={}
    for p in (ROOT/'runs').glob('*/*'):
        if p.is_file() and p.suffix in {'.json','.csv','.html'} and SUITE not in p.parents:
            old_files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    manifest={'config':config,'core_hashes':core_hashes(),'old_files_sha256':old_files,
        'created_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared',
        'scope':'Same predefined 50% cap, target 95-100% fill. No post-clipping or minimum-output matching.',
        'source_runs':SOURCES,'datasets':{}}
    for name,source_name in SOURCES.items():
        source=ROOT/'runs'/source_name; folder=SUITE/name; folder.mkdir(exist_ok=True)
        old=read_json(source/'predictions.json')
        selected=random.Random(config['sample_seed']).sample(sorted(old,key=lambda r:r['id']),100)
        examples=[{k:v for k,v in r.items() if k not in {'conditions','original_tokens'}} for r in selected]
        write_json(folder/'examples.json',examples)
        write_json(folder/'config.json',{**read_json(source/'manifest.json')['identity']['config'],
            **config,'dataset':name,'run_name':SUITE.name+'_'+name,'sample_size':100})
        contexts={r['context'] for r in examples}
        manifest['datasets'][name]={'n':100,'yes':sum(e['label'] for e in examples),
            'unique_contexts':len(contexts),'examples_sha256':fingerprint(examples),
            'source_dataset':read_json(source/'manifest.json')['identity']['dataset']}
        imported={}
        for p in (source/'aligned_compression_cache').glob('*.json'):
            entry=read_json(p); req=entry['request']; method=req['method']
            if req['context'] not in contexts or method not in {'selective_context','llmlingua2','llm_summary'}: continue
            if entry['sha256']!=fingerprint(entry['result']): raise ValueError('Invalid source cache')
            candidate_list=[]
            for c in entry['result']['candidates']:
                candidate_list.append({**c,'origin_file':p.relative_to(ROOT).as_posix(),'origin_sha256':file_hash(p)})
            context=req['context']; budget=math.floor(.5*tokenizer.count(context))
            if method=='llm_summary':
                draftfile=entry_path(folder,context,method).with_suffix('.drafts.json')
                write_json(draftfile,candidate_list)
                # Every initial complete draft is retained, but only an in-band one is final now.
                if any(in_band(c['tokens'],budget) for c in candidate_list):
                    save_entry(folder,context,method,budget,candidate_list)
            else:
                if not any(in_band(c['tokens'],budget) for c in candidate_list):
                    raise ValueError('Unexpected local budget miss; do not silently clip.')
                save_entry(folder,context,method,budget,candidate_list)
            imported[str(p.relative_to(ROOT))]=file_hash(p)
        write_json(folder/'compression_reuse.json',imported)
    write_json(SUITE/'manifest.json',manifest)
    print('Prepared independent 100+100 samples and native compression caches.',flush=True)
    return tokenizer,manifest

def run_dac(tokenizer,config):
    from blackwell.compressors.dac import DACCompressor
    preflight=read_json(ROOT/'.cache/mini100_dac_preflight.json')
    if preflight['status']!='passed' or not preflight['full_context_processed']:
        raise ValueError('Complete-context DAC preflight required.')
    shutil.copy2(ROOT/'.cache/mini100_dac_preflight.json',SUITE/'dac_preflight.json')
    engine=None
    for name in SOURCES:
        folder=SUITE/name; contexts=list(dict.fromkeys(r['context'] for r in read_json(folder/'examples.json')))
        for index,context in enumerate(contexts,1):
            if entry_path(folder,context,'dac').exists(): continue
            if engine is None: engine=DACCompressor()
            budget=math.floor(.5*tokenizer.count(context)); target=.975*budget; rate=.4875; candidates=[]
            for attempt in range(4):
                start=time.perf_counter(); text=engine.compress(context,retained_ratio=rate)
                count=tokenizer.count(text)
                candidates.append({'text':text,'tokens':count,'native_retained_ratio':rate,
                    'seconds':time.perf_counter()-start,'native_metadata':engine.last_metadata})
                if in_band(count,budget): break
                rate=min(.999,max(.001,rate*target/max(1,count)))
            result=save_entry(folder,context,'dac',budget,candidates)
            print(f'DAC {name} {index}/{len(contexts)} {result["tokens"]}/{budget} band={result["within_target_band"]}',flush=True)
    del engine
    import gc,torch
    gc.collect(); torch.cuda.empty_cache()

def revise_worker(worker,tasks,tokenizer,config,guard):
    folder=SUITE/'summary_api'/f'worker_{worker}'
    api={**config,'model':config['compression_model'],'stream':True,
         'timeout_seconds':1800,'max_api_attempts':360,'min_interval_seconds':0.}
    client=GuardedClient(api,folder,guard)
    next_start=0.
    for dataset,context in tasks:
        output=SUITE/dataset; path=entry_path(output,context,'llm_summary')
        if path.exists() and read_json(path)['result']['within_target_band']: continue
        budget=math.floor(.5*tokenizer.count(context)); draftpath=path.with_suffix('.drafts.json')
        candidates=read_json(draftpath)
        for revision in range(sum('revision' in c for c in candidates),config['summary_new_revisions']):
            previous=candidates[-1]
            messages=revision_messages(context,budget,previous['text'],previous['tokens'])
            maximum=min(32768,max(512,2*budget+512))
            wait=next_start-time.monotonic()
            if wait>0: time.sleep(wait)
            next_start=time.monotonic()+60*(sum(tokenizer.count(m['content']) for m in messages)+maximum+100)/(35000/config['workers'])
            start=time.perf_counter()
            try:
                text,key=client.complete(messages,maximum)
            except OutputLimitError:
                # Do not spend a third hidden generation or accept partial text.
                raise
            candidate={'text':text,'tokens':tokenizer.count(text),'request_hash':key,
                'revision':revision+1,'max_tokens':maximum,'api_folder':folder.relative_to(ROOT).as_posix(),
                'seconds':time.perf_counter()-start}
            candidates.append(candidate); write_json(draftpath,candidates)
            print(f'Summary {dataset} worker={worker} draft={revision+1} {candidate["tokens"]}/{budget}',flush=True)
            if in_band(candidate['tokens'],budget): break
        result=save_entry(output,context,'llm_summary',budget,candidates)
        if not result['within_target_band']:
            print(f'BUDGET_UNDERFILL {dataset}: {result["tokens"]}/{budget}; explicitly retained, never padded/clipped.',flush=True)

def run_summary(tokenizer,config,guard,smoke=False):
    tasks=[]
    for name in SOURCES:
        tasks.extend((name,c) for c in dict.fromkeys(r['context'] for r in read_json(SUITE/name/'examples.json')))
    if smoke:
        tasks=[task for dataset in SOURCES for task in [t for t in tasks if t[0]==dataset][:2]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(revise_worker,w,tasks[w::config['workers']],tokenizer,config,guard) for w in range(config['workers'])]
        for future in futures: future.result()

def read_condition(folder,context,method,tokenizer):
    if method in {'full','no_context'}:
        text=context if method=='full' else ''
        return {'text':text,'tokens':tokenizer.count(text),'budget_tokens':None,'budget_met':None,
                'length_repair_used':False,'compression_request_hashes':[]}
    entry=read_json(entry_path(folder,context,method))
    if entry['result_sha256']!=fingerprint(entry['result']): raise ValueError('Compression hash mismatch')
    return {**entry['result'],'local_compression_hash':fingerprint(entry['request'])}

def run_predictions(tokenizer,config,guard,offline=False):
    index=sources_index() if not offline else {}
    for name in SOURCES:
        folder=SUITE/name; examples=read_json(folder/'examples.json')
        cfg={**config,'stream':False,'timeout_seconds':90,'max_api_attempts':800,'min_interval_seconds':2.}
        client=CachedChatClient(cfg,folder,offline=True) if offline else GuardedClient(cfg,folder,guard)
        receipts=read_json(folder/'reader_reuse.json') if (folder/'reader_reuse.json').exists() else {}
        rows=[]
        for i,e in enumerate(examples,1):
            conditions={}; original=tokenizer.count(e['context'])
            for method in METHODS:
                c=read_condition(folder,e['context'],method,tokenizer)
                messages=prediction_messages(c['text'],e['question'])
                identity=payload(cfg,messages,128,True); key=fingerprint(identity)
                dest=folder/'api_cache'/f'{key}.json'
                if not offline and not dest.exists() and key in index:
                    source=index[key]; record=read_json(source)
                    if record['request']!=identity: raise ValueError('Reader cache identity mismatch')
                    parse_probability(CachedChatClient._extract(record['response'],key)[0])
                    dest.parent.mkdir(exist_ok=True); shutil.copy2(source,dest)
                    receipts[key]={'source':source.relative_to(ROOT).as_posix(),'sha256':file_hash(source)}
                raw,key=client.complete(messages,128,True); p=parse_probability(raw)
                conditions[method]={**c,'p_yes':p,'predicted_label':int(p>=.5),
                    'prediction_request_hash':key,'retained_ratio_actual':c['tokens']/original,
                    'reader_reused':key in receipts}
            rows.append({**e,'original_tokens':original,'conditions':conditions})
            if not offline:
                write_json(folder/'reader_reuse.json',receipts); write_json(folder/'predictions.json',rows)
                print(f'Predict {name} {i}/100',flush=True)
        if not offline:
            metrics=analyze(rows,{**config,'sample_size':100}); write_json(folder/'metrics.json',metrics)
            diagnostics={m:probability_diagnostics([r['conditions'][m]['p_yes'] for r in rows],[r['label'] for r in rows]) for m in METHODS}
            write_json(folder/'diagnostics.json',diagnostics)
        else:
            if rows!=read_json(folder/'predictions.json'): raise ValueError('Offline predictions differ')

def audit(tokenizer,config,manifest):
    from tools.extend_with_salt import check_cached_prediction, check_matrix
    for path,digest in manifest['old_files_sha256'].items():
        if file_hash(ROOT/path)!=digest: raise ValueError('Old output changed: '+path)
    for name in SOURCES:
        folder=SUITE/name; rows=read_json(folder/'predictions.json'); examples=read_json(folder/'examples.json')
        if len(rows)!=100 or fingerprint(examples)!=manifest['datasets'][name]['examples_sha256']: raise ValueError('Sample mismatch')
        checked=set(); underfill={m:0 for m in METHODS[2:]}
        for row,e in zip(rows,examples):
            if any(row[k]!=v for k,v in e.items()) or list(row['conditions'])!=METHODS: raise ValueError('Pairing mismatch')
            for m,c in row['conditions'].items():
                check_cached_prediction(folder,config,row,c)
                if tokenizer.count(c['text'])!=c['tokens']: raise ValueError('Length mismatch')
                if m in {'full','no_context'}:
                    if c['text']!=(row['context'] if m=='full' else ''): raise ValueError('Control changed')
                    continue
                b=math.floor(.5*row['original_tokens'])
                if c['budget_tokens']!=b or c['tokens']>b or c['post_clip_used'] or c['matching_trim_used']: raise ValueError('Budget protocol violation')
                underfill[m]+=not in_band(c['tokens'],b)
                path=entry_path(folder,row['context'],m)
                if path in checked: continue
                checked.add(path); entry=read_json(path)
                if entry['request']!={'context':row['context'],'budget_tokens':b,'method':m}: raise ValueError('Unexpected compressor input fields')
                selected=choose_candidate(entry['candidates'],b)
                if selected['text']!=c['text']: raise ValueError('Candidate selection mismatch')
                for j,candidate in enumerate(entry['candidates']):
                    if tokenizer.count(candidate['text'])!=candidate['tokens']: raise ValueError('Candidate token count mismatch')
                    if 'origin_file' in candidate:
                        if file_hash(ROOT/candidate['origin_file'])!=candidate['origin_sha256']: raise ValueError('Old compression changed')
                        original_entry=read_json(ROOT/candidate['origin_file'])
                        if original_entry['request']['context']!=row['context'] or not any(old['text']==candidate['text'] for old in original_entry['result']['candidates']): raise ValueError('Old candidate provenance mismatch')
                    if 'revision' in candidate:
                        previous=entry['candidates'][j-1]
                        messages=revision_messages(row['context'],b,previous['text'],previous['tokens'])
                        cfg={**config,'model':config['compression_model'],'stream':True}
                        identity=payload(cfg,messages,candidate['max_tokens'])
                        if fingerprint(identity)!=candidate['request_hash']: raise ValueError('Summary request mismatch')
                        record=read_json(ROOT/candidate['api_folder']/'api_cache'/f"{candidate['request_hash']}.json")
                        if record['request']!=identity or CachedChatClient._extract(record['response'],candidate['request_hash'])[0]!=candidate['text']: raise ValueError('Summary response mismatch')
        metrics=read_json(folder/'metrics.json'); check_matrix(rows,metrics,METHODS)
        if metrics!=analyze(rows,{**config,'sample_size':100}): raise ValueError('Metrics replay differs')
        write_json(folder/'audit.json',{'status':'passed','n':100,'forecasts':600,'matrix_cells':36,
            'query_blind_inputs':True,'fixed_cap_passed':True,'no_post_truncation':True,
            'underfilled_question_counts':underfill,'all_in_target_band':not any(underfill.values()),
            'old_outputs_unchanged':True,'reconstructed_requests':True})
    run_predictions(tokenizer,config,None,offline=True)
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(SUITE/'manifest.json',manifest)
    print('Both datasets audited; all 1200 forecasts replay offline.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','dac','summary','summary-smoke','predict','audit','all'],default='all')
    args=parser.parse_args(); config=read_json(ROOT/'configs/mini100_fixed50_v2.json')
    tokenizer,manifest=prepare(config); load_local_keys()
    guard=CostGuard(SUITE/'spending_reservations.json',tokenizer,config['cost_reserve_cny'])
    if args.stage in {'dac','all'}: run_dac(tokenizer,config)
    if args.stage in {'summary','summary-smoke','all'}: run_summary(tokenizer,config,guard,args.stage=='summary-smoke')
    if args.stage in {'predict','all'}: run_predictions(tokenizer,config,guard)
    if args.stage in {'audit','all'}: audit(tokenizer,config,manifest)

if __name__=='__main__': main()
