"""Local adapter training, held-out evaluation and checkpoint-matched lens readouts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dataset(text, seed=42):
    rows = []
    seen = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict): raise ValueError('expected an object')
            if 'messages' in row:
                messages = row['messages']
                if not isinstance(messages, list) or len(messages) < 2: raise ValueError('need at least user and assistant messages')
                if any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages): raise ValueError('messages need supported roles and nonempty text')
                if messages[-1]['role'] != 'assistant' or not any(m['role'] == 'user' for m in messages[:-1]): raise ValueError('last message must be an assistant answer to a user')
            else:
                if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('prompt','completion')): raise ValueError('need nonempty prompt and completion')
                messages = [{'role':'user','content':row['prompt']},{'role':'assistant','content':row['completion']}]
            # Group by the full input: conflicting/repeated answers must not leak across splits.
            key = json.dumps(messages[:-1], sort_keys=True, ensure_ascii=False)
            if key in seen: raise ValueError('duplicate input; remove repeated or conflicting examples')
            seen.add(key)
            rows.append({'messages': messages})
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'Line {number}: {error}') from error
    if len(rows) < 6: raise ValueError('Provide at least 6 distinct examples for training, validation and testing.')
    if len(rows) > 10000: raise ValueError('This local workflow supports at most 10,000 examples.')
    random.Random(seed).shuffle(rows)
    count = max(1, len(rows)//10)
    return {'train':rows[2*count:], 'valid':rows[count:2*count], 'test':rows[:count]}


def identity(model_path):
    root = Path(model_path).expanduser().resolve(strict=True)
    if (root/'.weights'/'config.json').is_file(): root=(root/'.weights').resolve()
    if not root.is_dir() or not (root/'config.json').is_file(): raise ValueError('Choose a local Transformers/MLX model folder containing config.json, tokenizer and safetensors weights. GGUF is not a training format here.')
    files = sorted(p for p in root.iterdir() if p.is_file() and (p.suffix in ('.safetensors', '.jinja') or p.name in ('config.json','tokenizer.json','tokenizer_config.json')))
    if not any(p.suffix == '.safetensors' for p in files): raise ValueError('No safetensors model weights found.')
    return {'path':str(root), 'sha256': hashlib.sha256(json.dumps([(p.name,digest(p)) for p in files]).encode()).hexdigest()}


def emit(kind, **fields):
    print('HB_LAB:' + json.dumps({'kind':kind, **fields}, default=str), flush=True)


def render(tokenizer, messages, answer=False):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not answer)


def parameters(config):
    for key, low, high in [('steps',1,10000),('max_length',32,2048),('rank',1,64),('layers',1,32),('seed',0,2147483647),('max_tokens',1,256)]:
        value=config.get(key, {'steps':100,'max_length':256,'rank':8,'layers':4,'seed':42,'max_tokens':64}[key])
        if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high: raise ValueError(f'{key} must be between {low} and {high}')
        config[key]=value
    rate=config.get('learning_rate',1e-4)
    if not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 1e-7<=rate<=0.01: raise ValueError('Learning rate must be between 0.0000001 and 0.01')
    config['learning_rate']=rate
    if config.get('backend') not in ('mlx','torch'): raise ValueError('Select MLX or PyTorch')
    return config


def check_lengths(tokenizer, splits, limit):
    lengths=[]
    for split, rows in splits.items():
        for i,row in enumerate(rows):
            size=len(tokenizer.encode(render(tokenizer,row['messages'],True),add_special_tokens=False))
            if size>limit: raise ValueError(f'{split} example {i+1} has {size} tokens, exceeding {limit}. Shorten it or raise the sequence limit; nothing was silently truncated.')
            lengths.append(size)
    return max(lengths)


def checkpoint(source, backend):
    from safetensors import safe_open
    folder=Path(source)/'adapter'
    candidates=sorted(folder.glob('*_adapters.safetensors'),reverse=True) if backend=='mlx' else []
    candidates += [folder/('adapters.safetensors' if backend=='mlx' else 'adapter_model.safetensors')]
    for path in candidates:
        try:
            with safe_open(path,framework='numpy') as weights:
                if not weights.keys():continue
                for key in weights.keys(): weights.get_tensor(key)
            return path
        except Exception:
            # A stopped write may be incomplete; try an earlier complete checkpoint.
            continue
    raise ValueError('No readable adapter checkpoint is available in this run.')


def train_mlx(config, root, splits):
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm import load
    import types
    model, tokenizer = load(config['model_path'])
    check_lengths(tokenizer,splits,config['max_length'])
    num_layers=min(config['layers'],len(model.layers))

    class Progress(TrainingCallback):
        def on_train_loss_report(self, info): emit('metric', **info)
        def on_val_loss_report(self, info): emit('validation', **info)
    args=dict(CONFIG_DEFAULTS)
    args.update(model=config['model_path'],data=str(root/'data'),train=True,iters=config['steps'],batch_size=1,num_layers=num_layers,max_seq_length=config['max_length'],learning_rate=config['learning_rate'],seed=config['seed'],adapter_path=str(root/'adapter'),steps_per_report=1,steps_per_eval=max(1,min(10,config['steps'])),val_batches=-1,save_every=max(1,min(10,config['steps'])),mask_prompt=True,grad_checkpoint=True,lora_parameters={'rank':config['rank'],'scale':float(config['rank']*2),'dropout':0.0},report_to=None)
    if config.get('resume_path'): args['resume_adapter_file']=str(checkpoint(config['resume_path'],'mlx'))
    args=types.SimpleNamespace(**args)
    train_set,valid_set,_=load_dataset(args,tokenizer)
    train_model(args,model,train_set,valid_set,training_callback=Progress())


def torch_model(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device),tokenizer


def train_torch(config,root,splits):
    import torch
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.manual_seed(config['seed'])
    model, tokenizer=torch_model(config['model_path'])
    check_lengths(tokenizer,splits,config['max_length'])
    if config.get('resume_path'):
        checkpoint(config['resume_path'],'torch')
        model=PeftModel.from_pretrained(model,str(Path(config['resume_path'])/'adapter'),is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=config['rank'],lora_alpha=config['rank']*2,target_modules='all-linear',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['learning_rate'])
    def loss(row):
        messages=row['messages']; prompt=render(tokenizer,messages[:-1]); full=render(tokenizer,messages,True)
        prefix=tokenizer.encode(prompt,add_special_tokens=False); ids=tokenizer.encode(full,add_special_tokens=False)
        if ids[:len(prefix)]!=prefix: raise ValueError('This chat template does not support safe response masking.')
        inputs=torch.tensor([ids],device=model.device);labels=inputs.clone();labels[:,:len(prefix)]=-100
        if (labels!=-100).sum()==0:raise ValueError('Example has no answer tokens.')
        return model(input_ids=inputs,labels=labels).loss
    randomizer=random.Random(config['seed']); rows=list(splits['train']); randomizer.shuffle(rows)
    for step in range(config['steps']):
        model.train(); value=loss(rows[step%len(rows)])
        if not torch.isfinite(value):raise ValueError('Training produced non-finite loss. Lower the learning rate.')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad()
        emit('metric',iteration=step+1,train_loss=float(value.detach()))
        if (step+1)%10==0 or step+1==config['steps']:
            model.eval()
            with torch.no_grad(): validation=sum(float(loss(row)) for row in splits['valid'])/len(splits['valid'])
            emit('validation',iteration=step+1,val_loss=validation)
            model.save_pretrained(root/'adapter',safe_serialization=True)
    tokenizer.save_pretrained(root/'adapter')


def train(config,root):
    config=parameters(config)
    splits=dataset(config['dataset'],config['seed'])
    emit('phase',message='Verifying model identity and dataset')
    model_id=identity(config['model_path'])
    config['model_path']=model_id['path']
    if config.get('resume_path'):
        previous=json.loads((Path(config['resume_path'])/'manifest.json').read_text())
        if previous['model']['sha256']!=model_id['sha256'] or previous['backend']!=config['backend']:raise ValueError('Adapter belongs to a different base model or backend.')
        if any(previous['settings'].get(k)!=config[k] for k in ('rank','layers')):raise ValueError('Use the original adapter rank and layer count when continuing.')
        checkpoint(config['resume_path'],config['backend'])
        if previous['dataset_sha256']!=hashlib.sha256(config['dataset'].encode()).hexdigest() or previous['seed']!=config['seed']:raise ValueError('Continue training with the original dataset and seed to preserve held-out splits.')
    data=root/'data';data.mkdir(exist_ok=True)
    for split,rows in splits.items(): (data/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    manifest={'schema':1,'model':model_id,'backend':config['backend'],'seed':config['seed'],'dataset_sha256':hashlib.sha256(config['dataset'].encode()).hexdigest(),'split_counts':{k:len(v) for k,v in splits.items()},'settings':{k:v for k,v in config.items() if k!='dataset'},'status':'training','continuation':'Adapter weights only; optimizer restarts.' if config.get('resume_path') else None}
    save(root/'manifest.json',manifest)
    emit('phase',message='Training adapter; original model stays unchanged')
    (train_mlx if config['backend']=='mlx' else train_torch)(config,root,splits)
    manifest['status']='trained';save(root/'manifest.json',manifest)
    return {'adapter_path':str(root/'adapter'),'manifest':manifest,'message':'Adapter saved. Run held-out evaluation before judging improvement.'}


def evaluate(config,root):
    import gc
    source=Path(config['run_path']).resolve(strict=True)
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['status']!='trained': raise ValueError('Choose a completed training run.')
    model_path=manifest['model']['path']
    if identity(model_path)['sha256']!=manifest['model']['sha256']:raise ValueError('Base weights changed since training.')
    rows=[json.loads(line) for line in (source/'data/test.jsonl').read_text().splitlines()]
    max_tokens=int(config.get('max_tokens',64))
    if not 1<=max_tokens<=256:raise ValueError('Evaluation output must be 1–256 tokens.')
    results=[{'prompt':r['messages'][:-1],'expected':r['messages'][-1]['content']} for r in rows]
    for variant in ('base','adapter'):
        emit('phase',message=f'Evaluating {variant} on {len(rows)} held-out examples')
        if manifest['backend']=='mlx':
            from mlx_lm import load,generate
            from mlx_lm.sample_utils import make_sampler
            model,tokenizer=load(model_path,adapter_path=str(source/'adapter') if variant=='adapter' else None)
        else:
            import torch
            model,tokenizer=torch_model(model_path)
            if variant=='adapter':
                from peft import PeftModel
                model=PeftModel.from_pretrained(model,str(source/'adapter'))
            model.eval()
        for index,row in enumerate(rows):
            prompt=render(tokenizer,row['messages'][:-1]);started=time.perf_counter()
            if manifest['backend']=='mlx': output=generate(model,tokenizer,prompt=prompt,max_tokens=max_tokens,sampler=make_sampler(temp=0),verbose=False)
            else:
                inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
                with torch.no_grad(): tokens=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                output=tokenizer.decode(tokens[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
            results[index][variant]={'text':output,'exact_match':output.strip()==row['messages'][-1]['content'].strip(),'seconds':time.perf_counter()-started}
            emit('evaluation',variant=variant,done=index+1,total=len(rows))
        del model;gc.collect()
        if manifest['backend']=='mlx':
            import mlx.core as mx
            mx.clear_cache()
    return {'training_run':str(source),'rows':results,'scores':{v:sum(r[v]['exact_match'] for r in results)/len(results) for v in ('base','adapter')},'metric':'Exact match after trimming whitespace; not a general quality score.','max_tokens':max_tokens}


def inspect(config,root):
    import torch
    import jlens
    model_id=identity(config['model_path']);model_path=model_id['path']
    model,tokenizer=torch_model(model_path)
    model.eval(); wrapped=jlens.from_hf(model,tokenizer)
    prompt=config.get('prompt','')
    if not prompt.strip():raise ValueError('Enter an inspection prompt.')
    ids=tokenizer.encode(prompt,add_special_tokens=False)
    if len(ids)>64:raise ValueError('Inspection is limited to 64 tokens; shorten the prompt.')
    lens_path=config.get('lens_path')
    lens=None
    if lens_path:
        metadata=json.loads(Path(lens_path+'.json').read_text())
        if metadata['model_sha256']!=model_id['sha256']: raise ValueError('Lens does not match these model weights. Fit a matching lens first.')
        if digest(lens_path)!=metadata['lens_sha256']:raise ValueError('Lens file checksum does not match its manifest.')
        lens=jlens.JacobianLens.load(lens_path)
    if config.get('action')=='fit_lens':
        prompts=[p.strip() for p in config.get('calibration','').splitlines() if p.strip()]
        if not 2<=len(prompts)<=100:raise ValueError('Provide 2–100 calibration prompts, one per line. Small fits are exploratory.')
        if any(len(tokenizer.encode(p))>64 for p in prompts):raise ValueError('Calibration prompts must be at most 64 tokens.')
        layers=sorted(set([0,min(wrapped.n_layers-2,max(1,wrapped.n_layers//2))]))
        emit('phase',message='Fitting a model-specific lens. This requires backward passes and can be slow.')
        lens=jlens.fit(wrapped,prompts,source_layers=layers,max_seq_len=64,skip_first=0,checkpoint_path=str(root/'fit-checkpoint.pt'))
        lens_path=str(root/'lens.pt');lens.save(lens_path)
        save(Path(lens_path+'.json'),{'model_sha256':model_id['sha256'],'lens_sha256':digest(lens_path),'calibration_count':len(prompts),'layers':layers,'scope':'Exploratory two-layer lens; validate on independent prompts.'})
    if lens is None:
        layers=sorted(set([0,max(0,wrapped.n_layers//2-1),wrapped.n_layers-1]))
        lens=jlens.JacobianLens({},n_prompts=0,d_model=wrapped.d_model)
    else:layers=lens.source_layers
    # Keep only top-k scores in the UI, not full vocabulary/activation dumps.
    views={}
    adapter_run=config.get('inspection_run_path')
    if adapter_run:
        metadata=json.loads((Path(adapter_run)/'manifest.json').read_text())
        if metadata['backend']!='torch' or metadata['model']['sha256']!=model_id['sha256'] or metadata['status']!='trained':
            raise ValueError('Inspection adapter must be a completed PyTorch run for this exact base model.')
    for variant in (['base','adapter'] if adapter_run else ['base']):
        if variant=='adapter':
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,str(Path(adapter_run)/'adapter')).merge_and_unload()
            model.eval();wrapped=jlens.from_hf(model,tokenizer)
        for kind in (['ordinary','jacobian'] if lens_path else ['ordinary']):
            scores, final, input_ids=lens.apply(wrapped,prompt,layers=layers,max_seq_len=64,use_jacobian=kind=='jacobian')
            cells=[]
            for layer,values in scores.items():
                top, indices=values.topk(5,dim=-1)
                for position in range(len(indices)):
                    cells.append({'layer':layer,'position':position,'top':[{'token':tokenizer.decode([int(t)]),'score':float(v)} for t,v in zip(indices[position],top[position])]})
            views[f'{kind} · {variant}' if adapter_run else kind]=cells
    return {'prompt':prompt,'tokens':[tokenizer.decode([int(t)]) for t in input_ids.flatten()],'views':views,'lens_path':lens_path,'calibration_count':lens.n_prompts if lens_path else None,'model':model_id,'meaning':'Vocabulary associations, not thoughts, probabilities or causal explanations. Ordinary and Jacobian lenses use the same selected positions/layers.' + (' The base-model lens is transferred to the adapted model for comparison; its validity after fine-tuning is unverified.' if adapter_run and lens_path else '')}



def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    path=Path(args.config).resolve();config=json.loads(path.read_text());root=path.parent
    try:
        action=config['action']
        if action=='validate':
            splits=dataset(config['dataset'],int(config.get('seed',42)))
            result={'counts':{k:len(v) for k,v in splits.items()},'preview':splits['train'][:3],'message':'Response-only training; test examples remain excluded from training and validation.'}
        elif action=='train':result=train(config,root)
        elif action=='evaluate':result=evaluate(config,root)
        elif action in ('inspect','fit_lens'):result=inspect(config,root)
        else:raise ValueError('Unsupported Lab action')
        save(root/'result.json',result);emit('complete',message='Results saved to result.json')
    except Exception as error:
        save(root/'error.json',{'message':str(error)});emit('error',message=str(error));raise

if __name__=='__main__': main()
