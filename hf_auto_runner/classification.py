"""Task-specific text and image classification training, with frozen test splits."""
import hashlib
import json
from pathlib import Path
import random
from .lab import save, digest, identity, emit, parameters

TASKS = ('text-classification', 'image-classification')


def split_dataset(text, task, seed=42):
    if task not in TASKS: raise ValueError('Unsupported classification task.')
    groups = {}; seen = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict): raise ValueError('Expected an object')
            label = row.get('label')
            if not isinstance(label, str) or not label.strip(): raise ValueError('label must be nonempty text')
            label = label.strip()
            field = 'text' if task == 'text-classification' else 'image'
            value = row.get(field)
            if not isinstance(value, str) or not value.strip(): raise ValueError(f'{field} must be nonempty text')
            if field == 'image':
                path = Path(value).expanduser()
                if not path.is_absolute() or not path.is_file(): raise ValueError('image must be an existing absolute local path')
                if path.stat().st_size > 30_000_000: raise ValueError('Image exceeds 30 MB')
                value = str(path.resolve()); key = digest(path)
            else: value = value.strip(); key = ' '.join(value.split()).casefold()
            if key in seen: raise ValueError('Duplicate input, including duplicates with conflicting labels')
            seen.add(key)
            groups.setdefault(label, []).append({field:value,'label':label,**({'sha256':key} if field=='image' else {})})
        except (ValueError,TypeError,OSError) as error: raise ValueError(f'Line {number}: {error}') from error
    if not 2 <= len(groups) <= 100: raise ValueError('Provide 2–100 classes.')
    if len(seen) > 10000: raise ValueError('Maximum 10,000 examples.')
    splits = {'train':[], 'valid':[], 'test':[]}; rng = random.Random(seed)
    for label, rows in sorted(groups.items()):
        if len(rows) < 6: raise ValueError(f'Class {label!r} needs at least 6 distinct examples.')
        rng.shuffle(rows); n = max(1,len(rows)//10)
        splits['test'].extend(rows[:n]); splits['valid'].extend(rows[n:2*n]); splits['train'].extend(rows[2*n:])
    for rows in splits.values(): rng.shuffle(rows)
    return splits, sorted(groups)


def metrics(actual, predicted, labels):
    if not actual or len(actual)!=len(predicted): raise ValueError('Expected aligned, nonempty predictions.')
    matrix=[[0]*len(labels) for _ in labels]
    for truth, pred in zip(actual,predicted): matrix[labels.index(truth)][labels.index(pred)]+=1
    scores=[]
    for i,label in enumerate(labels):
        tp=matrix[i][i]; support=sum(matrix[i]); guessed=sum(row[i] for row in matrix)
        precision=tp/guessed if guessed else 0; recall=tp/support if support else 0
        scores.append({'label':label,'support':support,'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0})
    return {'accuracy':sum(matrix[i][i] for i in range(len(labels)))/len(actual),'macro_f1':sum(s['f1'] for s in scores)/len(scores),'labels':labels,'confusion_matrix':matrix,'per_class':scores}


def capabilities(model_path, task):
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForImageClassification
    source=identity(model_path)
    config=AutoConfig.from_pretrained(source['path'],local_files_only=True,trust_remote_code=False)
    cls=AutoModelForSequenceClassification if task=='text-classification' else AutoModelForImageClassification
    if type(config) not in cls._model_mapping: raise ValueError(f'{config.model_type} is not supported for {task}. Select a compatible checkpoint; runnable does not imply trainable.')
    if getattr(config,'quantization_config',None): raise ValueError('Classification training requires non-quantized model weights.')
    return source,cls


def image(row):
    from PIL import Image, ImageOps
    path=Path(row['image'])
    if digest(path)!=row['sha256']: raise ValueError(f'Image changed since validation: {path.name}')
    with Image.open(path) as im:
        if im.width*im.height>40_000_000: raise ValueError('Image exceeds 40 megapixels')
        return ImageOps.exif_transpose(im).convert('RGB')


def processor_for(path,task):
    from transformers import AutoTokenizer, AutoImageProcessor
    if task=='text-classification': return AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    return AutoImageProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=False)


def inputs_for(row,processor,task,limit,device):
    if task=='text-classification':
        values=processor(row['text'],return_tensors='pt',truncation=False)
        if values['input_ids'].shape[1]>limit: raise ValueError(f'Example exceeds {limit} tokens. Increase the limit or shorten it; no silent truncation.')
    else: values=processor(images=image(row),return_tensors='pt')
    return {key:value.to(device) for key,value in values.items()}


def train(config,root):
    import torch
    task=config['task']; parameters(config)
    if config['backend']!='torch': raise ValueError('Classification training uses PyTorch.')
    if config.get('resume_path'): raise ValueError('Classification continuation is not supported yet; start a new run.')
    method=config.get('training_method','head')
    if method not in ('head','full'): raise ValueError('Choose head-only or full fine-tuning.')
    splits,labels=split_dataset(config['dataset'],task,config['seed'])
    source,model_cls=capabilities(config['model_path'],task)
    torch.manual_seed(config['seed'])
    model=model_cls.from_pretrained(source['path'],local_files_only=True,trust_remote_code=False,num_labels=len(labels),id2label=dict(enumerate(labels)),label2id={v:i for i,v in enumerate(labels)},ignore_mismatched_sizes=True)
    model.config.problem_type="single_label_classification"
    backbone=model.base_model
    if backbone is model: raise ValueError('This architecture does not expose a separate classifier backbone.')
    backbone_parameters={id(p) for p in backbone.parameters()}
    # Always initialise a new task head; an existing classifier may use unrelated labels.
    heads=[module for module in model.modules() if any(id(p) not in backbone_parameters for p in module.parameters(recurse=False))]
    if not heads: raise ValueError('No trainable classification head found.')
    for module in heads: model._init_weights(module)
    for parameter in backbone.parameters(): parameter.requires_grad_(method=='full')
    device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    processor=processor_for(source['path'],task)
    emit('phase',message='Checking all examples before training')
    for rows in splits.values():
        for row in rows: inputs_for(row,processor,task,config['max_length'],device)
    data=root/'data';data.mkdir(exist_ok=True)
    for split,rows in splits.items():save(data/(split+'.json'),rows)
    manifest={'schema':1,'task':task,'backend':'torch','model':source,'labels':labels,'training_method':method,'status':'training','settings':{k:v for k,v in config.items() if k!='dataset'},'dataset_sha256':hashlib.sha256(config['dataset'].encode()).hexdigest(),'split_counts':{k:len(v) for k,v in splits.items()},'baseline':'Pretrained backbone with the same freshly initialised task head, before training.'}
    save(root/'manifest.json',manifest)
    model.save_pretrained(root/'baseline',safe_serialization=True);processor.save_pretrained(root/'baseline')
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config['learning_rate'])
    def loss(row):return model(**inputs_for(row,processor,task,config['max_length'],device),labels=torch.tensor([labels.index(row['label'])],device=device)).loss
    rows=list(splits['train']);rng=random.Random(config['seed']);rng.shuffle(rows)
    for step in range(config['steps']):
        model.train()
        if method=='head':backbone.eval()
        value=loss(rows[step%len(rows)])
        if not torch.isfinite(value):raise ValueError('Non-finite loss; lower the learning rate.')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad()
        emit('metric',iteration=step+1,train_loss=float(value.detach()))
        if (step+1)%10==0 or step+1==config['steps']:
            model.eval()
            with torch.no_grad(): val=sum(float(loss(row)) for row in splits['valid'])/len(splits['valid'])
            emit('validation',iteration=step+1,val_loss=val)
    model.save_pretrained(root/'model',safe_serialization=True);processor.save_pretrained(root/'model')
    manifest['status']='trained';save(root/'manifest.json',manifest)
    return {'task':task,'model_path':str(root/'model'),'manifest':manifest,'message':'Classifier saved with label mapping and preprocessing. Evaluate the held-out set before using it.'}


def evaluate(config,root):
    import torch
    source=Path(config['run_path']);manifest=json.loads((source/'manifest.json').read_text());task=manifest['task']
    if manifest['status']!='trained':raise ValueError('Select a completed classification run.')
    labels=manifest['labels'];rows=json.loads((source/'data/test.json').read_text());result=[];stats={}
    _,cls=capabilities(str(source/'model'),task)
    for variant,folder in [('base','baseline'),('adapter','model')]:
        emit('phase',message=f'Evaluating {folder} on the fixed held-out set')
        model=cls.from_pretrained(source/folder,local_files_only=True,trust_remote_code=False).eval()
        device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device);processor=processor_for(str(source/folder),task)
        predicted=[]
        for index,row in enumerate(rows):
            with torch.no_grad(): logits=model(**inputs_for(row,processor,task,manifest['settings']['max_length'],device)).logits
            prediction=labels[int(logits.argmax(-1).item())];predicted.append(prediction)
            if variant=='base': result.append({'prompt':[{'role':'input','content':row.get('text',row.get('image'))}],'expected':row['label']})
            result[index][variant]={'text':prediction}
        stats[variant]=metrics([r['label'] for r in rows],predicted,labels)
        del model
    return {'task':task,'rows':result,'scores':{v:s['accuracy'] for v,s in stats.items()},'classification_metrics':stats,'metric':'Accuracy and macro F1 on a stratified held-out set. Baseline uses the same new task head before training; it is not an existing pretrained label mapping.'}
