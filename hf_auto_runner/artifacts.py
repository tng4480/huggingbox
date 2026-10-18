"""Resolve one runtime's files into a managed, resumable local model cache."""
import argparse
import json
from pathlib import Path, PurePosixPath

MARKER = '.huggingbox_complete'


def safe_path(root, name):
    part = PurePosixPath(name)
    if not name or part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name:
        raise ValueError(f'Invalid model file path: {name}')
    root = Path(root).resolve()
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Model file escapes cache: {name}')
    return path


def select_files(names, runtime, gguf=None):
    names = sorted(set(names))
    if runtime in ('llama_cpp', 'diffusers_gguf'):
        if not gguf or gguf not in names or not gguf.endswith('.gguf'):
            raise ValueError('Choose an available GGUF file before preparation.')
        # A split GGUF must be downloaded as a complete shard set.
        import re
        match = re.match(r'(.+)-\d{5}-of-(\d{5})\.gguf$', gguf)
        weights = [n for n in names if match and re.match(re.escape(match[1])+r'-\d{5}-of-'+match[2]+r'\.gguf$', n)] if match else [gguf]
        return weights
    common = [n for n in names if n.endswith(('.json', '.txt', '.model', '.tiktoken', '.jinja', '.py', '.md')) and not n.startswith(('onnx/', 'openvino/', '.'))]
    if runtime == 'onnxruntime':
        weights = [n for n in names if n.endswith(('.onnx', '.onnx_data', '.ort')) or n.startswith('onnx/') and n.endswith(('.json', '.txt', '.model'))]
    else:
        eligible = [n for n in names if not n.startswith(('onnx/', 'openvino/', '.'))]
        weights = [n for n in eligible if n.endswith('.safetensors') and '.fp16.' not in n]
        if not weights:
            weights = [n for n in eligible if n.endswith('.safetensors')]
        if not weights:
            weights = [n for n in eligible if n.endswith(('.bin', '.pt', '.pth'))]
    if not weights:
        raise ValueError(f'No supported weights found for {runtime}.')
    return sorted(set(common + weights))


def is_ready(directory, runtime=None, gguf=None):
    root = Path(directory)
    try:
        marker = json.loads((root / MARKER).read_text())
        if marker.get('schema') != 1 or (runtime and marker['runtime'] != runtime) or (gguf and marker.get('gguf') != gguf):
            return False
        return bool(marker['files']) and all(safe_path(root, f['path']).is_file() and safe_path(root, f['path']).stat().st_size == f['size'] for f in marker['files'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(model_id, directory, runtime, gguf=None, token=None):
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=token).model_info(model_id, files_metadata=True)
    names = [s.rfilename for s in info.siblings]
    selected = select_files(names, runtime, gguf)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    total = sum(sizes[n] for n in selected)
    revision = info.sha
    (root / MARKER).unlink(missing_ok=True)
    files = []
    for index, name in enumerate(selected):
        safe_path(root, name)
        downloaded = hf_hub_download(model_id, name, revision=revision, token=token, local_dir=str(root))
        files.append({'path': name, 'size': Path(downloaded).stat().st_size})
        done = sum(f['size'] for f in files)
        print('HB_DOWNLOAD_JSON:' + json.dumps({'percent': 100*(index+1)/len(selected), 'downloaded_bytes': done, 'total_bytes': total, 'speed_bps': 0, 'eta_seconds': None, 'phase': 'downloading', 'files_done': index+1, 'files_total': len(selected), 'filename': name}), flush=True)
    metadata = {'id': model_id, 'modelId': model_id, 'pipeline_tag': info.pipeline_tag, 'tags': info.tags or [], 'downloads': info.downloads or 0, 'author': model_id.split('/')[0], 'siblings': [{'rfilename': s.rfilename, 'size': s.size or 0} for s in info.siblings]}
    (root / 'huggingbox-model.json').write_text(json.dumps(metadata))
    record = {'schema': 1, 'model_id': model_id, 'revision': revision, 'runtime': runtime, 'gguf': gguf, 'files': files}
    temporary = root / (MARKER + '.tmp')
    temporary.write_text(json.dumps(record))
    temporary.replace(root / MARKER)


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument('model_id'); p.add_argument('directory'); p.add_argument('runtime'); p.add_argument('--gguf')
    args = p.parse_args()
    prepare(args.model_id, args.directory, args.runtime, args.gguf, os.environ.get('HF_TOKEN') or None)

if __name__ == '__main__':
    main()
