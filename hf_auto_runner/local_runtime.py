"""Shared local-artifact contract for generated scripts and warm workers."""
import os
from pathlib import Path

_loaded = {}


def model_source(model_id):
    directory = os.environ.get('HB_MODEL_DIR')
    if directory:
        if not Path(directory).is_dir():
            raise FileNotFoundError('Model cache is missing. Prepare the model again.')
        return directory
    return model_id


def model_file(model_id, filename, token=None):
    from .artifacts import safe_path
    directory = os.environ.get('HB_MODEL_DIR')
    if directory:
        path = safe_path(directory, filename)
        if not path.is_file():
            raise FileNotFoundError(f'Model file is missing: {filename}. Prepare the model again.')
        return str(path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=model_id, filename=filename, token=token)


def load_once(factory, *args, **kwargs):
    # Workers are keyed by script, model and device; standalone runs keep their normal lifetime.
    key = (factory, repr(args), repr(sorted(kwargs.items())))
    if key not in _loaded:
        _loaded[key] = factory(*args, **kwargs)
    else:
        print('[HuggingBox] Reusing loaded model component.', flush=True)
    return _loaded[key]


def clear_loaded():
    _loaded.clear()
