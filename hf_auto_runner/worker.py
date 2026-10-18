"""One local worker, one request at a time; stdout is a framed IPC channel."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
import time


def serve(requests, channel):
    baseline = {k: v for k, v in os.environ.items() if not k.startswith('HB_') and k != 'HF_TOKEN'}
    def emit(kind, job_id, **fields):
        channel.write(json.dumps({'kind': kind, 'job_id': job_id, **fields}) + '\n')
        channel.flush()
    class Stream(io.TextIOBase):
        def __init__(self, kind, job_id): self.kind, self.job_id = kind, job_id
        def write(self, text):
            if text: emit(self.kind, self.job_id, text=text)
            return len(text)
        def flush(self): channel.flush()
    for line in requests:
        request = json.loads(line)
        job_id = request['job_id']
        script = Path(request['script']).resolve(strict=True)
        os.environ.clear(); os.environ.update(baseline); os.environ.update(request['env'])
        previous_cwd = os.getcwd()
        previous_path = sys.path[:]
        os.chdir(script.parent)
        sys.path.insert(0, str(script.parent))
        started = time.perf_counter()
        exit_code = 0
        try:
            with contextlib.redirect_stdout(Stream('stdout', job_id)), contextlib.redirect_stderr(Stream('stderr', job_id)):
                runpy.run_path(str(script), run_name='__main__')
        except SystemExit as error:
            exit_code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        except BaseException:
            emit('stderr', job_id, text=traceback.format_exc())
            exit_code = 1
        finally:
            sys.path[:] = previous_path
            os.chdir(previous_cwd)
        emit('done', job_id, exit_code=exit_code, elapsed_ms=int((time.perf_counter()-started)*1000))


if __name__ == '__main__':
    serve(sys.stdin, sys.stdout)
