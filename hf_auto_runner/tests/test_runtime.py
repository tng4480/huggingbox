import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from hf_auto_runner.artifacts import is_ready, prepare, safe_path, select_files
from hf_auto_runner.local_runtime import clear_loaded, load_once, model_file, model_source
from hf_auto_runner.runtime_router import RuntimeRouter
from hf_auto_runner.script_generator import ScriptGenerator
from hf_auto_runner.worker import serve


class ArtifactsTests(unittest.TestCase):
    def test_only_selected_quantization_and_shards(self):
        self.assertEqual(select_files(['model.Q4_K_M.gguf','model.Q8_0.gguf','unused.onnx','tokenizer.json'],'llama_cpp','model.Q4_K_M.gguf'),['model.Q4_K_M.gguf'])
        shards=['m-Q4-00001-of-00002.gguf','m-Q4-00002-of-00002.gguf']
        self.assertEqual(select_files(shards+['m-Q8.gguf'],'llama_cpp',shards[0]),shards)

    def test_transformer_download_excludes_other_formats(self):
        files=select_files(['model.safetensors','pytorch_model.bin','onnx/model.onnx','onnx/config.json','config.json','tokenizer.json'],'transformers_llm')
        self.assertEqual(files,['config.json','model.safetensors','tokenizer.json'])

    def test_unsafe_paths_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as outside:
            for name in ['../secret','/secret','C:/secret','a\\secret']:
                with self.assertRaises(ValueError): safe_path(folder,name)
            Path(folder,'link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):safe_path(folder,'link/secret')

    def test_download_pins_revision_and_marks_ready_only_after_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            info=types.SimpleNamespace(sha='fixed-revision',pipeline_tag='text-generation',tags=[],downloads=0,siblings=[types.SimpleNamespace(rfilename=n,size=2) for n in ['model.safetensors','config.json','pytorch_model.bin']])
            hub=types.ModuleType('huggingface_hub');hub.HfApi=lambda **kw:types.SimpleNamespace(model_info=lambda *a,**k:info)
            def download(model_id,name,**kwargs):
                self.assertFalse(Path(folder,'.huggingbox_complete').exists());calls.append((name,kwargs));path=Path(folder,name);path.write_bytes(b'ok');return str(path)
            hub.hf_hub_download=download
            with patch.dict(sys.modules,{'huggingface_hub':hub}),contextlib.redirect_stdout(io.StringIO()):prepare('org/model',folder,'transformers_llm')
            self.assertTrue(is_ready(folder,'transformers_llm'));self.assertEqual(len(calls),2)
            self.assertTrue(all(k['revision']=='fixed-revision' for _,k in calls))
            Path(folder,'model.safetensors').write_bytes(b'bad-size');self.assertFalse(is_ready(folder))

    def test_failed_download_leaves_no_readiness_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            info=types.SimpleNamespace(sha='fixed',siblings=[types.SimpleNamespace(rfilename='m.gguf',size=2)])
            hub=types.ModuleType('huggingface_hub');hub.HfApi=lambda **kw:types.SimpleNamespace(model_info=lambda *a,**k:info)
            def fail(*a,**k):raise OSError('interrupted')
            hub.hf_hub_download=fail
            with patch.dict(sys.modules,{'huggingface_hub':hub}),self.assertRaises(OSError):prepare('org/model',folder,'llama_cpp','m.gguf')
            self.assertFalse(is_ready(folder))

    def test_local_runtime_does_not_silently_download_missing_files(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'HB_MODEL_DIR':folder}):
            self.assertEqual(model_source('org/model'),folder)
            with self.assertRaises(FileNotFoundError):model_file('org/model','missing.gguf')
            with self.assertRaises(ValueError):model_file('org/model','../outside')


class GeneratorTests(unittest.TestCase):
    def setUp(self):clear_loaded()

    def test_all_supported_templates_generate_valid_python(self):
        cases=[('llama_cpp','text-generation'),('onnxruntime','text-classification'),('diffusers','text-to-image'),('transformers_multimodal','image-text-to-text'),('transformers_audio','automatic-speech-recognition'),('transformers_llm','text-generation'),('transformers_generic','feature-extraction')]
        for runtime,tag in cases:
            for gpu in ['cpu','cuda','amd','intel']:
                with self.subTest(runtime=runtime,gpu=gpu):
                    code=ScriptGenerator('org/model-0.5B',{'pipeline_tag':tag,'filenames':['model.Q4_K_M.gguf'],'config':{}},runtime,'Test',gpu,8_000_000_000).get_raw_script()
                    compile(code,'generated.py','exec')
                    self.assertIn('model_source(',code)

    def test_cpu_preference_overrides_detected_gpu(self):
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True),float32='f32',float16='f16')
        for backend in ['cuda','amd','intel']:
            generator=ScriptGenerator('org/model',{'config':{}},'transformers_llm','Test',backend,0)
            scope={'os':os,'torch':torch}
            with patch.dict(os.environ,{'HB_DEVICE':'cpu'}),contextlib.redirect_stdout(io.StringIO()):
                exec(generator._device_init_code(),scope)
            self.assertEqual(scope['device'],'cpu')
            self.assertEqual(scope['torch_dtype'],'f32')

    def test_diffusion_gguf_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError,'GGUF image'):
            RuntimeRouter({'pipeline_tag':'text-to-image','filenames':['model.gguf']}).get_runtime()
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            RuntimeRouter({'pipeline_tag':'image-to-video','filenames':[]}).get_runtime()

    def test_generic_pipeline_consumes_input_and_reports_result(self):
        calls=[];torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(is_available=lambda:False);torch.float32='f32'
        transformers=types.ModuleType('transformers')
        class Pipeline:
            task='text-classification'
            def __call__(self,value):calls.append(value);return [{'label':'POSITIVE','score':.9}]
        transformers.pipeline=lambda **kwargs:Pipeline()
        code=ScriptGenerator('org/model',{'pipeline_tag':'text-classification'},'transformers_generic','Bert').get_raw_script()
        output=io.StringIO()
        with patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch.dict(os.environ,{'HB_INPUT':'use this input'}),contextlib.redirect_stdout(output):exec(code,{})
        self.assertEqual(calls,['use this input']);self.assertIn('POSITIVE',output.getvalue())

    def test_question_context_input_is_parsed(self):
        clear_loaded();calls=[];torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(is_available=lambda:False);torch.float32='f32'
        transformers=types.ModuleType('transformers')
        class Pipeline:
            task='question-answering'
            def __call__(self,value):calls.append(value);return {'answer':'Paris'}
        transformers.pipeline=lambda **kwargs:Pipeline()
        code=ScriptGenerator('org/model',{'pipeline_tag':'question-answering'},'transformers_generic','Bert').get_raw_script()
        with patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch.dict(os.environ,{'HB_INPUT':'Question: Capital?\nContext: Paris is the capital.'}),contextlib.redirect_stdout(io.StringIO()):exec(code,{})
        self.assertEqual(calls,[{'question':'Capital?','context':'Paris is the capital.'}])


class WorkerTests(unittest.TestCase):
    def test_two_requests_reuse_loaders_keep_job_ids_and_clear_old_inputs(self):
        clear_loaded();cwd=os.getcwd();environment=dict(os.environ)
        factory=types.ModuleType('review_factory');factory.count=0
        def create():factory.count+=1;return 'loaded'
        factory.create=create
        try:
            with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'review_factory':factory}):
                script=Path(folder,'main.py');script.write_text('from hf_auto_runner.local_runtime import load_once\nfrom review_factory import create\nimport os\nprint(load_once(create),os.environ.get("HB_INPUT"),os.environ.get("HB_NEGATIVE_PROMPT"))\n')
                requests=[{'job_id':'first','script':str(script),'env':{'HB_INPUT':'one','HB_NEGATIVE_PROMPT':'old'}},{'job_id':'second','script':str(script),'env':{'HB_INPUT':'two'}}]
                output=io.StringIO();serve(io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)),output)
                self.assertEqual(os.getcwd(),cwd)
                events=[json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(factory.count,1)
                self.assertEqual([r['job_id'] for r in events if r['kind']=='done'],['first','second'])
                second=''.join(r.get('text','') for r in events if r['job_id']=='second')
                self.assertIn('two',second);self.assertIn('None',second);self.assertNotIn('old',second)
        finally:os.chdir(cwd);os.environ.clear();os.environ.update(environment);clear_loaded()

    def test_worker_reports_failure_and_accepts_next_request(self):
        cwd=os.getcwd();environment=dict(os.environ)
        try:
            with tempfile.TemporaryDirectory() as folder:
                script=Path(folder,'main.py');script.write_text('import os\nif os.environ["HB_INPUT"]=="bad": raise ValueError("expected error")\nprint("recovered")\n')
                requests=[{'job_id':v,'script':str(script),'env':{'HB_INPUT':v}} for v in ['bad','good']]
                output=io.StringIO();serve(io.StringIO(''.join(json.dumps(r)+'\n' for r in requests)),output)
                self.assertEqual(os.getcwd(),cwd)
                events=[json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual([r['exit_code'] for r in events if r['kind']=='done'],[1,0])
        finally:os.chdir(cwd);os.environ.clear();os.environ.update(environment)

if __name__=='__main__':unittest.main()
