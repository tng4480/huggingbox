import json
import tempfile
import unittest
from unittest.mock import patch
import types
import sys
from pathlib import Path
from hf_auto_runner.lab import dataset, identity, parameters, check_lengths, save, checkpoint

class LabTests(unittest.TestCase):
    def rows(self,n=20):return '\n'.join(json.dumps({'prompt':f'Question {i}','completion':f'Answer {i}'}) for i in range(n))
    def test_split_is_reproducible_disjoint_and_seeded(self):
        a=dataset(self.rows(),42);b=dataset(self.rows(),42);self.assertEqual(a,b)
        sets=[{json.dumps(row['messages'][:-1]) for row in a[k]} for k in ('train','valid','test')]
        self.assertEqual([len(s) for s in sets],[16,2,2]);self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertNotEqual(a,dataset(self.rows(),43))
    def test_duplicate_prompts_cannot_leak_even_with_different_answers(self):
        with self.assertRaisesRegex(ValueError,'duplicate input'):dataset(self.rows()+'\n'+json.dumps({'prompt':'Question 0','completion':'Different answer'}))
    def test_invalid_rows_have_actionable_line_numbers(self):
        for row in ['{}','null','{"messages":[{"role":"user","content":"Hi"}]}','bad json']:
            with self.assertRaisesRegex(ValueError,'Line 2:'):dataset(self.rows(1)+'\n'+row)
        with self.assertRaisesRegex(ValueError,'at least 6'):dataset(self.rows(5))
    def test_settings_reject_nonfinite_and_unbounded_values(self):
        for change in [{'steps':0},{'steps':True},{'learning_rate':float('nan')},{'max_length':99999},{'backend':'unknown'}]:
            with self.assertRaises(ValueError):parameters({'backend':'torch',**change})
    def test_model_identity_changes_when_weights_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config.json').write_text('{}');(root/'model.safetensors').write_bytes(b'first')
            before=identity(folder);(root/'model.safetensors').write_bytes(b'other');self.assertNotEqual(before['sha256'],identity(folder)['sha256'])
            before=identity(folder);(root/'chat_template.jinja').write_text('new template');self.assertNotEqual(before['sha256'],identity(folder)['sha256'])
    def test_oversized_examples_are_not_silently_truncated(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kw):return 'long text'
            def encode(self,text,**kw):return list(range(200))
        with self.assertRaisesRegex(ValueError,'nothing was silently truncated'):check_lengths(Tokenizer(),dataset(self.rows()),64)
    def test_resume_skips_incomplete_checkpoint(self):
        class Weights:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def keys(self):return ['weight']
            def get_tensor(self,key):return [1]
        def open_weights(path,**kw):
            if Path(path).read_text()!='complete':raise ValueError('partial file')
            return Weights()
        fake=types.ModuleType('safetensors');fake.safe_open=open_weights
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'safetensors':fake}):
            adapter=Path(folder)/'adapter';adapter.mkdir()
            (adapter/'0000020_adapters.safetensors').write_text('partial')
            (adapter/'0000010_adapters.safetensors').write_text('complete')
            self.assertEqual(checkpoint(folder,'mlx').name,'0000010_adapters.safetensors')

    def test_atomic_json_result_is_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json';save(path,{'value':1});save(path,{'value':2});self.assertEqual(json.loads(path.read_text()),{'value':2});self.assertFalse(path.with_suffix('.json.tmp').exists())

if __name__=='__main__':unittest.main()
