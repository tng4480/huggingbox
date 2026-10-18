import json
from pathlib import Path
import tempfile
import unittest
from hf_auto_runner.classification import split_dataset, metrics

class ClassificationTests(unittest.TestCase):
    def data(self):return '\n'.join(json.dumps({'text':f'{label} example {i}','label':label}) for label in ['a','b'] for i in range(10))
    def test_stratification_and_no_overlap(self):
        splits,labels=split_dataset(self.data(),'text-classification')
        self.assertEqual(labels,['a','b']);self.assertEqual(splits,split_dataset(self.data(),'text-classification')[0])
        sets=[{r['text'] for r in rows} for rows in splits.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        for rows in splits.values():self.assertEqual({r['label'] for r in rows},{'a','b'})
    def test_duplicates_normalized_and_conflicts_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):split_dataset(self.data()+'\n'+json.dumps({'text':' A   EXAMPLE 0 ','label':'b'}),'text-classification')
    def test_insufficient_class_fails(self):
        with self.assertRaisesRegex(ValueError,'at least 6'):split_dataset(self.data()+'\n'+json.dumps({'text':'rare','label':'c'}),'text-classification')
    def test_metrics_include_missing_predictions(self):
        result=metrics(['a','a','b','b'],['a']*4,['a','b'])
        self.assertEqual(result['confusion_matrix'],[[2,0],[2,0]]);self.assertEqual(result['accuracy'],0.5);self.assertAlmostEqual(result['macro_f1'],1/3)
    def test_image_duplicates_by_content(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.png';b=Path(folder)/'b.png';a.write_bytes(b'image');b.write_bytes(b'image')
            text='\n'.join(json.dumps({'image':str(p),'label':l}) for p,l in [(a,'a'),(b,'b')])
            with self.assertRaisesRegex(ValueError,'Duplicate'):split_dataset(text,'image-classification')
    def test_bad_schema_and_missing_file(self):
        for row in [{'text':'x','label':1},{'image':'relative.png','label':'a'}]:
            with self.assertRaisesRegex(ValueError,'Line 1'):split_dataset(json.dumps(row),'image-classification' if 'image' in row else 'text-classification')

if __name__=='__main__':unittest.main()
