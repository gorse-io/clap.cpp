"""Stdlib regression for nonfinite components after a finite first component."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import muq_mulan_checkpoint_parity as oracle

class Numerical(unittest.TestCase):
    def test_valid_metrics(self):
        metrics=oracle.comparison_metrics([1.,0.],[1.,0.])
        self.assertEqual(metrics['max_abs_error'],0.)
        self.assertEqual(metrics['cosine'],1.)

    def test_published_coordinator_rejects_nonfinite(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for reference_bad in (False,True):
                with self.subTest(bad=bad,reference_bad=reference_bad),tempfile.TemporaryDirectory() as tmp:
                    tmp=Path(tmp);(tmp/'pytorch_model.bin').write_bytes(b'x')
                    native=[1.]+[0.]*511;reference=native.copy()
                    (reference if reference_bad else native)[1]=bad
                    (tmp/'text-reference.json').write_text(json.dumps([{'text':'a','expected':reference}]))
                    (tmp/'audio-reference.json').write_text(json.dumps({'wave':'unused','expected':reference}))
                    args=SimpleNamespace(out=tmp,checkpoint=tmp,branch=None,cli=tmp/'cli',gguf=tmp/'model',audio=None,tolerance=1e-4)
                    with patch.object(oracle,'SIZE',1),patch.object(oracle,'SHA256',hashlib.sha256(b'x').hexdigest()),patch.object(oracle,'parse_args',return_value=args),patch.object(oracle.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps(native))):
                        with self.assertRaises(ValueError):oracle.main()

if __name__=='__main__': unittest.main()
