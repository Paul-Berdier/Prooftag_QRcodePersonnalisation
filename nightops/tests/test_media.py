import tempfile
import time
import unittest
from pathlib import Path
from PIL import Image
from qrnight.common import write, read, sha
from qrnight.report import report


class MediaTests(unittest.TestCase):
    def test_real_frames_export_without_fake_scan(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td); g = w/'generated'/'synthetic-test'
            (g/'images').mkdir(parents=True); (g/'previews').mkdir()
            for i in range(3):
                Image.new('RGB',(64,64),(40+i*20,80,90)).save(g/'previews'/f'stage2_x0_estimate_step_{i:03d}.png')
            Image.new('RGB',(64,64),(100,80,90)).save(g/'images'/'stage2-raw.png')
            row={'candidate_id':'test','prompt_id':'test','image_path':str(g/'images/stage2-raw.png'),
                 'raw':True,'visual_guard_pass':True,'wechat_original_exact':False,'wechat_exact_presets':0}
            record={'row':row,'task':{'id':'synthetic-test','method':'fixed','candidate':{'prompt_id':'test','seed':1}}}
            write(g/'result.json',record)
            cfg={'source_plan_id':'SYNTHETIC','payload':'test','max_showcase':3}
            report(w,cfg,time.time()+90)
            self.assertTrue((w/'report/animations/synthetic-test.mp4').stat().st_size>0)
            with Image.open(w/'report/animations/synthetic-test.gif') as gif:
                self.assertGreater(gif.n_frames,1)
            self.assertEqual(read(w/'report/best/manifest.json'),[])
            self.assertEqual(read(w/'report/summary.json')['phone_validation'],'NOT_TESTED')


if __name__=='__main__':unittest.main()
