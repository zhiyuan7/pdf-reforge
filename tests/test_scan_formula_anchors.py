"""Source placement and independent ordinary words must own every label correction."""
from copy import deepcopy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from book_pipeline.ingestion.scan_formula_anchors import restore_left_labels, statement_text


def word(text, x, score=.96):
    return dict(text=text, score=score, bbox=[x, 4, 35, 25])


def row(n, words):
    return dict(id=n, pixel_bbox=[0, 0, 1400, 45], words=words)


def formula(latex, number=''):
    return dict(kind='formula', latex=latex, number=number)



class ScanFormulaAnchorTests(unittest.TestCase):
    def test_physical_rows_preserve_pixels_and_revalidate_changed_crops(self):
        from PIL import Image, ImageDraw
        from book_pipeline.ingestion.scan_formula_anchors import physical_rows
        import subprocess
        with tempfile.TemporaryDirectory() as name:
            image = Image.new('RGB', (1400, 130), 'white')
            draw = ImageDraw.Draw(image)
            for y in (20, 80):
                draw.rectangle((200,y,220,y+25),fill='black')
                draw.rectangle((1100,y,1180,y+25),fill='black')
            source = Path(name)/'source.png'
            image.save(source)
            original = source.read_bytes()
            tsv = b'level\tleft\ttop\twidth\theight\tconf\ttext\n5\t200\t4\t21\t26\t96\tSince\n'
            runner = subprocess.CompletedProcess([],0,stdout=tsv)
            with patch('book_pipeline.publishing.latin_word_ocr.runtime_signature',return_value='cpu-signature'), patch(
                    'book_pipeline.ingestion.scan_formula_anchors.subprocess.run',return_value=runner) as cpu:
                rows = physical_rows(source, Path(name)/'rows')
                self.assertEqual(len(rows),2)
                self.assertEqual(cpu.call_count,2)
                self.assertEqual(source.read_bytes(),original)
                for row in rows:
                    with Image.open(row['image']) as raw:
                        self.assertEqual(raw.tobytes(),image.crop(row['pixel_bbox']).tobytes())
                self.assertEqual(physical_rows(source,Path(name)/'rows'),rows)
                self.assertEqual(cpu.call_count,2)
                Image.new('RGB',(10,10)).save(rows[0]['image'])
                self.assertEqual(physical_rows(source,Path(name)/'rows'),rows)
                self.assertEqual(cpu.call_count,4)

    def test_left_label_restored_without_changing_the_mathematical_transcription(self):
        before = dict(items=[formula(r'\text{If } u/v=1,\ u/vw=1 \text{ if } vw \text{ is not contradictory.}', '23'),
                             formula(r'uv/w=0 \text{ by (3.2).}')])
        source = row(8, [word('(23)', 200), word('If', 260), word('is', 700), word('not', 750), word('contradictory.', 800)])
        snapshot = deepcopy(before)
        output, changes = restore_left_labels(before, [source])
        self.assertEqual(before, snapshot)
        self.assertEqual(output['items'][0]['text'], r'(23) If $u/v=1,\ u/vw=1$ if $vw$ is not contradictory.')
        self.assertEqual(output['items'][1], before['items'][1])
        self.assertEqual(changes[0]['row'], 8)
        self.assertEqual(len(changes), 1)

    def test_right_low_confidence_duplicate_or_unowned_label_cannot_move_a_tag(self):
        before = dict(items=[formula(r'\text{If } x=1 \text{ then the result follows.}', '23')])
        valid = [word('(23)', 200), word('If', 260), word('result', 700), word('follows.', 800)]
        for rows in ([row(1, [word('(23)', 1000), word('result', 1100), word('follows.', 1200)])],
                     [row(1, [dict(valid[0], score=.89), *valid[1:]])],
                     [row(1, [word('(24)', 200), *valid[1:]])],
                     [row(1, [valid[0], word('unrelated', 500), word('words', 650)])],
                     [row(1, valid), row(2, valid)]):
            self.assertEqual(restore_left_labels(before, rows), (before, []))
        self.assertIsNone(statement_text(r'\text{nested \textbf{word}} x=1', '23'))
        self.assertIsNone(statement_text(r'x=1\\y=2', '23'))



if __name__ == '__main__':
    unittest.main()
