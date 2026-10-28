"""Reviewed prose edits cannot consume mathematics or overlap each other."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from book_pipeline.config import fingerprint
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.text_corrections import validate_corrections


class TextCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.text = r'\text{Since } x/y=1 \quad \text{by (2.3).}'
        self.index = SourceIndex(dict(atoms=[dict(id='a', group='g', page=0,
            kind='display_math', content=self.text, length=len(self.text))]))

    def correction(self, expected, replacement, **changes):
        start = self.text.index(expected)
        return dict(atom='a', start_offset=start, end_offset=start+len(expected),
                    expected=expected, replacement=replacement, **changes)

    def validate(self, *items, source=None):
        return validate_corrections(self.index, dict(
            source_sha256=fingerprint(self.index.source) if source is None else source, items=list(items)))

    def test_source_bound_complete_labels_keep_math_and_input_verbatim(self):
        rows = [self.correction('by (2.3).', 'by (2.8).'), self.correction('Since ', 'Therefore ')]
        original = deepcopy(rows)
        accepted = self.validate(*rows)
        # Exercise the renderer's actual reverse-offset application, including
        # a length change before a second label and source-atom boundaries.
        with tempfile.TemporaryDirectory() as folder:
            renderer=Renderer(self.index,dict(corrections=accepted,joins=[],exclusions=[],bold_spans=[]),
                              [],[],Path(folder),{})
            result=renderer.content_fragment(self.index.atoms[0],0,len(self.text),False)
        self.assertIn(r'\text{Therefore } x/y=1 \quad \text{by (2.8).}',result)
        self.assertEqual(rows,original)
        self.assertEqual(self.index.atoms[0]['content'],self.text)
        accepted[0]['replacement']='changed'
        self.assertEqual(rows,original)

    def test_cross_label_or_partial_label_cannot_erase_formula(self):
        for expected in ('Since } x/y=1 \\quad \\text{by (2.3).', 'by (2.3)', 'x/y'):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError,'complete plain text label'):
                self.validate(self.correction(expected,'one label'))

    def test_raw_tex_syntax_cannot_change_math_outside_label(self):
        for replacement in ('new}x{', r'\frac{x}{y}', '$x$', 'a%b', 'a&b', 'a#b', 'a_b', 'a^b', 'a~b'):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError,'complete plain text label'):
                self.validate(self.correction('by (2.3).',replacement))

    def test_duplicate_nested_and_partial_overlap_rejected_but_adjacent_text_edits_allowed(self):
        self.text='abcdef'
        self.index.atoms[0].update(kind='text',content=self.text,length=len(self.text))
        for left,right in (('abc','abc'),('abcd','bc'),('abcd','cdef')):
            with self.subTest(left=left,right=right), self.assertRaisesRegex(ValueError,'overlap'):
                self.validate(self.correction(left,'x'),self.correction(right,'y'))
        self.assertEqual(len(self.validate(self.correction('abc','x'),self.correction('def','y'))),2)

    def test_stale_source_mismatched_text_unknown_atom_and_invalid_offsets_fail_closed(self):
        valid=self.correction('by (2.3).','by (2.8).')
        with self.assertRaisesRegex(ValueError,'edited source'):self.validate(valid,source='stale')
        for changes in (dict(atom='unknown'),dict(start_offset=True),dict(start_offset=1.0),
                        dict(end_offset=None),dict(expected='different'),dict(replacement=None)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.validate({**valid,**changes})
        for kind in ('image','table'):
            self.index.atoms[0]['kind']=kind
            with self.assertRaises(ValueError):self.validate(valid)


if __name__ == '__main__':
    unittest.main()
