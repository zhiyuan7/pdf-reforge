from types import SimpleNamespace
import unittest

import numpy as np

from book_pipeline.ingestion.ocr_geometry import adapt_detector


class OCRGeometryTests(unittest.TestCase):
    def test_list_and_array_boxes_have_identical_crops_and_public_results(self):
        from mineru.model.ocr.pp_ocr_v6_onnx import PPOCRv6ONNX
        image=np.arange(100*120*3,dtype=np.uint8).reshape(100,120,3)
        corners=[[[10,10],[90,10],[90,35],[10,35]],
                 [[10,10],[95,15],[90,45],[5,40]],
                 [[10,10],[35,10],[35,90],[10,90]]]
        results=[]; crops=[]
        for boxes in (corners,np.array(corners,dtype=np.float32)):
            model=PPOCRv6ONNX.__new__(PPOCRv6ONNX)
            model.is_seal=False; model.drop_score=.5; model.enable_merge_det_boxes=False
            model.text_detector=lambda im:(boxes,0.)
            current=[]
            def recognize(images):
                current.extend(images)
                return [('source text',.999) for _ in images],0.
            model.text_recognizer=recognize
            adapt_detector(model)
            detector=model.text_detector; adapt_detector(model)
            self.assertIs(model.text_detector,detector)
            results.append(model.ocr(image))
            crops.append(current)
        self.assertEqual(results[0],results[1])
        self.assertEqual(len(crops[0]),3)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(*crops)))

    def test_existing_arrays_and_empty_detections_are_passed_through(self):
        points=np.array([[[0,0],[10,0],[10,20],[0,20]]],dtype=np.int64)
        for boxes in (points,None,np.empty((0,4,2))):
            model=SimpleNamespace(text_detector=lambda im:(boxes,1.5))
            adapt_detector(model)
            result,elapsed=model.text_detector(None)
            self.assertIs(result,boxes)
            self.assertEqual(elapsed,1.5)
