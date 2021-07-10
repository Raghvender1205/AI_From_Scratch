import unittest
import torch
import gzip

import transformers as trf

from former import util
from former.util import slice_diag, compute_compression

class TestUtil(unittest.TestCase):
    def test_slice_diagnol(self):
        m = torch.randint(high=20, size=(16, 24, 3, 5, 9))

        print(m[0, 0, 0])
        print(slice_diag(m[0, 0, 0], l=5))

if __name__ == '__main__':
    unittest.main()