import unittest
from embedding import encoding
import numpy as np
import os


class EncodingTestCase(unittest.TestCase):
    def test_neats_exists(self) -> None:
        self.assertTrue(os.path.exists(encoding.neats_dir))
        self.assertTrue(os.path.exists(encoding.neats_compress))
        self.assertTrue(os.path.exists(encoding.neats_decompress))

    def test_tempfilename(self) -> None:
        with encoding.temp_filename() as f:
            fname = f.name
            f.write(bytes([1, 2, 3, 4]))
            self.assertTrue(os.path.exists(fname))
            f.close()
            self.assertTrue(os.path.exists(fname))
        self.assertFalse(os.path.exists(fname))

    @unittest.expectedFailure
    def test_roundtrip_exact(self) -> None:
        data = np.arange(1235, dtype=np.float32)
        stream = encoding.ApproximatedStream(data, 0.1)
        encoded = stream.tobytes(verbose=True)
        newstream, offset = encoding.ApproximatedStream.from_bytes(
            encoded, verbose=True
        )
        self.assertEqual(offset, len(encoded))
        decodeddata = newstream.stream
        print(f"data: {data}")
        print(f"decoded: {decodeddata}")
        print(f"diff: {decodeddata - data}")
        print(f"maxdiff: {np.max(np.fabs(decodeddata - data))}")
        print(f"non-zero indices: {np.nonzero(decodeddata - data)}")
        self.assertTrue(np.all(data == decodeddata))
