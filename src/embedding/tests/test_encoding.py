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

    def test_epsilon_to_decimals(self) -> None:
        self.assertEqual(0, encoding.epsilon_to_decimals(10))
        self.assertEqual(1, encoding.epsilon_to_decimals(0.9))
        self.assertEqual(1, encoding.epsilon_to_decimals(0.1))
        self.assertEqual(2, encoding.epsilon_to_decimals(0.01))
        self.assertEqual(3, encoding.epsilon_to_decimals(0.001))
        self.assertEqual(4, encoding.epsilon_to_decimals(0.0001))
        self.assertEqual(5, encoding.epsilon_to_decimals(0.00001))
        self.assertEqual(6, encoding.epsilon_to_decimals(0.000001))

    def test_roundtrip_exact(self) -> None:
        data = np.arange(1235, dtype=np.float32)
        stream = encoding.ApproximatedStream(0.1, data)
        encoded = stream.tobytes()
        newstream, offset = encoding.ApproximatedStream.from_bytes(encoded, 0)
        self.assertEqual(offset, len(encoded))
        decodeddata = newstream.stream
        self.assertEqual(newstream.decimals, stream.decimals)
        self.assertTrue(np.all(data == decodeddata))
