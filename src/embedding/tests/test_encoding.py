import unittest
from embedding import encoding
import numpy as np
import os

from embedding.tests.test_data import complex_data

from typing import Optional


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

    def _test_codec(
        self, codec, data: Optional[np.ndarray] = None, quality: float = 0.1
    ) -> None:
        if data is None:
            data = complex_data().flatten()
        b = codec.tobytes(data, quality, verbose=False)
        self.assertTrue(b is not None)
        after = codec.from_bytes(b, dtype=np.float32, verbose=False)
        self.assertEqual(len(after), len(data))
        self.assertTrue(np.allclose(data, after, rtol=0, atol=quality))

    def test_raw_codec(self) -> None:
        self._test_codec(encoding.RawCodec)

        # raw is lossless
        data = np.arange(1235, dtype=np.float32)
        encoded = encoding.RawCodec.tobytes(data, 0.1, verbose=False)
        self.assertFalse(encoded is None)
        assert encoded is not None
        decoded = encoding.RawCodec.from_bytes(encoded, dtype=np.float32, verbose=False)
        self.assertTrue(np.all(data == decoded))

    @unittest.expectedFailure
    def test_neats_codec(self) -> None:
        self._test_codec(encoding.NeatsCodec)

        # can't encode float64 (as of now)
        data = np.array(np.arange(17), dtype=np.float64)
        encoded = encoding.NeatsCodec.tobytes(data, 0.1, verbose=False)
        self.assertTrue(encoded is None)

        # can't encode empty data
        data = np.array([])
        encoded = encoding.NeatsCodec.tobytes(data, 0.1, verbose=False)
        self.assertTrue(encoded is None)

    def test_intify_codec(self) -> None:
        self._test_codec(encoding.IntifyCodec)

    def test_encode_coordinates(self) -> None:
        data = np.arange(7 * 8 * 2, dtype=np.float32)
        shape1 = (7 * 8 * 2,)
        encoded = encoding.encode_coordinates(data, quality=0.1)
        decoded, offset = encoding.decode_coordinates(
            encoded, offset=0, shape=shape1, dtype=np.float32
        )
        self.assertEqual(offset, len(encoded))
        self.assertEqual(decoded.shape, shape1)
        self.assertTrue(np.allclose(data, decoded, rtol=0, atol=0.1))

        shape2 = (7 * 8, 2)
        data = data.reshape(shape2)
        encoded = encoding.encode_coordinates(data, quality=0.1)
        decoded, offset = encoding.decode_coordinates(
            encoded, offset=0, shape=shape2, dtype=np.float32
        )
        self.assertEqual(offset, len(encoded))
        self.assertEqual(decoded.shape, shape2)
        self.assertTrue(np.allclose(data, decoded, rtol=0, atol=0.1))

        shape3 = (7, 8, 2)
        data = data.reshape(shape3)
        encoded = encoding.encode_coordinates(data, quality=0.1)
        decoded, offset = encoding.decode_coordinates(
            encoded, offset=0, shape=shape3, dtype=np.float32
        )
        self.assertEqual(offset, len(encoded))
        self.assertEqual(decoded.shape, shape3)
        self.assertTrue(np.allclose(data, decoded, rtol=0, atol=0.1))
