import unittest
from embedding import encoding, util
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

    def test_packbits(self) -> None:
        # pack 32 values in [0,15] in 4 bits, which should save half the space
        values = np.array([np.arange(16), np.flip(np.arange(16))])
        packed = encoding.packmultibits(values, 4)
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(len(packed), np.prod(values.shape) // 2)
        unpacked = encoding.unpackmultibits(packed, 4, int(np.prod(values.shape)))
        reshaped = unpacked.reshape(values.shape)
        self.assertTrue(np.all(values == reshaped))

        # pack 32 values in [0,7] in 3 bits, which should be 21 values per 64-bit
        # so 2 words should be enough.
        packed = encoding.packmultibits(values // 2, 3)
        self.assertTrue(packed.dtype == np.uint64)
        self.assertEqual(len(packed), 2)
        unpacked = encoding.unpackmultibits(packed, 3, int(np.prod(values.shape)))
        reshaped = unpacked.reshape(values.shape)
        self.assertTrue(np.all(values // 2 == reshaped))

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

    def test_encode_sparse(self) -> None:
        A = np.arange(1, 101).reshape((10, 10)).astype(np.float32)
        counts = np.arange(10) // 2
        B = util.zero_jagged(A, counts)

        sparse_a_bytes = encoding.encode_sparse_matrix(A, counts, 0.25)
        decodedB, decodedCounts, offset = encoding.decode_sparse_matrix(
            sparse_a_bytes, 0, nrows=10, dtype=A.dtype
        )
        self.assertEqual(offset, len(sparse_a_bytes))
        self.assertEqual(B.shape, decodedB.shape)
        self.assertTrue(np.allclose(B, decodedB, rtol=0, atol=0.25))
        self.assertEqual(A.dtype, decodedB.dtype)
