"""Reference-structure checks, not tests that merely repeat encoder branches."""
import unittest
import numpy as np

from cdholo.encoding import (DATA_POSITIONS, DATA_DELAYS, EFM, EFM_TRANSITIONS,
    encode_circ, encode_channel, f1_to_pcm_le, pcm_le_to_f1,
    nrzi, mark_runs, precompensate_systematic, subcode_q)


def gf_multiply(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 256:
            a ^= 0x11D
        b >>= 1
    return result


def polynomial_checks(words):
    """Horner evaluation independent of the implementation's GF lookup tables."""
    for row in words:
        for root in (1, 2, 4, 8):
            value = 0
            for byte in row:
                value = gf_multiply(value, root) ^ int(byte)
            if value:
                return False
    return True


class EncodingTests(unittest.TestCase):
    def test_efm_standard_table(self):
        self.assertEqual(len({tuple(w) for w in EFM}), 256)
        for byte, expected in {0: "01001000100000", 13: "00000001000000",
                               170: "10010001001001", 255: "00100000010010"}.items():
            self.assertEqual("".join(map(str, EFM[byte])), expected)
        self.assertEqual(int(EFM_TRANSITIONS[13]), 1)
        self.assertEqual(int(EFM_TRANSITIONS[170]), 5)
        for word in EFM:
            distances = np.diff(np.flatnonzero(word))
            self.assertTrue(np.all((distances >= 3) & (distances <= 11)))

    def test_c4_golden_systematic_routes(self):
        # Explicit transcription from ECMA-130 Figure C.4, output slot order.
        source = [0, 1, 8, 9, 16, 17, 2, 3, 10, 11, 18, 19,
                  4, 5, 12, 13, 20, 21, 6, 7, 14, 15, 22, 23]
        slot = list(range(12)) + list(range(16, 28))
        delays = [3, 6, 11, 14, 19, 22, 27, 30, 35, 38, 43, 46,
                  65, 68, 73, 76, 81, 84, 89, 92, 97, 100, 105, 108]
        data = np.random.default_rng(15).integers(0, 256, (256, 24), dtype=np.uint8)
        result = encode_circ(data)
        for src, out, delay in zip(source, slot, delays):
            np.testing.assert_array_equal(result[delay:, out], data[:-delay, src])
            self.assertEqual(int(DATA_POSITIONS[src]), out)
            self.assertEqual(int(DATA_DELAYS[src]), delay)

    def test_c1_c2_reed_solomon_checks(self):
        data = np.random.default_rng(18).integers(0, 256, (400, 24), dtype=np.uint8)
        f2 = encode_circ(data)
        f2[:, 12:16] ^= 255
        f2[:, 28:] ^= 255
        # Undo final alternating delay and then main 4*j delay without
        # using any encoder permutation or parity-coefficient helper.
        c1 = f2[:-1].copy()
        c1[:, ::2] = f2[1:, ::2]
        self.assertTrue(polynomial_checks(c1[110:140]))
        c2 = np.column_stack([c1[4*j:4*j+200, j] for j in range(28)])
        self.assertTrue(polynomial_checks(c2[10:40]))
        c2[10, 7] ^= 1
        self.assertFalse(polynomial_checks(c2[10:11]))

    def test_systematic_inverse_and_streaming(self):
        rng = np.random.default_rng(110)
        desired = rng.integers(0, 256, (400, 32), dtype=np.uint8)
        data = precompensate_systematic(desired)
        result = encode_circ(data)
        np.testing.assert_array_equal(result[108:, DATA_POSITIONS], desired[108:, DATA_POSITIONS])
        continued = encode_circ(data[200:], history=data[91:200])
        np.testing.assert_array_equal(result[200:], continued)

    def test_channel_constraints_and_symbol_identity(self):
        rng = np.random.default_rng(123)
        f2 = rng.integers(0, 256, (196, 32), dtype=np.uint8)
        channel = encode_channel(f2)
        self.assertEqual(channel.shape, (196 * 588,))
        distances = np.diff(np.flatnonzero(channel))
        self.assertTrue(np.all((distances >= 3) & (distances <= 11)))
        text = "".join(map(str, channel))
        sync = "100000000001000000000010"
        positions = []
        pos = text.find(sync)
        while pos != -1:
            positions.append(pos)
            pos = text.find(sync, pos + 1)
        self.assertEqual(positions, list(range(0, len(channel), 588)))
        rows = channel.reshape(-1, 588)
        for j in range(32):
            np.testing.assert_array_equal(rows[:, 44 + 17*j:58 + 17*j], EFM[f2[:, j]])
        # Segmenting the same stream must preserve all transition bits.
        first, state = encode_channel(f2[:98], return_state=True)
        second = encode_channel(f2[98:], **state)
        np.testing.assert_array_equal(channel, np.r_[first, second])

    def test_byte_order_nrzi_and_runs(self):
        payload = bytes(range(24))
        f1 = pcm_le_to_f1(payload)
        self.assertEqual(f1[0, :4].tolist(), [1, 0, 3, 2])
        self.assertEqual(f1_to_pcm_le(f1), payload)
        bits = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8)
        self.assertEqual(nrzi(bits).tolist(), [1, 1, 1, 0, 0, 0, 0, 1])
        self.assertEqual(mark_runs(bits).tolist(), [[0, 3, 1], [3, 4, 0], [7, 1, 1]])

    def test_q_crc(self):
        # First track INDEX 01, relative 00:00:00, absolute 00:02:00.
        q = subcode_q(0)
        self.assertEqual(q[:10], (1, 1, 1, 0, 0, 0, 0, 0, 2, 0))
        # Polynomial long division of 80 data bits with 16 zero bits.
        dividend = int.from_bytes(bytes(q[:10]), "big") << 16
        polynomial = (1 << 16) | (1 << 12) | (1 << 5) | 1
        for bit in range(95, 15, -1):
            if dividend & (1 << bit):
                dividend ^= polynomial << (bit - 16)
        self.assertEqual((q[10] << 8) | q[11], dividend ^ 0xFFFF)


if __name__ == "__main__":
    unittest.main()
