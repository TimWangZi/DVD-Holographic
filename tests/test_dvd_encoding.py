"""DVD-R framing tests against independent polynomial definitions."""
import unittest
import numpy as np
from cdholo import dvd_encoding as dvd


def gf_mul_slow(a,b):
    """Bit-polynomial multiplication; does not use encoder tables."""
    result=0
    for _ in range(8):
        if b&1: result^=a
        a<<=1
        if a&0x100: a^=0x11D
        b>>=1
    return result


def independent_syndromes(word,n):
    output=[]; root=1
    for _ in range(n):
        accum=0
        for byte in word: accum=gf_mul_slow(accum,root)^int(byte)
        output.append(accum)
        root=gf_mul_slow(root,2)
    return output


def long_division_edc(data):
    """Literal GF(2) polynomial division of all data bits, followed by 32 zeros."""
    polynomial=0x180000011
    remainder=int.from_bytes(bytes(data),'big')<<32
    while remainder.bit_length()>32:
        remainder^=polynomial<<(remainder.bit_length()-33)
    return remainder


class DvdEncodingTests(unittest.TestCase):
    def test_data_zone_header_and_edc_bit_order(self):
        rng=np.random.default_rng(901)
        payload=rng.integers(0,256,2048,dtype=np.uint8)
        frame=dvd.data_frames(payload,0x124,scrambled=False)
        self.assertEqual(bytes(frame[:4]),bytes.fromhex('00030124'))
        self.assertEqual(independent_syndromes(frame[:6],2),[0,0])
        self.assertEqual(bytes(frame[6:12]),bytes(6))
        self.assertEqual(int.from_bytes(bytes(frame[-4:]),'big'),long_division_edc(frame[:-4]))
        self.assertEqual(long_division_edc(frame),0)
        corrupted=frame.copy(); corrupted[400]^=0x01
        self.assertNotEqual(long_division_edc(corrupted),0)

    def test_scrambler_sequence_and_group_number(self):
        # Fig.26 register starts 0...0001, shifts towards r14, feeds r14 XOR r10
        # into r0; first four emitted low bytes follow directly from this diagram.
        np.testing.assert_array_equal(dvd.scramble_sequences()[0,:4],[1,0,0x22,4])
        # Table4 initial values; low byte is emitted before the first shift.
        np.testing.assert_array_equal(dvd.scramble_sequences()[:,0],
            [1,0,2,0,4,0,8,0,16,0,32,1,64,2,128,5])
        zeros=np.zeros((257,2048),np.uint8)
        encoded=dvd.scramble_payload(zeros,np.arange(257))
        np.testing.assert_array_equal(encoded[0],encoded[15])
        self.assertFalse(np.array_equal(encoded[15],encoded[16]))
        np.testing.assert_array_equal(encoded[0],encoded[256])
        np.testing.assert_array_equal(dvd.scramble_payload(encoded,np.arange(257)),zeros)

    def test_scrambling_only_payload_and_after_edc(self):
        payload=np.arange(2048,dtype=np.uint16).astype(np.uint8)
        plain=dvd.data_frames(payload,16,scrambled=False)
        coded=dvd.data_frames(payload,16)
        np.testing.assert_array_equal(plain[:12],coded[:12])
        np.testing.assert_array_equal(plain[-4:],coded[-4:])
        np.testing.assert_array_equal(dvd.scramble_payload(coded[12:2060],16),payload)
        self.assertNotEqual(long_division_edc(coded),0)

    def test_ecc_independent_column_row_checks_and_recording_order(self):
        rng=np.random.default_rng(735)
        payload=rng.integers(0,256,(16,2048),dtype=np.uint8)
        encoded=dvd.encode_ecc_block(payload,0x80)
        self.assertEqual(encoded.shape,(16,2366))
        ecc=dvd.deinterleave_recording_frames(encoded)
        self.assertEqual(ecc.shape,(208,182))
        frames=dvd.data_frames(payload,0x80+np.arange(16))
        np.testing.assert_array_equal(ecc[:192,:172],frames.reshape(192,172))
        for row in (0,11,12,191,192,207):
            self.assertEqual(independent_syndromes(ecc[row],10),[0]*10)
        for col in (0,4,12,90,171):
            self.assertEqual(independent_syndromes(ecc[:,col],16),[0]*16)
        rows=encoded.reshape(16,13,182)
        np.testing.assert_array_equal(rows[:,12,:],ecc[192:,:])
        np.testing.assert_array_equal(rows[1,:12,:],ecc[12:24,:])
        # The product code must also satisfy PO checks across the PI columns.
        self.assertEqual(independent_syndromes(ecc[:,181],16),[0]*16)

    def test_payload_spatial_offsets_are_direct_and_unique(self):
        self.assertEqual(len(np.unique(dvd.PAYLOAD_RECORDING_OFFSETS)),2048)
        self.assertEqual(dvd.PAYLOAD_CHANNEL_OFFSETS[0],32+12*16)
        # Main-data index 78 is at col90; index79 starts the second sync frame.
        self.assertEqual(dvd.PAYLOAD_CHANNEL_OFFSETS[78],32+90*16)
        self.assertEqual(dvd.PAYLOAD_CHANNEL_OFFSETS[79],1488+32)
        self.assertEqual(dvd.PAYLOAD_CHANNEL_OFFSETS[160],2976+32)
        rng=np.random.default_rng(2026)
        desired=rng.integers(0,256,(16,2048),dtype=np.uint8)
        payload=dvd.scramble_payload(desired,np.arange(16))
        encoded=dvd.encode_ecc_block(payload,0)
        np.testing.assert_array_equal(encoded[:,dvd.PAYLOAD_RECORDING_OFFSETS],desired)

    def test_batch_matches_separate_blocks_and_alignment(self):
        data=np.arange(3*16*2048,dtype=np.uint32).astype(np.uint8).reshape(3,16,2048)
        start=np.array([0,32,256])
        together=dvd.encode_ecc_blocks(data,start)
        for i in range(3):
            np.testing.assert_array_equal(together[i],dvd.encode_ecc_block(data[i],int(start[i])))
        with self.assertRaises(ValueError): dvd.encode_ecc_block(data[0],1)


if __name__=='__main__': unittest.main()
