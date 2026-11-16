import unittest
import numpy as np
from cdholo import dvd_modulation as m


class DvdModulationTests(unittest.TestCase):
    def test_primary_table_vectors_and_dimensions(self):
        # Direct ECMA-359 Annex G transcriptions, including an ambiguous pair.
        self.assertEqual(m.MAIN.shape,(256,4)); self.assertEqual(m.SUB.shape,(88,4))
        self.assertEqual(m.transitions(1,0),(int('0010000000001001',2),1))
        self.assertEqual(m.transitions(1,5),(int('0010000000100100',2),2))
        self.assertEqual(m.transitions(1,6),(int('0010000000100100',2),3))
        self.assertEqual(m.transitions(4,87,True),(int('0100010001000100',2),2))
        self.assertEqual(int(m.SYNC[1,6,1]),0x80440011)
        self.assertEqual(list(m.SYNC_SEQUENCE),[0,5,1,5,2,5,3,5,4,5,1,6,2,6,3,6,4,6,1,7,2,7,3,7,4,7])

    def test_all_standard_words_and_prescribed_joins(self):
        for table,nxt in [(m.MAIN,m.MAIN_NEXT),(m.SUB,m.SUB_NEXT)]:
            for row,states in zip(table,nxt):
                for w,ns in zip(row,states):
                    word=int(w); ones=[i for i,c in enumerate(f'{word:016b}') if c=='1']
                    self.assertTrue(all(3<=b-a<=11 for a,b in zip(ones,ones[1:])))
                    trailing=15-ones[-1]
                    # Every standard main/sub word leaving prescribed state must join legally.
                    for nexttable in (m.MAIN,m.SUB):
                        for future in nexttable[:,ns-1]:
                            leading=16-int(future).bit_length()
                            self.assertTrue(3<=trailing+leading+1<=11)

    def test_stateless_decode_full_ecc_block_and_sync_runs(self):
        # Independent DVD table decoder uses current word and x15/x3 of next word.
        decode={}
        for table,nxt in [(m.MAIN,m.MAIN_NEXT),(m.SUB,m.SUB_NEXT)]:
            for byte,(row,states) in enumerate(zip(table,nxt)):
                for word,ns in zip(row,states):
                    for discriminator in ([False] if ns==2 else [True] if ns==3 else [False,True]):
                        key=(int(word),discriminator)
                        if key in decode: self.assertEqual(decode[key],byte)
                        decode[key]=byte
        source=np.random.default_rng(473).integers(0,256,(16,2366),dtype=np.uint8)
        bits=m.encode_recording_frames(source)
        self.assertEqual(len(bits),16*38688)
        ones=np.flatnonzero(bits); runs=np.diff(ones)
        self.assertTrue(np.all(((runs>=3)&(runs<=11))|(runs==14)))
        self.assertEqual(np.count_nonzero(runs==14),16*26)
        np.testing.assert_array_equal(ones[1:][runs==14]%1488,27)
        frames=bits.reshape(-1,1488)
        word_values=[]; decoded=[]
        for index,frame in enumerate(frames):
            words=[int(''.join(map(str,b)),2) for b in frame[32:].reshape(91,16)]
            nextsync=int(''.join(map(str,frames[index+1,:16])),2) if index+1<len(frames) else None
            for j,word in enumerate(words):
                following=words[j+1] if j<90 else nextsync
                if following is not None: decoded.append(decode[(word,bool(following&0x8008))])
        np.testing.assert_array_equal(decoded,source.reshape(-1)[:-1])

    def test_streaming_and_polarity(self):
        source=np.random.default_rng(58).integers(0,256,(3,2366),dtype=np.uint8)
        full=m.encode_recording_frames(source)
        first,state=m.encode_recording_frames(source[:1],return_state=True)
        last=m.encode_recording_frames(source[1:],**state)
        np.testing.assert_array_equal(np.r_[first,last],full)
        np.testing.assert_array_equal(m.nrzi(full,0)^1,m.nrzi(full,1))

    def test_palette_state_cycles_do_not_drift(self):
        # 0x93 was selected instead of0x92 due to the latter's biased state2 loop.
        for byte in (0x93,0xa5):
            for state in range(1,5):
                sign=1; dsv=0; trace=[]
                for _ in range(200):
                    word,state=m.transitions(state,byte)
                    for bit in f'{word:016b}':
                        if bit=='1': sign=-sign
                        dsv+=sign
                    trace.append(dsv)
                self.assertEqual(trace[-1]-trace[-101],0)
        mixed=np.resize(np.repeat(np.array([0x93,0xa5],np.uint8),13),(16,2366))
        bits=m.encode_recording_frames(mixed); marks=m.nrzi(bits)
        dsv=np.cumsum(1-2*marks.astype(int))
        self.assertLess(np.max(np.abs(dsv)),100)


if __name__=='__main__': unittest.main()
